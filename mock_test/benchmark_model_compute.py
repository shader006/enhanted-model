#!/usr/bin/env python3
"""Benchmark SegMamba model compute on a GPU node.

This script intentionally imports the model the same way 3_train.py does:
project root first, advanced_model second. That avoids accidentally importing
the vendored MONAI package before the project package layout is ready.

Examples:
  conda run -n brats23 python enhanted-model/mock_test/benchmark_model_compute.py --device cuda:0
  conda run -n brats23 python enhanted-model/mock_test/benchmark_model_compute.py --device cuda:0 --amp bf16
  conda run -n brats23 python enhanted-model/mock_test/benchmark_model_compute.py --device cuda:0 --mamba-stages 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "enhanted-model"
ADVANCED_MODEL_DIR = PROJECT_ROOT / "advanced_model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ADVANCED_MODEL_DIR) not in sys.path:
    sys.path.append(str(ADVANCED_MODEL_DIR))

from model_segmamba.segmamba import SegMamba  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--in-chans", type=int, default=4)
    parser.add_argument("--out-chans", type=int, default=4)
    parser.add_argument("--depths", type=int, nargs=4, default=[1, 1, 1, 1])
    parser.add_argument("--feat-size", type=int, nargs=4, default=[48, 96, 192, 384])
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--mamba-stages", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mode", choices=["forward", "train"], default="train")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def amp_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def sync(device: str):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_model(args):
    return SegMamba(
        use_settings=False,
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        depths=args.depths,
        feat_size=args.feat_size,
        hidden_size=args.hidden_size,
        input_size=args.input_size,
        mamba_stages=args.mamba_stages,
    )


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment. Run this on a GPU node.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    model = make_model(args).to(device)
    if args.compile:
        model = torch.compile(model)

    x = torch.randn(args.batch_size, args.in_chans, *args.input_size, device=device)
    target = torch.randint(0, args.out_chans, (args.batch_size, *args.input_size), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.mode == "train" else None
    loss_fn = torch.nn.CrossEntropyLoss()
    enabled = args.amp != "off"
    dtype = amp_dtype(args.amp)

    def step():
        if args.mode == "train":
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                y = model(x)
                loss = loss_fn(y, target)
            loss.backward()
            optimizer.step()
            return loss
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                return model(x)

    model.train(args.mode == "train")
    for _ in range(args.warmup):
        step()
    sync(args.device)

    start_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    start = time.perf_counter()
    for _ in range(args.iterations):
        step()
    sync(args.device)
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    result = {
        "mode": args.mode,
        "device": args.device,
        "amp": args.amp,
        "compile": args.compile,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "depths": args.depths,
        "feat_size": args.feat_size,
        "hidden_size": args.hidden_size,
        "mamba_stages": args.mamba_stages,
        "params": count_params(model),
        "iterations": args.iterations,
        "mean_ms": elapsed * 1000 / args.iterations,
        "peak_memory_mb": peak_mem / 1024 / 1024,
        "start_memory_mb": start_mem / 1024 / 1024,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
