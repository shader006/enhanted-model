#!/usr/bin/env python3
"""Benchmark NumpyToTensor and nearby label conversion choices.

Run:
  conda run -n brats23 python enhanted-model/mock_test/benchmark_numpy_to_tensor.py
  conda run -n brats23 python enhanted-model/mock_test/benchmark_numpy_to_tensor.py --device cuda:0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from batchgenerators.transforms.utility_transforms import NumpyToTensor


def clone_np_batch(batch: dict) -> dict:
    return {
        "data": batch["data"].copy(),
        "seg": batch["seg"].copy(),
        "properties": batch["properties"],
        "keys": batch["keys"],
    }


def current_numpy_to_tensor(batch: dict) -> dict:
    transform = NumpyToTensor(["data", "seg"], "float")
    return transform(**batch)


def from_numpy_no_cast(batch: dict) -> dict:
    batch["data"] = torch.from_numpy(batch["data"]).contiguous()
    batch["seg"] = torch.from_numpy(batch["seg"]).contiguous()
    return batch


def from_numpy_data_float_seg_long(batch: dict) -> dict:
    batch["data"] = torch.from_numpy(batch["data"]).float().contiguous()
    batch["seg"] = torch.from_numpy(batch["seg"]).long().contiguous()
    return batch


def from_numpy_data_float_seg_int16(batch: dict) -> dict:
    batch["data"] = torch.from_numpy(batch["data"]).float().contiguous()
    batch["seg"] = torch.from_numpy(batch["seg"]).short().contiguous()
    return batch


def get_input_like(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    image = batch["data"]
    label = batch["seg"]
    label = label[:, 0].long()
    label.clamp_(0, 3)
    return image, label


def to_device_like(batch: dict, device: str) -> dict:
    for key in ("data", "seg"):
        batch[key] = batch[key].to(device).contiguous()
    return batch


def bench(name: str, fn, iterations: int, warmup: int) -> dict:
    for _ in range(warmup):
        obj = fn()
        del obj
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        obj = fn()
        del obj
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000 / iterations
    print(f"{name:<42} {ms:>9.4f} ms/iter")
    return {"label": name, "ms": ms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    b = args.batch_size
    d, h, w = args.patch_size
    rng = np.random.default_rng(42)
    data = rng.normal(size=(b, 4, d, h, w)).astype(np.float32)
    seg = rng.integers(-1, 5, size=(b, 1, d, h, w), dtype=np.int16).astype(np.float32)
    base_batch = {
        "data": data,
        "seg": seg,
        "properties": [{} for _ in range(b)],
        "keys": np.arange(b),
    }

    print(f"batch_size={b}, patch_size={args.patch_size}, device={args.device}")
    print(f"data dtype={base_batch['data'].dtype}, seg dtype={base_batch['seg'].dtype}")
    print()

    bench(
        "benchmark overhead: clone np batch only",
        lambda: clone_np_batch(base_batch),
        args.iterations,
        args.warmup,
    )
    bench(
        "NumpyToTensor current float(data+seg)",
        lambda: current_numpy_to_tensor(clone_np_batch(base_batch)),
        args.iterations,
        args.warmup,
    )
    bench(
        "torch.from_numpy no cast",
        lambda: from_numpy_no_cast(clone_np_batch(base_batch)),
        args.iterations,
        args.warmup,
    )
    bench(
        "custom data float + seg long",
        lambda: from_numpy_data_float_seg_long(clone_np_batch(base_batch)),
        args.iterations,
        args.warmup,
    )
    bench(
        "custom data float + seg int16",
        lambda: from_numpy_data_float_seg_int16(clone_np_batch(base_batch)),
        args.iterations,
        args.warmup,
    )
    bench(
        "current NumpyToTensor + get_input",
        lambda: get_input_like(current_numpy_to_tensor(clone_np_batch(base_batch))),
        args.iterations,
        args.warmup,
    )
    bench(
        "custom seg long + get_input",
        lambda: get_input_like(from_numpy_data_float_seg_long(clone_np_batch(base_batch))),
        args.iterations,
        args.warmup,
    )

    if args.device != "cpu":
        if not torch.cuda.is_available():
            print("\nCUDA is not available; skipping device transfer benchmarks.")
            return 0
        print()
        bench(
            "current tensor + to_device + get_input",
            lambda: get_input_like(to_device_like(current_numpy_to_tensor(clone_np_batch(base_batch)), args.device)),
            max(args.iterations // 10, 20),
            max(args.warmup // 10, 5),
        )
        bench(
            "custom seg long + to_device + get_input",
            lambda: get_input_like(to_device_like(from_numpy_data_float_seg_long(clone_np_batch(base_batch)), args.device)),
            max(args.iterations // 10, 20),
            max(args.warmup // 10, 5),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
