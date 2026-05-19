import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != BASE_DIR
]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(BASE_DIR)

import settings  # noqa: E402
from model_segmamba.segmamba import SegMamba  # noqa: E402


def count_params(module, trainable=None):
    params = module.parameters()
    if trainable is True:
        params = (p for p in params if p.requires_grad)
    elif trainable is False:
        params = (p for p in params if not p.requires_grad)
    return sum(p.numel() for p in params)


def count_unique_params(module):
    return sum(p.numel() for p in {id(p): p for p in module.parameters()}.values())


def format_count(value):
    return f"{value:,} ({value / 1e6:.3f}M)"


def print_param_summary(model):
    total = count_params(model)
    trainable = count_params(model, trainable=True)
    frozen = count_params(model, trainable=False)
    unique = count_unique_params(model)

    print("Parameter count")
    print(f"  total:     {format_count(total)}")
    print(f"  trainable: {format_count(trainable)}")
    print(f"  frozen:    {format_count(frozen)}")
    print(f"  unique parameter objects: {format_count(unique)}")
    if unique != total:
        print("  warning: duplicated parameter objects detected; total may double-count shared weights.")

    print("\nTop-level parameter split")
    for name, child in model.named_children():
        n = count_params(child)
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {name:24s} {n:14,d} {n / 1e6:9.3f}M {pct:6.2f}%")


def module_param_table(model, min_params=1):
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own >= min_params:
            rows.append((own, name, module.__class__.__name__))
    rows.sort(reverse=True)
    return rows


def print_largest_leaf_params(model, limit=40):
    print(f"\nLargest modules by own parameters (top {limit})")
    for own, name, cls_name in module_param_table(model)[:limit]:
        print(f"  {own:12,d} {own / 1e6:8.3f}M  {name:70s} {cls_name}")


def collect_custom_param_modules(model):
    known = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.Embedding,
    )
    rows = []
    by_type = defaultdict(int)
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own and not isinstance(module, known):
            cls_name = module.__class__.__name__
            rows.append((own, name, cls_name))
            by_type[cls_name] += own
    rows.sort(reverse=True)
    return rows, by_type


def print_flop_caveats(model):
    rows, by_type = collect_custom_param_modules(model)
    if not rows:
        return

    print("\nCustom parameterized modules")
    print("  These params are included in the parameter count, but vanilla THOP may miss or undercount their FLOPs.")
    for cls_name, n in sorted(by_type.items(), key=lambda item: item[1], reverse=True):
        print(f"  {cls_name:32s} {n:14,d} {n / 1e6:9.3f}M")


def try_thop_profile(model, input_shape, device):
    try:
        from thop import profile
    except ModuleNotFoundError:
        print("\nFLOPs")
        print("  thop is not installed in this environment.")
        return

    model = model.to(device).eval()
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)

    print("\nTHOP profile")
    print("  Note: THOP reports MACs. Many papers report FLOPs = 2 * MACs.")
    print(f"  input shape: {tuple(input_shape)}")
    print(f"  params from THOP: {format_count(int(params))}")
    print(f"  MACs lower-bound: {macs / 1e9:.3f} GMACs")
    print(f"  FLOPs if 1 MAC = 2 FLOPs: {2 * macs / 1e9:.3f} GFLOPs")
    print("  Caveat: custom ops such as Mamba selective scan, KAN/SKAN functional linear, grid_sample,")
    print("          softmax, indexing/window ops, and Onsampling neighbor sampling are not fully covered by vanilla THOP.")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit SegMamba parameter and FLOP accounting.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, nargs=3, default=list(settings.INPUT_SIZE))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--largest", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Effective settings")
    for name in (
        "INPUT_SIZE",
        "SEGMAMBA_IN_CHANS",
        "SEGMAMBA_OUT_CHANS",
        "SEGMAMBA_DEPTHS",
        "SEGMAMBA_FEAT_SIZE",
        "SEGMAMBA_HIDDEN_SIZE",
        "SEGMAMBA_KAN",
        "SEGMAMBA_SKAN",
        "SEGMAMBA_ONSAMPLING",
    ):
        print(f"  {name}: {getattr(settings, name)}")

    model = SegMamba()
    print_param_summary(model)
    print_largest_leaf_params(model, limit=args.largest)
    print_flop_caveats(model)

    if not args.skip_flops:
        input_shape = (args.batch_size, settings.SEGMAMBA_IN_CHANS, *args.input_size)
        try_thop_profile(model, input_shape, args.device)


if __name__ == "__main__":
    main()
