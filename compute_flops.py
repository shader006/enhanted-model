import argparse
import os
import sys

# Profiling does not need deterministic kernels, but if settings enable them
# CUDA may require this variable to be present before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import settings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ADVANCED_MODEL_DIR = os.path.join(PROJECT_ROOT, "advanced_model")
if ADVANCED_MODEL_DIR not in sys.path:
    sys.path.insert(0, ADVANCED_MODEL_DIR)

from model_segmamba.segmamba import SegMamba


def format_count(value: float, unit: float, suffix: str) -> str:
    return f"{value / unit:.4f} {suffix}"


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_window_size(window_size_arg: str | None) -> tuple[int, int, int]:
    if not window_size_arg:
        return tuple(int(v) for v in settings.SEGMAMBA_KAN_Z_WINDOW_SIZE)
    cleaned = window_size_arg.replace("x", ",").replace("X", ",")
    values = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Window size must contain exactly 3 integers, for example 4,4,4.")
    return tuple(int(v) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Params/MACs/FLOPs for SegMamba variants.")
    parser.add_argument(
        "--device",
        default=None,
        help='Device for model profiling, for example "cuda", "cuda:0", or "cpu". Defaults to CUDA if available.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for profiling.",
    )
    parser.add_argument(
        "--compare-kan",
        action="store_true",
        help="Profile three variants together: baseline, KAN, and KAN + Z-window.",
    )
    parser.add_argument(
        "--kan",
        dest="kan",
        action="store_true",
        default=None,
        help="Enable KAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--no-kan",
        dest="kan",
        action="store_false",
        help="Disable KAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--z-window",
        dest="z_window",
        action="store_true",
        default=None,
        help="Enable KAN Z-window for single-variant profiling. Implies --kan.",
    )
    parser.add_argument(
        "--no-z-window",
        dest="z_window",
        action="store_false",
        help="Disable KAN Z-window for single-variant profiling.",
    )
    parser.add_argument(
        "--z-window-size",
        default=None,
        help="Override Z-window size as d,h,w or dxhxw. Defaults to settings.SEGMAMBA_KAN_Z_WINDOW_SIZE.",
    )
    return parser.parse_args()


def build_variant_specs(args: argparse.Namespace) -> list[dict]:
    z_window_size = parse_window_size(args.z_window_size)
    if args.compare_kan:
        return [
            {"name": "baseline", "kan_enabled": False, "kan_z_window_enabled": False, "kan_z_window_size": z_window_size},
            {"name": "kan", "kan_enabled": True, "kan_z_window_enabled": False, "kan_z_window_size": z_window_size},
            {"name": "kan_z_window", "kan_enabled": True, "kan_z_window_enabled": True, "kan_z_window_size": z_window_size},
        ]

    kan_enabled = settings.SEGMAMBA_KAN if args.kan is None else bool(args.kan)
    z_window_enabled = settings.SEGMAMBA_KAN_Z_WINDOW if args.z_window is None else bool(args.z_window)
    if z_window_enabled:
        kan_enabled = True
    return [
        {
            "name": "current_settings",
            "kan_enabled": kan_enabled,
            "kan_z_window_enabled": z_window_enabled,
            "kan_z_window_size": z_window_size,
        }
    ]


def instantiate_model(spec: dict, device: torch.device) -> torch.nn.Module:
    return SegMamba(
        kan_enabled=spec["kan_enabled"],
        kan_z_window_enabled=spec["kan_z_window_enabled"],
        kan_z_window_size=spec["kan_z_window_size"],
    ).to(device)


def profile_variant(spec: dict, dummy_input: torch.Tensor, device: torch.device, profile_fn) -> dict:
    model = instantiate_model(spec, device)
    model.eval()
    with torch.no_grad():
        macs, params = profile_fn(model, inputs=(dummy_input,), verbose=False)
    model_cpu = model.to("cpu")
    del model_cpu
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": spec["name"],
        "kan_enabled": spec["kan_enabled"],
        "kan_z_window_enabled": spec["kan_z_window_enabled"],
        "kan_z_window_size": spec["kan_z_window_size"],
        "macs": float(macs),
        "params": float(params),
        "flops": float(macs) * 2.0,
    }


def print_variant_result(result: dict, input_shape: tuple[int, ...], device: torch.device) -> None:
    print(f"Variant: {result['name']}")
    print(
        "  Config: "
        f"KAN={result['kan_enabled']}, "
        f"Z-window={result['kan_z_window_enabled']}, "
        f"window_size={result['kan_z_window_size']}"
    )
    print(f"  Device: {device}")
    print(f"  Input shape: {input_shape}")
    print(f"  Params: {int(result['params']):,} ({format_count(result['params'], 1e6, 'M')})")
    print(f"  MACs: {int(result['macs']):,} ({format_count(result['macs'], 1e9, 'G')})")
    print(f"  FLOPs: {int(result['flops']):,} ({format_count(result['flops'], 1e9, 'G')})")


def print_delta_result(base: dict, target: dict) -> None:
    delta_params = target["params"] - base["params"]
    delta_macs = target["macs"] - base["macs"]
    delta_flops = target["flops"] - base["flops"]

    def pct(delta: float, ref: float) -> str:
        if ref == 0:
            return "n/a"
        return f"{(delta / ref) * 100:.2f}%"

    print(f"Delta: {target['name']} vs {base['name']}")
    print(f"  Params: {delta_params:+,.0f} ({pct(delta_params, base['params'])})")
    print(f"  MACs: {delta_macs:+,.0f} ({pct(delta_macs, base['macs'])})")
    print(f"  FLOPs: {delta_flops:+,.0f} ({pct(delta_flops, base['flops'])})")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    try:
        from thop import profile
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'thop'. Install it first, for example: pip install thop"
        ) from exc

    settings.set_global_reproducibility(deterministic=False)

    input_shape = (
        args.batch_size,
        settings.SEGMAMBA_IN_CHANS,
        *settings.INPUT_SIZE,
    )
    dummy_input = torch.randn(*input_shape, device=device)

    results = []
    for spec in build_variant_specs(args):
        results.append(profile_variant(spec, dummy_input, device, profile))

    for index, result in enumerate(results):
        if index:
            print()
        print_variant_result(result, input_shape, device)

    if len(results) > 1:
        print()
        base = results[0]
        for result in results[1:]:
            print_delta_result(base, result)
        if len(results) >= 3:
            print_delta_result(results[1], results[2])

    print()
    print("Note: FLOPs are reported as 2 x MACs. thop may skip unsupported custom ops, so use these numbers as estimates.")


if __name__ == "__main__":
    main()
