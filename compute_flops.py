import argparse
import os
import sys
import warnings

# Profiling does not need deterministic kernels, but if settings enable them
# CUDA may require this variable to be present before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import torch

import settings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ADVANCED_MODEL_DIR = os.path.join(PROJECT_ROOT, "advanced_model")
if ADVANCED_MODEL_DIR not in sys.path:
    sys.path.append(ADVANCED_MODEL_DIR)

from model_segmamba.segmamba import SegMamba


def format_count(value: float, unit: float, suffix: str) -> str:
    return f"{value / unit:.4f} {suffix}"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        help="Profile baseline and KAN variants together.",
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
        "--skan",
        dest="skan",
        action="store_true",
        default=None,
        help="Enable SKAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--no-skan",
        dest="skan",
        action="store_false",
        help="Disable SKAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--groupkan",
        dest="groupkan",
        action="store_true",
        default=None,
        help="Enable GroupKAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--no-groupkan",
        dest="groupkan",
        action="store_false",
        help="Disable GroupKAN for single-variant profiling when --compare-kan is not used.",
    )
    parser.add_argument(
        "--morton-z",
        dest="morton_z",
        action="store_true",
        default=None,
        help="Enable Morton-Z token ordering for KAN/SKAN/GroupKAN blocks.",
    )
    parser.add_argument(
        "--no-morton-z",
        dest="morton_z",
        action="store_false",
        help="Disable Morton-Z token ordering for KAN/SKAN/GroupKAN blocks.",
    )
    parser.add_argument(
        "--3d-conv",
        dest="use_unet_3d_conv",
        action="store_true",
        default=None,
        help="Use MONAI UNet 3D blocks for encoder/decoder.",
    )
    parser.add_argument(
        "--pseudo3d",
        dest="use_unet_3d_conv",
        action="store_false",
        help="Use pseudo3D blocks for encoder/decoder.",
    )
    return parser.parse_args()


def build_variant_specs(args: argparse.Namespace) -> list[dict]:
    if args.compare_kan:
        use_unet_3d_conv = settings.SEGMAMBA_3D_CONV if args.use_unet_3d_conv is None else bool(args.use_unet_3d_conv)
        return [
            {"name": "baseline", "kan_enabled": False, "skan_enabled": False, "groupkan_enabled": False, "use_unet_3d_conv": use_unet_3d_conv},
            {"name": "kan", "kan_enabled": True, "skan_enabled": False, "groupkan_enabled": False, "use_unet_3d_conv": use_unet_3d_conv},
            {"name": "skan", "kan_enabled": False, "skan_enabled": True, "groupkan_enabled": False, "use_unet_3d_conv": use_unet_3d_conv},
            {"name": "groupkan", "kan_enabled": False, "skan_enabled": False, "groupkan_enabled": True, "use_unet_3d_conv": use_unet_3d_conv},
        ]

    kan_enabled = settings.SEGMAMBA_KAN if args.kan is None else bool(args.kan)
    skan_enabled = settings.SEGMAMBA_SKAN if args.skan is None else bool(args.skan)
    groupkan_enabled = settings.SEGMAMBA_GROUPKAN if args.groupkan is None else bool(args.groupkan)
    morton_z_enabled = settings.SEGMAMBA_KAN_MORTON_Z if args.morton_z is None else bool(args.morton_z)
    use_unet_3d_conv = settings.SEGMAMBA_3D_CONV if args.use_unet_3d_conv is None else bool(args.use_unet_3d_conv)
    return [
        {
            "name": "current_settings",
            "kan_enabled": kan_enabled,
            "skan_enabled": skan_enabled,
            "groupkan_enabled": groupkan_enabled,
            "kan_morton_z_enabled": morton_z_enabled,
            "use_unet_3d_conv": use_unet_3d_conv,
        }
    ]


def instantiate_model(spec: dict, device: torch.device) -> torch.nn.Module:
    return SegMamba(
        kan_enabled=spec["kan_enabled"],
        skan_enabled=spec["skan_enabled"],
        groupkan_enabled=spec["groupkan_enabled"],
        kan_morton_z_enabled=spec.get("kan_morton_z_enabled", settings.SEGMAMBA_KAN_MORTON_Z),
        use_unet_3d_conv=spec.get("use_unet_3d_conv", settings.SEGMAMBA_3D_CONV),
    ).to(device)


def profile_variant(spec: dict, dummy_input: torch.Tensor, device: torch.device, profile_fn) -> dict:
    model = instantiate_model(spec, device)
    model.eval()
    pytorch_params = count_params(model)
    with torch.no_grad():
        macs, thop_params = profile_fn(model, inputs=(dummy_input,), verbose=False)
    model_cpu = model.to("cpu")
    del model_cpu
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": spec["name"],
        "kan_enabled": spec["kan_enabled"],
        "skan_enabled": spec["skan_enabled"],
        "groupkan_enabled": spec["groupkan_enabled"],
        "kan_morton_z_enabled": spec.get("kan_morton_z_enabled", settings.SEGMAMBA_KAN_MORTON_Z),
        "use_unet_3d_conv": spec.get("use_unet_3d_conv", settings.SEGMAMBA_3D_CONV),
        "macs": float(macs),
        "params": float(pytorch_params),
        "thop_params": float(thop_params),
        "flops": float(macs) * 2.0,
    }


def print_variant_result(result: dict, input_shape: tuple[int, ...], device: torch.device) -> None:
    print(f"Variant: {result['name']}")
    print(
        "  Config: "
        f"KAN={result['kan_enabled']}, "
        f"SKAN={result['skan_enabled']}, "
        f"GroupKAN={result['groupkan_enabled']}, "
        f"Morton-Z={result['kan_morton_z_enabled']}, "
        f"3DConv={result['use_unet_3d_conv']}"
    )
    print(f"  Device: {device}")
    print(f"  Input shape: {input_shape}")
    print(f"  Params (PyTorch): {int(result['params']):,} ({format_count(result['params'], 1e6, 'M')})")
    print(f"  Params (THOP): {int(result['thop_params']):,} ({format_count(result['thop_params'], 1e6, 'M')})")
    if int(result["params"]) != int(result["thop_params"]):
        missing = result["params"] - result["thop_params"]
        print(f"  Params not seen by THOP: {int(missing):,} ({format_count(missing, 1e6, 'M')})")
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
    print("Note: Params (PyTorch) is the authoritative parameter count.")
    print("      FLOPs are reported as 2 x MACs. THOP may skip unsupported custom ops, so use MACs/FLOPs as estimates.")


if __name__ == "__main__":
    main()
