import argparse
import gc
import os
import sys
import time
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
from model_segmamba.segmamba import (  # noqa: E402
    GSC,
    MambaLayer,
    Pseudo3DBottleneckBlock,
    Pseudo3DUpBlock,
    TSMambaLayer,
    TokenGroupKANPseudo3DBlock,
    TokenSKANPseudo3DUpBlock,
    TokenSKANPseudo3DBlock,
)
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock  # noqa: E402


BYTES_IN_MB = 1024 * 1024


class IdentitySequenceOp(nn.Module):
    def forward(self, x):
        return x


def format_num(value):
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def format_ms(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def param_memory_mb(module):
    return sum(p.numel() * p.element_size() for p in module.parameters()) / BYTES_IN_MB


def tensor_memory_mb(tensor):
    return tensor.numel() * tensor.element_size() / BYTES_IN_MB


def replace_sequence_ops(module):
    replaced = []
    for name, child in list(module.named_children()):
        cls_name = child.__class__.__name__.lower()
        if cls_name in {"mamba", "mamba2", "mamba3"}:
            setattr(module, name, IdentitySequenceOp())
            replaced.append(name)
        else:
            replaced.extend(f"{name}.{n}" for n in replace_sequence_ops(child))
    return replaced


def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def estimate_module_macs(module, inputs, output):
    if isinstance(output, (tuple, list)):
        return 0
    if not torch.is_tensor(output):
        return 0

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_spatial = _prod(output.shape[2:])
        kernel_ops = module.in_channels // module.groups * _prod(module.kernel_size)
        return batch * out_channels * out_spatial * kernel_ops

    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_spatial = _prod(output.shape[2:])
        kernel_ops = module.in_channels // module.groups * _prod(module.kernel_size)
        return batch * out_channels * out_spatial * kernel_ops

    if isinstance(module, nn.Linear):
        return output.numel() * module.in_features

    if isinstance(module, (nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return output.numel() * 2

    return 0


def profile_forward(module, inputs, device, warmup, iters):
    macs_by_type = defaultdict(int)
    macs_by_name = defaultdict(int)
    activation_bytes = 0
    handles = []

    named_modules = {m: n for n, m in module.named_modules()}

    def hook(submodule, hook_inputs, hook_output):
        nonlocal activation_bytes
        macs = estimate_module_macs(submodule, hook_inputs, hook_output)
        if macs:
            macs_by_type[submodule.__class__.__name__] += macs
            macs_by_name[named_modules.get(submodule, submodule.__class__.__name__)] += macs
        if torch.is_tensor(hook_output):
            activation_bytes += hook_output.numel() * hook_output.element_size()

    for submodule in module.modules():
        if submodule is module:
            continue
        handles.append(submodule.register_forward_hook(hook))

    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(iters):
            output = module(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    for handle in handles:
        handle.remove()

    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / BYTES_IN_MB

    denom = max(warmup + iters, 1)
    return {
        "latency_ms": elapsed * 1000.0 / max(iters, 1),
        "macs": sum(macs_by_type.values()) / denom,
        "macs_by_type": {name: value / denom for name, value in macs_by_type.items()},
        "macs_by_name": {name: value / denom for name, value in macs_by_name.items()},
        "activation_mb": activation_bytes / denom / BYTES_IN_MB,
        "peak_vram_mb": peak_vram_mb,
        "output_shape": tuple(output.shape) if torch.is_tensor(output) else str(type(output)),
    }


def mamba3_kwargs(args):
    return {
        "mamba3_mimo_enabled": args.mamba3_mimo,
        "mamba3_mimo_rank": args.mamba3_mimo_rank,
        "mamba3_rope_fraction": args.mamba3_rope_fraction,
        "mamba3_outproj_norm_enabled": args.mamba3_outproj_norm,
    }


def make_block(block_name, channels, spatial, upsample_mode, norm_name, args):
    block_name = block_name.lower()
    if block_name == "gsc":
        return GSC(channels), (torch.randn(1, channels, *spatial),)
    if block_name == "pseudo3d":
        return Pseudo3DBottleneckBlock(channels, channels, norm_name=norm_name), (torch.randn(1, channels, *spatial),)
    if block_name == "mambalayer":
        return MambaLayer(
            dim=channels,
            mamba_impl=settings.ADVANCED_SEGMAMBA_MAMBA_IMPL,
            morton_z_enabled=settings.SEGMAMBA_KAN_MORTON_Z,
            **mamba3_kwargs(args),
        ), (torch.randn(1, channels, *spatial),)
    if block_name == "tsmamba":
        return TSMambaLayer(
            dim=channels,
            mamba_impl=settings.ADVANCED_SEGMAMBA_MAMBA_IMPL,
            morton_z_enabled=settings.SEGMAMBA_KAN_MORTON_Z,
            **mamba3_kwargs(args),
        ), (torch.randn(1, channels, *spatial),)
    if block_name == "tokenskan":
        return TokenSKANPseudo3DBlock(
            channels,
            channels,
            norm_name=norm_name,
            morton_z_enabled=settings.SEGMAMBA_KAN_MORTON_Z,
        ), (torch.randn(1, channels, *spatial),)
    if block_name == "tokengroupkan":
        return TokenGroupKANPseudo3DBlock(
            channels,
            channels,
            norm_name=norm_name,
            active_group=settings.SEGMAMBA_GROUPKAN_ACTIVE_GROUP,
            channel_group=settings.SEGMAMBA_GROUPKAN_CHANNEL_GROUP,
            spatial_mixer=settings.SEGMAMBA_GROUPKAN_SPATIAL_MIXER,
            morton_z_enabled=settings.SEGMAMBA_KAN_MORTON_Z,
        ), (torch.randn(1, channels, *spatial),)
    raise ValueError(f"Unknown block: {block_name}")


def make_comparison_cases(channels, spatial, norm_name, args):
    low = [max(1, size // 2) for size in spatial]
    cases = []

    cases.append((
        "encoder_3d_unetr",
        UnetrBasicBlock(
            spatial_dims=3,
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ),
        (torch.randn(1, channels, *spatial),),
    ))
    cases.append((
        "encoder_pseudo3d",
        Pseudo3DBottleneckBlock(
            channels,
            channels,
            norm_name=norm_name,
            res_block=True,
            morton_z_enabled=False,
        ),
        (torch.randn(1, channels, *spatial),),
    ))
    cases.append((
        "encoder_skan_no_morton",
        TokenSKANPseudo3DBlock(
            channels,
            channels,
            norm_name=norm_name,
            res_block=True,
            morton_z_enabled=False,
        ),
        (torch.randn(1, channels, *spatial),),
    ))
    cases.append((
        "encoder_skan_morton",
        TokenSKANPseudo3DBlock(
            channels,
            channels,
            norm_name=norm_name,
            res_block=True,
            morton_z_enabled=True,
        ),
        (torch.randn(1, channels, *spatial),),
    ))

    cases.append((
        "decoder_3d_transconv",
        UnetrUpBlock(
            spatial_dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ),
        (torch.randn(1, channels * 2, *low), torch.randn(1, channels, *spatial)),
    ))
    cases.append((
        "decoder_pseudo3d_transconv",
        Pseudo3DUpBlock(
            spatial_dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            upsample_mode="transconv",
        ),
        (torch.randn(1, channels * 2, *low), torch.randn(1, channels, *spatial)),
    ))
    cases.append((
        "decoder_pseudo3d_onsampling",
        Pseudo3DUpBlock(
            spatial_dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            upsample_mode="onsampling",
        ),
        (torch.randn(1, channels * 2, *low), torch.randn(1, channels, *spatial)),
    ))
    cases.append((
        "decoder_skan_transconv",
        TokenSKANPseudo3DUpBlock(
            spatial_dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            upsample_mode="transconv",
            morton_z_enabled=False,
        ),
        (torch.randn(1, channels * 2, *low), torch.randn(1, channels, *spatial)),
    ))
    cases.append((
        "decoder_skan_morton_transconv",
        TokenSKANPseudo3DUpBlock(
            spatial_dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            upsample_mode="transconv",
            morton_z_enabled=True,
        ),
        (torch.randn(1, channels * 2, *low), torch.randn(1, channels, *spatial)),
    ))
    return cases


def print_result(name, module, inputs, result, mocked_ops, verbose=True):
    params = result.get("original_params")
    trainable = result.get("original_trainable_params")
    executable_params, executable_trainable = count_params(module)
    input_shapes = [tuple(x.shape) for x in inputs]
    macs = result.get("macs")
    print(f"\n[{name}]")
    print(f"  class: {module.__class__.__name__}")
    print(f"  input: {input_shapes}")
    print(f"  output: {result.get('output_shape')}")
    print(f"  params: {params:,} ({params / 1e6:.3f}M), trainable: {trainable:,}")
    print(f"  param_memory_mb: {result.get('original_param_memory_mb'):.3f}")
    if mocked_ops:
        print(f"  executable_params_after_mock: {executable_params:,} ({executable_trainable:,} trainable)")
    print(f"  input_memory_mb: {sum(tensor_memory_mb(x) for x in inputs):.3f}")
    print(f"  activation_memory_mb_per_forward_rough: {result.get('activation_mb', 0):.3f}")
    print(f"  latency_ms: {format_ms(result.get('latency_ms'))}")
    print(f"  conv_linear_norm_macs: {format_num(macs)} ({(macs or 0) / 1e9:.6f} GMAC)")
    print(f"  conv_linear_norm_flops_2xmac: {((macs or 0) * 2) / 1e9:.6f} GFLOPs")
    print(f"  peak_vram_mb: {format_ms(result.get('peak_vram_mb'))}")
    if mocked_ops:
        print(f"  mocked_sequence_ops: {', '.join(mocked_ops)}")
    if verbose and result.get("macs_by_type"):
        print("  macs_by_type:")
        for cls_name, cls_macs in sorted(result["macs_by_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {cls_name:24s} {cls_macs / 1e9:.6f} GMAC")


def print_summary_table(results):
    if not results:
        return

    print("\nSummary table")
    print(
        "  "
        f"{'case':34s} {'params(M)':>10s} {'GMAC':>10s} {'GFLOPs':>10s} "
        f"{'lat(ms)':>10s} {'VRAM(MB)':>10s} {'act(MB)':>10s}"
    )
    for row in results:
        macs = row["macs"] or 0
        print(
            "  "
            f"{row['name']:34s} "
            f"{row['params'] / 1e6:10.3f} "
            f"{macs / 1e9:10.3f} "
            f"{2 * macs / 1e9:10.3f} "
            f"{row['latency_ms']:10.3f} "
            f"{format_ms(row['peak_vram_mb']):>10s} "
            f"{row['activation_mb']:10.3f}"
        )

    fastest = min(results, key=lambda row: row["latency_ms"])
    lowest_params = min(results, key=lambda row: row["params"])
    lowest_vram_candidates = [row for row in results if row["peak_vram_mb"] is not None]
    print("\nQuick picks")
    print(f"  fastest: {fastest['name']} ({fastest['latency_ms']:.3f} ms)")
    print(f"  fewest params: {lowest_params['name']} ({lowest_params['params'] / 1e6:.3f}M)")
    if lowest_vram_candidates:
        lowest_vram = min(lowest_vram_candidates, key=lambda row: row["peak_vram_mb"])
        print(f"  lowest peak VRAM: {lowest_vram['name']} ({lowest_vram['peak_vram_mb']:.3f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile advanced SegMamba blocks with mock inputs.")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["gsc", "pseudo3d", "mambalayer", "tsmamba"],
        help="Blocks: gsc pseudo3d mambalayer tsmamba tokenskan tokengroupkan",
    )
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--spatial", type=int, nargs=3, default=[24, 24, 24])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--upsample-mode", default="transconv", choices=["transconv", "onsampling"])
    parser.add_argument("--norm-name", default=getattr(settings, "SEGMAMBA_NORM_NAME", "instance"))
    parser.add_argument(
        "--mock-sequence-ops",
        action="store_true",
        help="Replace Mamba/Mamba2/Mamba3 with identity. Useful for CPU shape/timing of surrounding block logic.",
    )
    parser.add_argument("--mamba3-mimo", action="store_true", help="Enable Mamba3 MIMO for Mamba3-based blocks.")
    parser.add_argument("--mamba3-mimo-rank", type=int, default=getattr(settings, "ADVANCED_SEGMAMBA_MAMBA3_MIMO_RANK", 4))
    parser.add_argument("--mamba3-rope-fraction", type=float, default=getattr(settings, "ADVANCED_SEGMAMBA_MAMBA3_ROPE_FRACTION", 0.5))
    parser.add_argument("--mamba3-outproj-norm", action="store_true", help="Enable Mamba3 output RMSNorm/gate norm.")
    parser.add_argument(
        "--compare-suite",
        choices=["none", "advanced"],
        default="none",
        help="Run a predefined comparison suite instead of --blocks.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Only print the final comparison table.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print("Advanced block profile")
    print(f"  device: {device}")
    print(f"  torch: {torch.__version__}")
    print(f"  channels: {args.channels}")
    print(f"  spatial: {tuple(args.spatial)}")
    print(f"  warmup/iters: {args.warmup}/{args.iters}")
    print(f"  mamba3_mimo: {args.mamba3_mimo}, rank: {args.mamba3_mimo_rank}")
    print("  FLOPs caveat: counts below cover Conv/Linear/Norm only; Mamba selective scan, SKAN/KAN")
    print("               custom functions, indexing, and Onsampling internals are lower-bound/uncounted.")

    if args.compare_suite == "advanced":
        cases = make_comparison_cases(args.channels, args.spatial, args.norm_name, args)
    else:
        cases = [
            (block_name, *make_block(block_name, args.channels, args.spatial, args.upsample_mode, args.norm_name, args))
            for block_name in args.blocks
        ]

    summary_rows = []
    for name, module, inputs in cases:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            original_params, original_trainable_params = count_params(module)
            original_param_memory_mb = param_memory_mb(module)
            mocked = replace_sequence_ops(module) if args.mock_sequence_ops else []
            module = module.to(device)
            inputs = tuple(x.to(device) for x in inputs)
            result = profile_forward(module, inputs, device, args.warmup, args.iters)
            result["original_params"] = original_params
            result["original_trainable_params"] = original_trainable_params
            result["original_param_memory_mb"] = original_param_memory_mb
            if not args.summary_only:
                print_result(name, module, inputs, result, mocked)
            summary_rows.append({
                "name": name,
                "params": original_params,
                "macs": result.get("macs"),
                "latency_ms": result.get("latency_ms"),
                "peak_vram_mb": result.get("peak_vram_mb"),
                "activation_mb": result.get("activation_mb", 0),
            })
        except Exception as exc:
            print(f"\n[{name}]")
            print(f"  ERROR: {exc.__class__.__name__}: {exc}")
            if not args.mock_sequence_ops:
                print("  Hint: try --mock-sequence-ops on CPU, or run on a CUDA machine for Mamba3/Triton.")

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
