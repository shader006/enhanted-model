import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from advanced_model.model_segmamba.dcnv4 import DCNv4_3D, HAS_REAL_DCNV4


class Conv3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x) + x


class P3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x) + x


class DepthwiseSeparable3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x) + x


class P3DDCNv4Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dcn = DCNv4_3D(channels, kernel_size=3, pad=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.act1 = nn.GELU()
        self.depth = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.act2 = nn.GELU()
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm3 = nn.InstanceNorm3d(channels)
        self.act3 = nn.GELU()

    def forward(self, x):
        out = self.act1(self.norm1(self.dcn(x)))
        out = self.act2(self.norm2(self.depth(out)))
        out = self.act3(self.norm3(self.pointwise(out)))
        return out + x


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def conv3d_macs(in_channels, out_channels, kernel, output_shape, groups=1):
    d, h, w = output_shape
    kernel_ops = kernel[0] * kernel[1] * kernel[2] * (in_channels // groups)
    return d * h * w * out_channels * kernel_ops


def estimate_macs(name, channels, shape):
    _, _, d, h, w = shape
    if name == "Conv3D":
        return (
            conv3d_macs(channels, channels, (3, 3, 3), (d, h, w))
            + conv3d_macs(channels, channels, (3, 3, 3), (d, h, w))
            + conv3d_macs(channels, channels, (1, 1, 1), (d, h, w))
        )
    if name == "P3D":
        return (
            conv3d_macs(channels, channels, (1, 3, 3), (d, h, w))
            + conv3d_macs(channels, channels, (3, 1, 1), (d, h, w))
            + conv3d_macs(channels, channels, (1, 1, 1), (d, h, w))
        )
    if name == "DW":
        return (
            conv3d_macs(channels, channels, (3, 3, 3), (d, h, w), groups=channels)
            + conv3d_macs(channels, channels, (1, 1, 1), (d, h, w))
            + conv3d_macs(channels, channels, (3, 3, 3), (d, h, w), groups=channels)
            + conv3d_macs(channels, channels, (1, 1, 1), (d, h, w))
        )
    return None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_one(name, module, x, args):
    device = x.device
    module.train(args.train_mode)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0) if args.backward else None

    for _ in range(args.warmup):
        y = module(x)
        if args.backward:
            loss = y.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        y = module(x)
        if args.backward:
            loss = y.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        sync(device)
    elapsed = time.perf_counter() - start

    ms = elapsed * 1000.0 / args.iters
    voxels = x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    return {
        "name": name,
        "params": count_params(module),
        "ms": ms,
        "samples_per_s": x.shape[0] * 1000.0 / ms,
        "mvoxels_per_s": voxels / ms / 1000.0,
        "peak_mem_mb": peak_mem,
    }


def format_millions(value):
    return f"{value / 1_000_000:.3f}M"


def print_table(rows, shape):
    headers = ["block", "params", "est_macs", "ms/iter", "samples/s", "Mvox/s", "peak_mem"]
    print(" | ".join(headers))
    print(" | ".join(["-" * len(h) for h in headers]))
    for row in rows:
        macs = estimate_macs(row["name"], shape[1], shape)
        peak_mem = "-" if row["peak_mem_mb"] is None else f"{row['peak_mem_mb']:.1f} MB"
        print(
            " | ".join(
                [
                    row["name"],
                    format_millions(row["params"]),
                    "-" if macs is None else format_millions(macs),
                    f"{row['ms']:.3f}",
                    f"{row['samples_per_s']:.3f}",
                    f"{row['mvoxels_per_s']:.3f}",
                    peak_mem,
                ]
            )
        )


def parse_shape(value):
    parts = [int(part) for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("shape must be B,C,D,H,W, e.g. 1,48,16,32,32")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conv3D, P3D, DW, and P3D+DCNv4 blocks on mock 3D tensors.")
    parser.add_argument("--shape", type=parse_shape, default=(1, 48, 16, 32, 32), help="Input shape B,C,D,H,W.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backward", action="store_true", help="Benchmark forward + backward instead of forward only.")
    parser.add_argument("--train-mode", action="store_true", help="Run modules in train mode.")
    parser.add_argument("--channels-last-3d", action="store_true", help="Use channels_last_3d input/module memory format.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 CPU benchmark is not supported.")

    b, c, d, h, w = args.shape
    if c % 16 != 0:
        print("[!] For real DCNv4, channels should usually be divisible by 16. Current C:", c)

    x = torch.randn(args.shape, device=device, dtype=dtype)
    if args.channels_last_3d:
        x = x.contiguous(memory_format=torch.channels_last_3d)

    blocks = {
        "Conv3D": Conv3DBlock(c),
        "P3D": P3DBlock(c),
        "DW": DepthwiseSeparable3DBlock(c),
        "P3D+DCNv4": P3DDCNv4Block(c),
    }

    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"shape: {args.shape}")
    print(f"mode: {'forward+backward' if args.backward else 'forward'}")
    print(f"channels_last_3d: {args.channels_last_3d}")
    print(f"HAS_REAL_DCNV4: {HAS_REAL_DCNV4}")
    rows = []
    for name, block in blocks.items():
        block = block.to(device=device, dtype=dtype)
        block_x = x
        if args.channels_last_3d:
            block = block.to(memory_format=torch.channels_last_3d)
            block_x = block_x.contiguous(memory_format=torch.channels_last_3d)
        try:
            row = bench_one(name, block, block_x, args)
            rows.append(row)
        except Exception as exc:
            print(f"[!] {name} failed: {type(exc).__name__}: {exc}")

    print_table(rows, args.shape)


if __name__ == "__main__":
    main()
