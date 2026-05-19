#!/usr/bin/env python3
"""Benchmark alternative implementations for get_input/convert_labels."""

from __future__ import annotations

import argparse
import time

import torch


def bench(name, fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    ms = (time.perf_counter() - start) * 1000 / iterations
    print(f"{name:<36} {ms:.4f} ms/iter")
    return ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--shape", type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()

    d, h, w = args.shape
    values = torch.tensor([-1, 0, 1, 2, 3, 4], dtype=torch.int16)
    seg = values[torch.randint(0, len(values), (1, 1, d, h, w))]
    label = seg[:, 0].long()

    def get_input_current():
        x = seg[:, 0].long()
        x[x == -1] = 0
        x[x == 4] = 3
        return x

    def get_input_clamp():
        x = seg[:, 0].long()
        return x.clamp_(0, 3)

    def convert_current():
        labels = label[:, None]
        result = [
            (labels == 1) | (labels == 3),
            (labels == 1) | (labels == 3) | (labels == 2),
            labels == 3,
        ]
        return torch.cat(result, dim=1).float()

    def convert_reuse_masks():
        labels = label[:, None]
        is1 = labels == 1
        is2 = labels == 2
        is3 = labels == 3
        tc = is1 | is3
        wt = tc | is2
        return torch.cat((tc, wt, is3), dim=1).float()

    def convert_prealloc_assign():
        x = label
        out = torch.empty((x.shape[0], 3, *x.shape[1:]), dtype=torch.float32)
        is1 = x == 1
        is2 = x == 2
        is3 = x == 3
        out[:, 0] = is1 | is3
        out[:, 1] = is1 | is2 | is3
        out[:, 2] = is3
        return out

    assert torch.equal(get_input_current(), get_input_clamp())
    assert torch.equal(convert_current(), convert_reuse_masks())
    assert torch.equal(convert_current(), convert_prealloc_assign())

    print(f"shape=(1, 1, {d}, {h}, {w}), iterations={args.iterations}")
    bench("get_input current", get_input_current, args.iterations, args.warmup)
    bench("get_input clamp_", get_input_clamp, args.iterations, args.warmup)
    bench("convert current cat float", convert_current, args.iterations, args.warmup)
    bench("convert reuse masks", convert_reuse_masks, args.iterations, args.warmup)
    bench("convert prealloc assign", convert_prealloc_assign, args.iterations, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
