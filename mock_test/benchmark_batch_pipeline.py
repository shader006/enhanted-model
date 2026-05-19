#!/usr/bin/env python3
"""Benchmark batch generation and batchgenerators augmentation.

This focuses on the training input pipeline after MedicalDataset.__getitem__:
  1. DataLoaderMultiProcess.generate_train_batch(): sample cases, crop/pad patches.
  2. batchgenerators augmenter: spatial/intensity augmentation and NumpyToTensor.

Examples:
  python enhanted-model/mock_test/benchmark_batch_pipeline.py --stage batch --n-batches 20
  python enhanted-model/mock_test/benchmark_batch_pipeline.py --stage augmenter --transform noaug --n-batches 20
  python enhanted-model/mock_test/benchmark_batch_pipeline.py --stage augmenter --transform full --backend single-thread --n-batches 10
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ADVANCED_MODEL_DIR = REPO_ROOT / "enhanted-model" / "advanced_model"
if str(ADVANCED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ADVANCED_MODEL_DIR))

from light_training.dataloading.base_data_loader import DataLoaderMultiProcess  # noqa: E402


class CachedMemmapDataset:
    """Minimal MedicalDataset-compatible dataset for pipeline benchmarks."""

    def __init__(self, paths: list[Path], test: bool = False):
        self.datalist = paths
        self.test = test
        self.data_cached = [self._load_pkl(path) for path in paths]
        self._memmap_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _load_pkl(path: Path):
        with open(str(path).replace(".npz", ".pkl"), "rb") as f:
            return pickle.load(f)

    def _load_memmap(self, path: str):
        if path not in self._memmap_cache:
            self._memmap_cache[path] = np.load(path, mmap_mode="r")
        return self._memmap_cache[path]

    def read_data(self, path: Path):
        image = self._load_memmap(str(path).replace(".npz", ".npy"))
        seg = None if self.test else self._load_memmap(str(path).replace(".npz", "_seg.npy"))
        return image, seg

    def __getitem__(self, index: int):
        image, seg = self.read_data(self.datalist[index])
        item = {
            "data": image,
            "properties": self.data_cached[index],
        }
        if seg is not None:
            item["seg"] = seg
        return item

    def __len__(self):
        return len(self.datalist)


def validate_cases(paths: list[Path]) -> list[Path]:
    valid = []
    for path in paths:
        required = [
            Path(str(path).replace(".npz", ".npy")),
            Path(str(path).replace(".npz", "_seg.npy")),
            Path(str(path).replace(".npz", ".pkl")),
        ]
        if all(p.exists() for p in required):
            valid.append(path)
    return valid


def resolve_data_dir(explicit: str | None) -> Path:
    candidates = [
        explicit,
        "enhanted-model/data/fullres/train",
        "enhanted-model/advanced_model/data/fullres/train",
        "SegMamba/data/fullres/train",
        "Segmamba2/SegMamba/data/fullres/train",
    ]
    data_dir = next((Path(p) for p in candidates if p and Path(p).is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError("Could not find preprocessed data. Pass --data-dir explicitly.")
    return data_dir


def build_transform(name: str, patch_size: list[int]):
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.color_transforms import (
        BrightnessMultiplicativeTransform,
        ContrastAugmentationTransform,
        GammaTransform,
    )
    from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
    from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
    from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform
    from light_training.augment.train_augment import (
        ModalityDropoutTransform,
        get_train_transforms,
        get_train_transforms_noaug,
        get_train_transforms_nomirror,
        get_train_transforms_onlymirror,
        get_train_transforms_onlyspatial,
    )

    kwargs = {
        "patch_size": patch_size,
        "mirror_axes": [0, 1, 2],
        "modality_dropout_prob": 0.0,
        "modality_dropout_max_channels": 1,
    }
    if name == "noaug":
        return get_train_transforms_noaug(**kwargs)
    if name == "full":
        return get_train_transforms(**kwargs)
    if name == "nomirror":
        return get_train_transforms_nomirror(**kwargs)
    if name == "onlymirror":
        return get_train_transforms_onlymirror(**kwargs)
    if name == "onlyspatial":
        return get_train_transforms_onlyspatial(**kwargs)
    if name == "fast":
        angle = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
        tr_transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=angle,
                angle_y=angle,
                angle_z=angle,
                p_rot_per_axis=1,
                do_scale=True,
                scale=(0.85, 1.2),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=1,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0,
                p_scale_per_sample=0.1,
                p_rot_per_sample=0.1,
                independent_scale_for_each_axis=False,
            ),
            GaussianNoiseTransform(p_per_sample=0.05),
            GaussianBlurTransform(
                (0.5, 0.8),
                different_sigma_per_channel=True,
                p_per_sample=0.1,
                p_per_channel=0.25,
            ),
            BrightnessMultiplicativeTransform(multiplier_range=(0.8, 1.2), p_per_sample=0.1),
            ContrastAugmentationTransform(p_per_sample=0.1),
            GammaTransform((0.8, 1.25), False, True, retain_stats=True, p_per_sample=0.15),
        ]
        if kwargs["mirror_axes"] is not None and len(kwargs["mirror_axes"]) > 0:
            tr_transforms.append(MirrorTransform(kwargs["mirror_axes"]))
        tr_transforms.append(ModalityDropoutTransform(p=0.0, max_channels=1))
        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(NumpyToTensor(["data", "seg"], "float"))
        return Compose(tr_transforms)
    raise ValueError(f"Unsupported transform: {name}")


def build_single_transform(name: str, patch_size: list[int]):
    from batchgenerators.transforms.color_transforms import (
        BrightnessMultiplicativeTransform,
        ContrastAugmentationTransform,
        GammaTransform,
    )
    from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
    from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
    from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
    from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform
    from light_training.augment.train_augment import ModalityDropoutTransform

    angle = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
    transforms = {
        "spatial": SpatialTransform(
            patch_size,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0, 0),
            sigma=(0, 0),
            do_rotation=True,
            angle_x=angle,
            angle_y=angle,
            angle_z=angle,
            p_rot_per_axis=1,
            do_scale=True,
            scale=(0.7, 1.4),
            border_mode_data="constant",
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=-1,
            order_seg=1,
            random_crop=False,
            p_el_per_sample=0,
            p_scale_per_sample=1.0,
            p_rot_per_sample=1.0,
            independent_scale_for_each_axis=False,
        ),
        "spatial_prob": SpatialTransform(
            patch_size,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0, 0),
            sigma=(0, 0),
            do_rotation=True,
            angle_x=angle,
            angle_y=angle,
            angle_z=angle,
            p_rot_per_axis=1,
            do_scale=True,
            scale=(0.7, 1.4),
            border_mode_data="constant",
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=-1,
            order_seg=1,
            random_crop=False,
            p_el_per_sample=0,
            p_scale_per_sample=0.2,
            p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
        ),
        "noise": GaussianNoiseTransform(p_per_sample=1.0),
        "blur": GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=1.0,
            p_per_channel=1.0,
        ),
        "brightness": BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=1.0),
        "contrast": ContrastAugmentationTransform(p_per_sample=1.0),
        "lowres": SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=1.0,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=1.0,
            ignore_axes=None,
        ),
        "gamma_invert": GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=1.0),
        "gamma": GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=1.0),
        "mirror": MirrorTransform([0, 1, 2]),
        "modality_dropout": ModalityDropoutTransform(p=1.0, max_channels=1),
        "remove_label": RemoveLabelTransform(-1, 0),
        "to_tensor": NumpyToTensor(["data", "seg"], "float"),
    }
    if name not in transforms:
        raise ValueError(f"Unsupported single transform: {name}. Available: {sorted(transforms)}")
    return transforms[name]


def clone_batch(batch: dict) -> dict:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def benchmark_transform(transform, base_batch: dict, n_batches: int, warmup: int) -> list[float]:
    times = []
    for i in range(warmup + n_batches):
        gc.collect()
        batch = clone_batch(base_batch)
        start = time.perf_counter()
        obj = transform(**batch)
        elapsed = time.perf_counter() - start
        del obj
        if i >= warmup:
            times.append(elapsed)
    return times


def summarize(times: list[float], label: str) -> dict:
    times_ms = [t * 1000 for t in times]
    return {
        "label": label,
        "n": len(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def print_result(result: dict) -> None:
    print()
    print(f"{'stage':<34} {'n':>5} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'std':>10}")
    print("-" * 95)
    print(
        f"{result['label']:<34} {result['n']:>5} "
        f"{result['mean_ms']:>9.1f}ms {result['median_ms']:>9.1f}ms "
        f"{result['min_ms']:>9.1f}ms {result['max_ms']:>9.1f}ms {result['std_ms']:>9.1f}ms"
    )


def benchmark_callable(fn: Callable[[], object], n_batches: int, warmup: int) -> list[float]:
    times = []
    for i in range(warmup + n_batches):
        gc.collect()
        start = time.perf_counter()
        obj = fn()
        elapsed = time.perf_counter() - start
        del obj
        if i >= warmup:
            times.append(elapsed)
    return times


def close_augmenter(augmenter) -> None:
    target = getattr(augmenter, "augmenter", augmenter)
    for method_name in ("_finish", "finish", "_end"):
        method = getattr(target, method_name, None)
        if callable(method):
            method()
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--stage", choices=["batch", "augmenter", "transform-profile"], default="batch")
    parser.add_argument("--transform", choices=["noaug", "full", "fast", "nomirror", "onlymirror", "onlyspatial"], default="noaug")
    parser.add_argument(
        "--single-transform",
        choices=[
            "all",
            "spatial",
            "spatial_prob",
            "noise",
            "blur",
            "brightness",
            "contrast",
            "lowres",
            "gamma_invert",
            "gamma",
            "mirror",
            "modality_dropout",
            "remove_label",
            "to_tensor",
        ],
        default="all",
    )
    parser.add_argument("--backend", choices=["single-thread", "multi-thread", "nondet-multiprocess"], default="single-thread")
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-cached", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    paths = validate_cases([Path(p) for p in sorted(glob.glob(str(data_dir / "*.npz")))])
    paths = paths[: args.n_cases]
    if not paths:
        raise FileNotFoundError(f"No valid .npz/.npy/_seg.npy/.pkl cases found in {data_dir}")

    dataset = CachedMemmapDataset(paths)
    loader = DataLoaderMultiProcess(
        dataset,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        print_time=False,
    )

    print(f"data_dir: {data_dir}")
    print(f"stage: {args.stage}")
    print(f"cases: {len(paths)}")
    print(f"batch_size: {args.batch_size}")
    print(f"patch_size: {args.patch_size}")

    if args.stage == "batch":
        label = f"generate_train_batch B={args.batch_size}"
        times = benchmark_callable(loader.generate_train_batch, args.n_batches, args.warmup)
        result = summarize(times, label)
    elif args.stage == "transform-profile":
        names = [
            "spatial_prob",
            "spatial",
            "noise",
            "blur",
            "brightness",
            "contrast",
            "lowres",
            "gamma_invert",
            "gamma",
            "mirror",
            "modality_dropout",
            "remove_label",
            "to_tensor",
        ]
        if args.single_transform != "all":
            names = [args.single_transform]
        base_batch = loader.generate_train_batch()
        results = []
        for name in names:
            transform = build_single_transform(name, args.patch_size)
            print(f"running transform: {name}")
            times = benchmark_transform(transform, base_batch, args.n_batches, args.warmup)
            item = summarize(times, name)
            item.update(
                {
                    "data_dir": str(data_dir),
                    "stage": args.stage,
                    "backend": args.backend,
                    "batch_size": args.batch_size,
                    "patch_size": args.patch_size,
                    "n_cases": len(paths),
                    "n_batches": args.n_batches,
                    "warmup": args.warmup,
                }
            )
            results.append(item)

        print()
        print(f"{'transform':<22} {'n':>5} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'std':>10}")
        print("-" * 83)
        for item in sorted(results, key=lambda x: x["mean_ms"], reverse=True):
            print(
                f"{item['label']:<22} {item['n']:>5} "
                f"{item['mean_ms']:>9.1f}ms {item['median_ms']:>9.1f}ms "
                f"{item['min_ms']:>9.1f}ms {item['max_ms']:>9.1f}ms {item['std_ms']:>9.1f}ms"
            )

        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2)
            print(f"wrote: {out_path}")
        return 0
    else:
        from light_training.augment.multi_processor import create_limited_len_augmenter

        transform = build_transform(args.transform, args.patch_size)
        augmenter = create_limited_len_augmenter(
            mode=args.backend,
            my_imaginary_length=args.n_batches + args.warmup,
            data_loader=loader,
            transform=transform,
            num_processes=args.num_processes,
            num_cached=args.num_cached,
            seeds=[args.seed + i for i in range(max(args.num_processes, 1))],
            pin_memory=True,
            wait_time=0.02,
        )
        label = f"augmenter {args.transform} {args.backend} B={args.batch_size}"
        try:
            times = benchmark_callable(lambda: next(augmenter), args.n_batches, args.warmup)
        finally:
            close_augmenter(augmenter)
        result = summarize(times, label)
    result.update(
        {
            "data_dir": str(data_dir),
            "stage": args.stage,
            "transform": args.transform,
            "backend": args.backend,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "n_cases": len(paths),
            "n_batches": args.n_batches,
            "warmup": args.warmup,
        }
    )
    print_result(result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
