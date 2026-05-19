#!/usr/bin/env python3
"""Mock tests for BraTSTrainer label mapping helpers.

Run:
  conda run -n brats23 python enhanted-model/mock_test/test_label_mapping.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ADVANCED_MODEL_DIR = REPO_ROOT / "enhanted-model" / "advanced_model"
if str(ADVANCED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ADVANCED_MODEL_DIR))

import importlib.util  # noqa: E402


def load_train_module():
    module_path = ADVANCED_MODEL_DIR / "3_train.py"
    spec = importlib.util.spec_from_file_location("advanced_train", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_tensor_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def test_get_input_label_mapping(train_module) -> None:
    trainer = object.__new__(train_module.BraTSTrainer)
    image = torch.arange(1 * 4 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2, 2)
    seg = torch.tensor(
        [[[[[-1, 0], [1, 2]], [[3, 4], [4, -1]]]]],
        dtype=torch.int16,
    )
    batch = {
        "data": image,
        "seg": seg,
    }

    mapped_image, mapped_label = trainer.get_input(batch)
    expected_label = torch.tensor(
        [[[[0, 0], [1, 2]], [[3, 3], [3, 0]]]],
        dtype=torch.long,
    )

    assert mapped_image is image
    assert mapped_label.dtype == torch.long
    assert tuple(mapped_label.shape) == (1, 2, 2, 2)
    assert_tensor_equal("mapped_label", mapped_label, expected_label)


def test_convert_labels_regions(train_module) -> None:
    trainer = object.__new__(train_module.BraTSTrainer)
    labels = torch.tensor(
        [[[[[0, 1], [2, 3]], [[1, 2], [3, 0]]]]],
        dtype=torch.long,
    )

    regions = trainer.convert_labels(labels)
    expected_tc = torch.tensor([[[[[False, True], [False, True]], [[True, False], [True, False]]]]])
    expected_wt = torch.tensor([[[[[False, True], [True, True]], [[True, True], [True, False]]]]])
    expected_et = torch.tensor([[[[[False, False], [False, True]], [[False, False], [True, False]]]]])
    expected = torch.cat([expected_tc, expected_wt, expected_et], dim=1).float()

    assert tuple(regions.shape) == (1, 3, 2, 2, 2)
    assert regions.dtype == torch.float32
    assert_tensor_equal("regions", regions, expected)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--shape", type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()

    train_module = load_train_module()
    tests = [
        test_get_input_label_mapping,
        test_convert_labels_regions,
    ]
    for test in tests:
        test(train_module)
        print(f"PASS {test.__name__}")

    if args.benchmark:
        trainer = object.__new__(train_module.BraTSTrainer)
        d, h, w = args.shape
        image = torch.randn(1, 4, d, h, w)
        seg_values = torch.tensor([-1, 0, 1, 2, 3, 4], dtype=torch.int16)
        seg = seg_values[torch.randint(0, len(seg_values), (1, 1, d, h, w))]
        batch = {"data": image, "seg": seg}

        for _ in range(args.warmup):
            trainer.get_input(batch)
        start = time.perf_counter()
        for _ in range(args.iterations):
            trainer.get_input(batch)
        get_input_ms = (time.perf_counter() - start) * 1000 / args.iterations

        _, label = trainer.get_input(batch)
        label = label[:, None]
        for _ in range(args.warmup):
            trainer.convert_labels(label)
        start = time.perf_counter()
        for _ in range(args.iterations):
            trainer.convert_labels(label)
        convert_ms = (time.perf_counter() - start) * 1000 / args.iterations

        print()
        print(f"benchmark shape=(1, 1, {d}, {h}, {w}), iterations={args.iterations}")
        print(f"get_input:      {get_input_ms:.4f} ms/iter")
        print(f"convert_labels: {convert_ms:.4f} ms/iter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
