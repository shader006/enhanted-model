#!/usr/bin/env python3


import argparse
import json
import random
from pathlib import Path


MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_cases(root: Path):
    """Collect cases that contain 4 modalities + seg label."""
    records = []

    # Expect case directories one level under TrainingData, but scan recursively for robustness.
    case_dirs = sorted([p for p in root.rglob("BraTS-GLI-*") if p.is_dir()])

    for case_dir in case_dirs:
        case_id = case_dir.name

        image_paths = []
        for mod in MODALITIES:
            f = case_dir / f"{case_id}-{mod}.nii.gz"
            if not f.exists():
                image_paths = []
                break
            image_paths.append(str(f.resolve()))

        mask_path = case_dir / f"{case_id}-seg.nii.gz"
        if not image_paths or not mask_path.exists():
            continue

        records.append(
            {
                "id": case_id,
                "image": image_paths,
                "mask": str(mask_path.resolve()),
            }
        )

    return records


def split_records(records, train_ratio=0.7, val_ratio=0.1, seed=42):
    if not records:
        return [], [], []

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    # Safety: keep exact total.
    assert len(train) + len(val) + len(test) == n
    assert len(test) == n_test

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split BraTS2023 into 70/10/20 JSON.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/cuc.buithi/BRATS/BraTS2023_TrainVal"),
        help="Path to BraTS2023_TrainVal root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/cuc.buithi/BRATS/BRATS23/brats23_split_70_10_20.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_json = args.output_json.resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    records = find_cases(data_root)
    train, val, test = split_records(records, train_ratio=0.7, val_ratio=0.1, seed=args.seed)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "data_root": str(data_root),
            "seed": args.seed,
            "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
            "total": len(records),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] JSON created:", output_json)
    print("[COUNT] total={}, train={}, val={}, test={}".format(len(records), len(train), len(val), len(test)))


if __name__ == "__main__":
    main()
