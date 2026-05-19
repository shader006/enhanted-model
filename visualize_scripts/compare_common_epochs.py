#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_RE = re.compile(r"mean_dice is ([0-9.]+), mean_hd95 is ([0-9.]+)")


def parse_log(log_path):
    epochs = []
    with Path(log_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = METRIC_RE.search(line)
            if not match:
                continue
            epochs.append(
                {
                    "epoch": len(epochs),
                    "mean_dice": float(match.group(1)),
                    "mean_hd95": float(match.group(2)),
                }
            )
    if not epochs:
        raise ValueError(f"No validation metrics found in {log_path}")
    return epochs


def default_label(log_path):
    path = Path(log_path)
    if path.name == "terminal.log":
        return path.parent.name
    return path.stem


def write_csv(rows, output_path):
    fieldnames = [
        "epoch",
        "log1_dice",
        "log2_dice",
        "dice_diff_log2_minus_log1",
        "log1_hd95",
        "log2_hd95",
        "hd95_diff_log2_minus_log1",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_common_epochs(rows, label1, label2, output_path, title):
    epochs = [row["epoch"] for row in rows]
    log1_dice = [row["log1_dice"] for row in rows]
    log2_dice = [row["log2_dice"] for row in rows]
    log1_hd95 = [row["log1_hd95"] for row in rows]
    log2_hd95 = [row["log2_hd95"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    axes[0].plot(epochs, log1_dice, label=label1, linewidth=2)
    axes[0].plot(epochs, log2_dice, label=label2, linewidth=2)
    axes[0].set_ylabel("Mean Dice")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, log1_hd95, label=label1, linewidth=2)
    axes[1].plot(epochs, log2_hd95, label=label2, linewidth=2)
    axes[1].set_xlabel("Common Epoch")
    axes[1].set_ylabel("Mean HD95")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def best_by(rows, key, mode):
    if mode == "max":
        return max(rows, key=lambda row: row[key])
    if mode == "min":
        return min(rows, key=lambda row: row[key])
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare mean Dice and mean HD95 from two terminal.log files over common epochs."
    )
    parser.add_argument("log1", help="Path to first terminal.log")
    parser.add_argument("log2", help="Path to second terminal.log")
    parser.add_argument("--label1", default=None, help="Plot label for first log")
    parser.add_argument("--label2", default=None, help="Plot label for second log")
    parser.add_argument(
        "--outdir",
        default="visualize_outputs",
        help="Output directory for PNG and CSV files",
    )
    parser.add_argument(
        "--prefix",
        default="common_epoch_comparison",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    log1_path = Path(args.log1)
    log2_path = Path(args.log2)
    if not log1_path.is_file():
        raise FileNotFoundError(f"Missing log1: {log1_path}")
    if not log2_path.is_file():
        raise FileNotFoundError(f"Missing log2: {log2_path}")

    label1 = args.label1 or default_label(log1_path)
    label2 = args.label2 or default_label(log2_path)

    log1 = parse_log(log1_path)
    log2 = parse_log(log2_path)
    common_epochs = min(len(log1), len(log2))
    rows = []
    for epoch in range(common_epochs):
        d1 = log1[epoch]["mean_dice"]
        d2 = log2[epoch]["mean_dice"]
        h1 = log1[epoch]["mean_hd95"]
        h2 = log2[epoch]["mean_hd95"]
        rows.append(
            {
                "epoch": epoch,
                "log1_dice": d1,
                "log2_dice": d2,
                "dice_diff_log2_minus_log1": d2 - d1,
                "log1_hd95": h1,
                "log2_hd95": h2,
                "hd95_diff_log2_minus_log1": h2 - h1,
            }
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{args.prefix}.png"
    csv_path = outdir / f"{args.prefix}.csv"

    title = f"{label1} vs {label2} (common epochs 0..{common_epochs - 1})"
    plot_common_epochs(rows, label1, label2, png_path, title)
    write_csv(rows, csv_path)

    best_log1_dice = best_by(rows, "log1_dice", "max")
    best_log2_dice = best_by(rows, "log2_dice", "max")
    best_log1_hd95 = best_by(rows, "log1_hd95", "min")
    best_log2_hd95 = best_by(rows, "log2_hd95", "min")

    print(f"Parsed {len(log1)} epochs from {log1_path}")
    print(f"Parsed {len(log2)} epochs from {log2_path}")
    print(f"Compared common epochs: 0..{common_epochs - 1} ({common_epochs} epochs)")
    print(f"Saved plot: {png_path}")
    print(f"Saved CSV: {csv_path}")
    print()
    print("Best mean Dice in common epochs:")
    print(f"  {label1}: epoch {best_log1_dice['epoch']}, dice {best_log1_dice['log1_dice']:.6f}, hd95 {best_log1_dice['log1_hd95']:.6f}")
    print(f"  {label2}: epoch {best_log2_dice['epoch']}, dice {best_log2_dice['log2_dice']:.6f}, hd95 {best_log2_dice['log2_hd95']:.6f}")
    print("Best mean HD95 in common epochs:")
    print(f"  {label1}: epoch {best_log1_hd95['epoch']}, dice {best_log1_hd95['log1_dice']:.6f}, hd95 {best_log1_hd95['log1_hd95']:.6f}")
    print(f"  {label2}: epoch {best_log2_hd95['epoch']}, dice {best_log2_hd95['log2_dice']:.6f}, hd95 {best_log2_hd95['log2_hd95']:.6f}")


if __name__ == "__main__":
    main()
