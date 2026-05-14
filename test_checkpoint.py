import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRATS23_DIR = BASE_DIR
SEGMAMBA_DIR = os.path.join(BRATS23_DIR, "SegMamba")
for path in (SEGMAMBA_DIR, BRATS23_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from tqdm import tqdm

import settings
from light_training.dataloading.dataset import MedicalDataset, _resolve_split_paths
from light_training.evaluation.metric import dice, hausdorff_distance_95


# Change this variable when you want to test a specific model checkpoint.
MODEL_PATH = "/home/cuc.buithi/BRATS/BRATS23/Log/SegMamba/segmamba_20260428_171740/checkpoints/best_model_dice_dice0.9203_hd952.2411.pt"
DEFAULT_CHECKPOINT = MODEL_PATH
DEFAULT_DATA_DIR = os.path.join(BRATS23_DIR, "data", "fullres", "train")
DEFAULT_SPLIT_JSON = os.path.join(BRATS23_DIR, "brats23_split_70_10_20.json")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_default_test_root(checkpoint_path):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.join(os.path.dirname(checkpoint_dir), "test")
    return os.path.join(checkpoint_dir, "test")


def create_test_log_dir(checkpoint_path, log_dir=None):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        return os.path.abspath(log_dir)

    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_root = get_default_test_root(checkpoint_path)
    run_log_dir = os.path.join(test_root, f"{timestamp}_{checkpoint_name}")
    os.makedirs(run_log_dir, exist_ok=True)
    return run_log_dir


def start_logging(log_dir):
    log_file = os.path.join(log_dir, "test_checkpoint.log")
    log_handle = open(log_file, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    return log_file, log_handle, original_stdout, original_stderr


def load_state_dict(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    state_dict = {
        (key[7:] if str(key).startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def build_model(device):
    try:
        from model_segmamba.segmamba import SegMamba
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Could not import SegMamba. This model depends on the local mamba/triton "
            "stack, so run it in a CUDA-capable environment with the correct conda env."
        ) from exc

    return SegMamba().to(device)


def convert_brats_regions(labels):
    labels = labels.long()
    labels[labels == -1] = 0
    labels[labels == 4] = 3
    return torch.cat(
        [
            ((labels == 1) | (labels == 3)).float(),
            ((labels == 1) | (labels == 3) | (labels == 2)).float(),
            (labels == 3).float(),
        ],
        dim=1,
    )


def case_metric(gt, pred, voxel_spacing=(1.0, 1.0, 1.0)):
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        return float(dice(pred, gt)), float(hausdorff_distance_95(pred, gt, voxel_spacing=voxel_spacing))
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    return 0.0, 50.0


def get_case_name(properties, fallback):
    if isinstance(properties, list) and properties:
        properties = properties[0]
    if isinstance(properties, dict):
        name = properties.get("name")
        if isinstance(name, (list, tuple)):
            return str(name[0])
        if name is not None:
            return str(name)
    return fallback


def get_test_dataset_from_split_json(data_dir, split_json_file):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not os.path.isfile(split_json_file):
        raise FileNotFoundError(f"Split JSON does not exist: {split_json_file}")

    with open(split_json_file, "r") as f:
        split = json.load(f)
    if "test" not in split:
        raise KeyError(f"Split JSON does not contain a 'test' key: {split_json_file}")
    test_datalist = _resolve_split_paths(data_dir, split["test"])
    print(f"test data is {len(test_datalist)}")
    return MedicalDataset(test_datalist)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a SegMamba checkpoint on the split-json test set.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split_json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--roi_size", nargs=3, type=int, default=settings.INPUT_SIZE)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--log_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = create_test_log_dir(args.checkpoint, args.log_dir)
    log_file, log_handle, original_stdout, original_stderr = start_logging(log_dir)
    try:
        print(f"Test log dir: {log_dir}")
        print(f"Console log: {log_file}")
        print(f"Checkpoint: {args.checkpoint}")

        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

        settings.set_global_reproducibility()
        set_determinism(settings.REPRO_SEED)

        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        if str(device) != args.device:
            print(f"CUDA is not available; using {device}.")

        model = build_model(device)
        load_state_dict(model, args.checkpoint)
        model.eval()

        inferer = SlidingWindowInferer(
            roi_size=args.roi_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            mode="gaussian",
        )

        test_ds = get_test_dataset_from_split_json(args.data_dir, args.split_json)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=str(device).startswith("cuda"),
        )

        rows = []
        region_names = ("TC", "WT", "ET")
        with torch.no_grad():
            progress = tqdm(test_loader, total=len(test_loader), desc="Testing", unit="case")
            for index, batch in enumerate(progress):
                image = batch["data"].to(device).float().contiguous()
                label = batch["seg"].to(device).contiguous()
                if label.ndim == 4:
                    label = label[:, None]

                logits = inferer(image, model)
                pred_label = logits.argmax(dim=1, keepdim=True)
                pred_regions = convert_brats_regions(pred_label).cpu().numpy()[0]
                gt_regions = convert_brats_regions(label).cpu().numpy()[0]

                case_name = get_case_name(batch.get("properties"), f"case_{index:04d}")
                case_values = {"case": case_name}
                for region_index, region_name in enumerate(region_names):
                    dsc, hd95 = case_metric(gt_regions[region_index], pred_regions[region_index])
                    case_values[f"{region_name}_dice"] = dsc
                    case_values[f"{region_name}_hd95"] = hd95
                rows.append(case_values)
                mean_dice = np.mean([case_values[f"{name}_dice"] for name in region_names])
                mean_hd95 = np.mean([case_values[f"{name}_hd95"] for name in region_names])
                progress.set_postfix(
                    case=case_name,
                    mean_dice=f"{mean_dice:.4f}",
                    mean_hd95=f"{mean_hd95:.4f}",
                )

        output_csv = args.output_csv
        if output_csv is None:
            output_csv = os.path.join(log_dir, "test_metrics.csv")
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

        fieldnames = ["case"]
        for name in region_names:
            fieldnames.extend([f"{name}_dice", f"{name}_hd95"])
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("\nSummary")
        print(f"Evaluated cases: {len(rows)}/{len(test_ds)}")
        for name in region_names:
            print(
                f"{name}: dice={np.mean([row[f'{name}_dice'] for row in rows]):.4f}, "
                f"hd95={np.mean([row[f'{name}_hd95'] for row in rows]):.4f}"
            )
        print(
            f"Mean: dice={np.mean([[row[f'{name}_dice'] for name in region_names] for row in rows]):.4f}, "
            f"hd95={np.mean([[row[f'{name}_hd95'] for name in region_names] for row in rows]):.4f}"
        )
        print(f"Saved CSV: {output_csv}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
