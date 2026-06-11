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
ADVANCED_MODEL_DIR = os.path.join(BRATS23_DIR, "advanced_model")

# Filter out ADVANCED_MODEL_DIR to prevent importing local monai instead of system monai
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != os.path.abspath(ADVANCED_MODEL_DIR)
]

if BRATS23_DIR not in sys.path:
    sys.path.insert(0, BRATS23_DIR)
sys.path.append(ADVANCED_MODEL_DIR)

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from tqdm import tqdm
import time

import settings
from light_training.dataloading.dataset import MedicalDataset, _resolve_split_paths
from light_training.evaluation.metric import dice, hausdorff_distance_95


# ==============================================================================
# PASTE YOUR CHECKPOINT PATH HERE
# ==============================================================================
CHECKPOINT_PATH = "/home/21013187/BRATS23/BRATS23/enhanted-model/Log/SegMamba/advanced model 128/checkpoints/best_model_hd95_tiebreak_dice0.9017_hd953.0459.pt"
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

import cc3d

def filter_components(mask, min_volume=100):
    if not np.any(mask):
        return mask
    labels = cc3d.connected_components(mask.astype(np.int32))
    counts = np.bincount(labels.flat)
    keep_labels = np.where(counts >= min_volume)[0]
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0 and len(counts) > 1:
        # Fallback: keep only the largest component
        largest_label = np.argmax(counts[1:]) + 1
        keep_labels = np.array([largest_label])
    if len(keep_labels) == 0:
        return np.zeros_like(mask)
    filtered_mask = np.isin(labels, keep_labels).astype(mask.dtype)
    return filtered_mask


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
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split_json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--roi_size", nargs=3, type=int, default=settings.INPUT_SIZE)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to the model checkpoint")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--postprocess", action="store_true", default=True, help="Apply Connected Component post-processing filtering")
    parser.add_argument("--no_postprocess", action="store_false", dest="postprocess", help="Disable post-processing")
    parser.add_argument("--min_volume", type=int, default=100, help="Minimum volume (voxel count) for Connected Component filtering")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    
    # Try to load config.json dynamically to restore exact training architecture / settings
    config_settings = None
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    parent_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(parent_dir, "config.json")
    if os.path.exists(config_path):
        try:
            print(f"Found config.json at {config_path}. Loading training settings...")
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Update paths from config.json if not explicitly provided in command line arguments
            paths = config_data.get("paths", {})
            if "data_dir" in paths and not any(arg.startswith("--data_dir") for arg in sys.argv):
                args.data_dir = paths["data_dir"]
                print(f"Using data directory from config.json: {args.data_dir}")
            if "split_json_file" in paths and not any(arg.startswith("--split_json") for arg in sys.argv):
                args.split_json = paths["split_json_file"]
                print(f"Using split JSON from config.json: {args.split_json}")

            config_settings = config_data.get("settings", {})
            for key, val in config_settings.items():
                if hasattr(settings, key):
                    setattr(settings, key, val)
            print(f"Dynamically updated settings from config.json.")
            
            # Override args.roi_size if defined in config.json and not explicitly provided in argv
            if "INPUT_SIZE" in config_settings:
                if not any(arg.startswith("--roi_size") for arg in sys.argv):
                    args.roi_size = config_settings["INPUT_SIZE"]
                    print(f"Using ROI size from config.json: {args.roi_size}")
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    log_dir = create_test_log_dir(checkpoint_path, args.log_dir)
    log_file, log_handle, original_stdout, original_stderr = start_logging(log_dir)
    try:
        print(f"Test log dir: {log_dir}")
        print(f"Console log: {log_file}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Post-processing (CCA): {args.postprocess}")
        if args.postprocess:
            print(f"Minimum component volume: {args.min_volume} voxels")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        settings.set_global_reproducibility()
        set_determinism(settings.REPRO_SEED)

        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        if str(device) != args.device:
            print(f"CUDA is not available; using {device}.")

        model = build_model(device)
        load_state_dict(model, checkpoint_path)
        model.eval()

        # Compute number of parameters and FLOPs
        num_params = sum(p.numel() for p in model.parameters())
        flops = 0.0
        try:
            from thop import profile
            in_chans = getattr(settings, "SEGMAMBA_IN_CHANS", 4)
            dummy_input = torch.randn(1, in_chans, *args.roi_size, device=device)
            with torch.no_grad():
                macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops = float(macs) * 2.0
            print(f"Model Parameters: {num_params:,}")
            print(f"Model FLOPs (for input size {args.roi_size}): {flops:,}")
        except Exception as e:
            print(f"Warning: Could not compute FLOPs via THOP: {e}")

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

                # Reset CUDA peak memory stats
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                
                # Measure inference time
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                logits = inferer(image, model)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                
                # Retrieve peak VRAM
                if device.type == "cuda":
                    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # in MB
                else:
                    peak_vram = 0.0

                pred_label = logits.argmax(dim=1, keepdim=True)
                pred_regions = convert_brats_regions(pred_label).cpu().numpy()[0]
                gt_regions = convert_brats_regions(label).cpu().numpy()[0]

                if args.postprocess:
                    try:
                        filtered_regions = []
                        for r_idx in range(pred_regions.shape[0]):
                            filtered_regions.append(filter_components(pred_regions[r_idx], min_volume=args.min_volume))
                        pred_regions = np.stack(filtered_regions, axis=0)
                    except Exception as e:
                        print(f"Warning: Failed to apply Connected Component Analysis: {e}")

                properties = batch.get("properties")
                case_name = get_case_name(properties, f"case_{index:04d}")
                
                # Extract voxel spacing from properties if available
                if isinstance(properties, list) and properties:
                    properties = properties[0]
                voxel_spacing = (1.0, 1.0, 1.0)
                if isinstance(properties, dict) and "spacing" in properties:
                    spacing_val = properties["spacing"]
                    if isinstance(spacing_val, (list, tuple, np.ndarray)):
                        voxel_spacing = tuple(float(x) for x in spacing_val)
                    elif torch.is_tensor(spacing_val):
                        voxel_spacing = tuple(float(x) for x in spacing_val.cpu().numpy())

                case_values = {"case": case_name}
                for region_index, region_name in enumerate(region_names):
                    dsc, hd95 = case_metric(
                        gt_regions[region_index],
                        pred_regions[region_index],
                        voxel_spacing=voxel_spacing
                    )
                    case_values[f"{region_name}_dice"] = dsc
                    case_values[f"{region_name}_hd95"] = hd95
                case_values["inference_time"] = elapsed_time
                case_values["vram_peak"] = peak_vram
                case_values["num_params"] = num_params
                case_values["flops"] = flops
                rows.append(case_values)
                mean_dice = np.mean([case_values[f"{name}_dice"] for name in region_names])
                mean_hd95 = np.mean([case_values[f"{name}_hd95"] for name in region_names])
                progress.set_postfix(
                    case=case_name,
                    mean_dice=f"{mean_dice:.4f}",
                    mean_hd95=f"{mean_hd95:.4f}",
                )

        # Calculate summary statistics
        summary_rows = []
        
        # 1. Mean
        mean_row = {"case": "Mean"}
        for name in region_names:
            mean_row[f"{name}_dice"] = float(np.mean([row[f'{name}_dice'] for row in rows]))
            
            # Exclude 50.0 values for HD95 Mean
            hd95_vals = [row[f'{name}_hd95'] for row in rows]
            valid_hd95_vals = [v for v in hd95_vals if v != 50.0]
            mean_row[f"{name}_hd95"] = float(np.mean(valid_hd95_vals)) if valid_hd95_vals else 50.0
            
        mean_row["inference_time"] = float(np.mean([row["inference_time"] for row in rows]))
        mean_row["vram_peak"] = float(np.mean([row["vram_peak"] for row in rows]))
        mean_row["num_params"] = num_params
        mean_row["flops"] = flops
        summary_rows.append(mean_row)
        
        # 2. Std (Standard Deviation)
        std_row = {"case": "Std"}
        for name in region_names:
            std_row[f"{name}_dice"] = float(np.std([row[f'{name}_dice'] for row in rows]))
            
            # Exclude 50.0 values for HD95 Std
            hd95_vals = [row[f'{name}_hd95'] for row in rows]
            valid_hd95_vals = [v for v in hd95_vals if v != 50.0]
            std_row[f"{name}_hd95"] = float(np.std(valid_hd95_vals)) if valid_hd95_vals else 0.0
            
        std_row["inference_time"] = float(np.std([row["inference_time"] for row in rows]))
        std_row["vram_peak"] = float(np.std([row["vram_peak"] for row in rows]))
        std_row["num_params"] = ""
        std_row["flops"] = ""
        summary_rows.append(std_row)
        
        # 3. Median
        median_row = {"case": "Median"}
        for name in region_names:
            median_row[f"{name}_dice"] = float(np.median([row[f'{name}_dice'] for row in rows]))
            
            # Exclude 50.0 values for HD95 Median
            hd95_vals = [row[f'{name}_hd95'] for row in rows]
            valid_hd95_vals = [v for v in hd95_vals if v != 50.0]
            median_row[f"{name}_hd95"] = float(np.median(valid_hd95_vals)) if valid_hd95_vals else 50.0
            
        median_row["inference_time"] = float(np.median([row["inference_time"] for row in rows]))
        median_row["vram_peak"] = float(np.median([row["vram_peak"] for row in rows]))
        median_row["num_params"] = ""
        median_row["flops"] = ""
        summary_rows.append(median_row)

        # 4. Detection Rate
        det_row = {"case": "Detection Rate"}
        for name in region_names:
            hd95_vals = [row[f'{name}_hd95'] for row in rows]
            valid_hd95_vals = [v for v in hd95_vals if v != 50.0]
            det_rate = len(valid_hd95_vals) / len(rows) if rows else 0.0
            det_row[f"{name}_dice"] = ""
            det_row[f"{name}_hd95"] = f"{det_rate * 100:.2f}%"
        det_row["inference_time"] = ""
        det_row["vram_peak"] = ""
        det_row["num_params"] = ""
        det_row["flops"] = ""
        summary_rows.append(det_row)

        output_csv = args.output_csv
        if output_csv is None:
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            suffix = f"_cca{args.min_volume}" if args.postprocess else ""
            output_csv = os.path.join(BASE_DIR, f"test_metrics_{checkpoint_name}{suffix}.csv")
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

        fieldnames = ["case"]
        for name in region_names:
            fieldnames.extend([f"{name}_dice", f"{name}_hd95"])
        fieldnames.extend(["inference_time", "vram_peak", "num_params", "flops"])
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            # Write a blank separator row to distinguish case-by-case data from summary stats
            writer.writerow({f: "" for f in fieldnames})
            writer.writerows(summary_rows)

        print("\nSummary (HD95 stats exclude 50.0 penalty cases)")
        print(f"Evaluated cases: {len(rows)}/{len(test_ds)}")
        for name in region_names:
            det_rate = det_row[f"{name}_hd95"]
            print(
                f"{name}: dice={mean_row[f'{name}_dice']:.4f} (std={std_row[f'{name}_dice']:.4f}, median={median_row[f'{name}_dice']:.4f}), "
                f"hd95={mean_row[f'{name}_hd95']:.4f} (std={std_row[f'{name}_hd95']:.4f}, median={median_row[f'{name}_hd95']:.4f}) [Detection Rate: {det_rate}]"
            )
            
        # Compute case-average metrics first to get correct global statistics
        case_mean_dices = [np.mean([row[f'{name}_dice'] for name in region_names]) for row in rows]
        
        # Exclude 50.0 values from case-averaged HD95 values
        case_mean_hd95s = []
        for row in rows:
            valid_vals = [row[f'{name}_hd95'] for name in region_names if row[f'{name}_hd95'] != 50.0]
            if valid_vals:
                case_mean_hd95s.append(np.mean(valid_vals))
        
        # Compute overall detection rate
        all_hd95_vals = [row[f'{name}_hd95'] for name in region_names for row in rows]
        valid_all_hd95s = [v for v in all_hd95_vals if v != 50.0]
        global_det_rate = len(valid_all_hd95s) / len(all_hd95_vals) if all_hd95_vals else 0.0

        global_mean_dice = float(np.mean(case_mean_dices))
        global_std_dice = float(np.std(case_mean_dices))
        global_median_dice = float(np.median(case_mean_dices))
        
        global_mean_hd95 = float(np.mean(case_mean_hd95s)) if case_mean_hd95s else 50.0
        global_std_hd95 = float(np.std(case_mean_hd95s)) if case_mean_hd95s else 0.0
        global_median_hd95 = float(np.median(case_mean_hd95s)) if case_mean_hd95s else 50.0
        
        print(
            f"Mean: dice={global_mean_dice:.4f} (std={global_std_dice:.4f}, median={global_median_dice:.4f}), "
            f"hd95={global_mean_hd95:.4f} (std={global_std_hd95:.4f}, median={global_median_hd95:.4f}) [Overall Detection Rate: {global_det_rate * 100:.2f}%]"
        )
        print(
            f"Inference Time: mean={mean_row['inference_time']:.4f}s (std={std_row['inference_time']:.4f}s, median={median_row['inference_time']:.4f}s)"
        )
        print(
            f"VRAM Peak: mean={mean_row['vram_peak']:.2f}MB (std={std_row['vram_peak']:.2f}MB, median={median_row['vram_peak']:.2f}MB)"
        )
        print(f"Parameters: {num_params:,}")
        print(f"FLOPs: {flops:,}")
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
