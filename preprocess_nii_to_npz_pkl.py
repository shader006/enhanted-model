#!/usr/bin/env python3


import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SCRIPT_DIR = Path(__file__).resolve().parent
SEGMAMBA_DIR = SCRIPT_DIR / "SegMamba"
if not SEGMAMBA_DIR.exists():
    raise FileNotFoundError(f"SegMamba folder not found at: {SEGMAMBA_DIR}")

# SegMamba preprocessors import from light_training, so include SegMamba root on sys.path.
sys.path.insert(0, str(SEGMAMBA_DIR))

from light_training.preprocessing.preprocessors.preprocessor_mri import (  # pylint: disable=wrong-import-position
    MultiModalityPreprocessor,
)


# Configure your preprocessing inputs here.
BASE_DIR = "../BraTS2023_TrainVal"
IMAGE_DIR = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUTPUT_DIR = "./data/fullres/train"
MODALITIES = ["t2w.nii.gz", "t2f.nii.gz", "t1n.nii.gz", "t1c.nii.gz"]
SEG_FILENAME = "seg.nii.gz"
SPACING = [1.0, 1.0, 1.0]
LABELS = [1, 2, 3]
NUM_PROCESSES = 8
SKIP_PLAN = True


class RobustMultiModalityPreprocessor(MultiModalityPreprocessor):
    """Resolve either short names (t2w.nii.gz) or prefixed names (*-t2w.nii.gz)."""

    def _resolve_case_file(self, case_dir: str, file_name: str) -> str:
        direct_path = os.path.join(case_dir, file_name)
        if os.path.isfile(direct_path):
            return direct_path

        candidates = [
            os.path.join(case_dir, f)
            for f in os.listdir(case_dir)
            if f.endswith(file_name)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple matches for '{file_name}' in {case_dir}: "
                + ", ".join(os.path.basename(c) for c in candidates)
            )
        raise FileNotFoundError(
            f"Cannot find '{file_name}' (or '*-{file_name}') in case folder: {case_dir}"
        )

    def read_data(self, case_name):
        assert len(self.data_filenames) != 0
        case_dir = os.path.join(self.base_dir, self.image_dir, case_name)

        data = []
        spacing = None
        for dfname in self.data_filenames:
            image_path = self._resolve_case_file(case_dir, dfname)
            d = sitk.ReadImage(image_path)
            spacing = d.GetSpacing()
            data.append(sitk.GetArrayFromImage(d).astype(np.float32)[None, ])

        data = np.concatenate(data, axis=0)

        seg_arr = None
        if self.seg_filename != "":
            seg_path = self._resolve_case_file(case_dir, self.seg_filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)
            seg_arr = seg_arr[None]
            intensities_per_channel, intensity_statistics_per_channel = self.collect_foreground_intensities(seg_arr, data)
        else:
            intensities_per_channel = []
            intensity_statistics_per_channel = []

        properties = {
            "spacing": spacing,
            "raw_size": data.shape[1:],
            "name": case_name.split(".")[0],
            "intensities_per_channel": intensities_per_channel,
            "intensity_statistics_per_channel": intensity_statistics_per_channel,
        }
        return data, seg_arr, properties


def main() -> None:
    base_dir = os.path.abspath(BASE_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)
    case_root = os.path.join(base_dir, IMAGE_DIR)

    if not os.path.isdir(case_root):
        raise FileNotFoundError(f"Image directory not found: {case_root}")

    os.makedirs(output_dir, exist_ok=True)

    preprocessor = RobustMultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=IMAGE_DIR,
        data_filenames=MODALITIES,
        seg_filename=SEG_FILENAME,
    )

    if not SKIP_PLAN:
        print("[INFO] Running data analysis plan...")
        preprocessor.run_plan()

    print("[INFO] Running preprocessing to .npz + .pkl...")
    preprocessor.run(
        output_spacing=list(SPACING),
        output_dir=output_dir,
        all_labels=list(LABELS),
        num_processes=NUM_PROCESSES,
    )
    print("[DONE] Preprocessing completed.")


if __name__ == "__main__":
    main()
