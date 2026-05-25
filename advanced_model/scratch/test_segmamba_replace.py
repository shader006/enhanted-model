import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRATS23_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)
sys.path.append(BRATS23_DIR)

from model_segmamba.segmamba import SegMamba, DynamicErf, _replace_norm_with_derf

print("Initializing SegMamba with derf_norm_enabled=False...")
model = SegMamba()

# Let's count norms before replacement
def count_norms(m):
    ln_count = 0
    derf_count = 0
    for name, module in m.named_modules():
        class_name = module.__class__.__name__
        if "LayerNorm" in class_name:
            if isinstance(module, DynamicErf):
                derf_count += 1
            else:
                ln_count += 1
    return ln_count, derf_count

ln_before, derf_before = count_norms(model)
print(f"Before manual replacement: LayerNorm={ln_before}, DynamicErf={derf_before}")

print("Performing manual replacement...")
_replace_norm_with_derf(model)

ln_after, derf_after = count_norms(model)
print(f"After manual replacement: LayerNorm={ln_after}, DynamicErf={derf_after}")
