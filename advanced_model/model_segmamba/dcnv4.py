import sys
import os
import math
from pathlib import Path
import torch
import torch.nn as nn

DCNV4_DIR = Path(__file__).resolve().parents[2] / "DCNv4" / "DCNv4_op"
if DCNV4_DIR.exists() and str(DCNV4_DIR) not in sys.path:
    sys.path.insert(0, str(DCNV4_DIR))

try:
    import DCNv4
    from DCNv4 import ext  # Verify if C++ extension is compiled
    HAS_REAL_DCNV4 = True
except ImportError:
    HAS_REAL_DCNV4 = False
    sys.path = [p for p in sys.path if 'DCNv4_op' not in p]

class DCNv4_3D(nn.Module):
    _warned = False

    def __init__(self, channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        if HAS_REAL_DCNV4:
            group = channels // 16 if channels >= 16 else 1
            if group == 0: group = 1
            self.dcn = DCNv4.DCNv4(
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                pad=pad,
                group=group,
                without_pointwise=False,
                output_bias=False
            )
        else:
            if not DCNv4_3D._warned:
                print("[!] DCNv4 not found or not compiled. DCNv4_3D is using standard Conv3d as fallback.")
                DCNv4_3D._warned = True
            self.dcn = nn.Conv3d(channels, channels, kernel_size, stride, pad)

    def forward(self, x):
        if HAS_REAL_DCNV4:
            B, C, D, H, W = x.shape
            x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
            x_flat = x_perm.view(B * D, H * W, C)
            out = self.dcn(x_flat, shape=(H, W))
            out_reshaped = out.view(B, D, H, W, C)
            out_final = out_reshaped.permute(0, 4, 1, 2, 3).contiguous()
            return out_final
        else:
            return self.dcn(x)



