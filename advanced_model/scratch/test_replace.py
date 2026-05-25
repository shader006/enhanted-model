import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model_segmamba.segmamba import SegMamba, LayerNorm, DynamicErf, _replace_norm_with_derf

print("Class LayerNorm id:", id(LayerNorm))
print("Class nn.LayerNorm id:", id(nn.LayerNorm))

# Let's inspect a simple model containing LayerNorm
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(10)
        self.ln2 = LayerNorm(10)
        self.sub = nn.Sequential(
            nn.LayerNorm(10),
            LayerNorm(10)
        )

model = SimpleModel()
print("Before replacement:")
print("model.ln1:", type(model.ln1))
print("model.ln2:", type(model.ln2))
print("model.sub[0]:", type(model.sub[0]))
print("model.sub[1]:", type(model.sub[1]))

converted = _replace_norm_with_derf(model)

print("\nAfter replacement:")
print("model.ln1:", type(model.ln1))
print("model.ln2:", type(model.ln2))
print("model.sub[0]:", type(model.sub[0]))
print("model.sub[1]:", type(model.sub[1]))
