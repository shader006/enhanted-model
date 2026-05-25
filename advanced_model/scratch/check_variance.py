import torch
import torch.nn as nn
import numpy as np

class OriginalDynamicErf(nn.Module):
    def __init__(self, dim, alpha_init=0.5, shift_init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.shift = nn.Parameter(torch.tensor([shift_init]))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-6)
        
        x_erf = torch.erf(self.alpha * x_norm + self.shift)
        return x_erf * self.weight + self.bias

class StableDynamicErf(nn.Module):
    def __init__(self, dim, alpha_init=0.5, shift_init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.shift = nn.Parameter(torch.tensor([shift_init]))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 1. Standard normalization
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-6)
        
        # 2. Dynamic Erf transformation
        x_erf = torch.erf(self.alpha * x_norm + self.shift)
        
        # 3. Re-normalization to preserve unit variance (1.0)
        erf_mean = x_erf.mean(-1, keepdim=True)
        erf_var = x_erf.var(-1, keepdim=True, unbiased=False)
        x_erf_norm = (x_erf - erf_mean) / torch.sqrt(erf_var + 1e-6)
        
        return x_erf_norm * self.weight + self.bias

# Simulation
torch.manual_seed(42)
dim = 256
batch_size = 8
seq_len = 128

print("="*60)
print("     ACTIVATION VARIANCE PROPAGATION SIMULATION (50 LAYERS)")
print("="*60)

# Simulate 50 layers of: x = x + layer(x) or just sequential layer(x)
# Let's check sequential normalization layer(x) directly:
x_orig = torch.randn(batch_size, seq_len, dim)
x_stable = x_orig.clone()

print(f"Initial Activation Variance: {x_orig.var().item():.6f}")
print("-" * 60)

orig_net = nn.ModuleList([OriginalDynamicErf(dim) for _ in range(50)])
stable_net = nn.ModuleList([StableDynamicErf(dim) for _ in range(50)])

# We simulate a simplified deep network with residual connections:
# x = x + block(LN(x))
# To make it simple, let's pass it sequentially through the normalization layer
# and a mock linear transformation (which keeps variance stable if input is normalized)
for i in range(50):
    # original
    x_orig = orig_net[i](x_orig)
    # stable
    x_stable = stable_net[i](x_stable)
    
    if (i + 1) in [1, 2, 5, 10, 20, 50]:
        print(f"Layer {i+1:02d}:")
        print(f"  - Original DynamicErf Variance : {x_orig.var().item():.3e}")
        print(f"  - Stable DynamicErf Variance   : {x_stable.var().item():.3f}")
        print("-" * 60)
