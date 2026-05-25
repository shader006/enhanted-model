import torch
import torch.nn as nn

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

class MockResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer
        # A simple linear layer that preserves variance when input is normalized
        self.linear = nn.Linear(dim, dim, bias=False)
        # Initialize linear weight with small scale as typical in residual blocks (e.g. LayerScale or standard init)
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, x):
        # Standard residual block: x = x + block(norm(x))
        return x + self.linear(self.norm(x))

# Simulation
torch.manual_seed(42)
dim = 256
batch_size = 8
seq_len = 128

x_orig = torch.randn(batch_size, seq_len, dim)
x_stable = x_orig.clone()

print("="*60)
print("     RESIDUAL STREAM VARIANCE SIMULATION (50 LAYERS)")
print("="*60)
print(f"Initial Residual Stream Variance: {x_orig.var().item():.6f}")
print("-" * 60)

orig_blocks = nn.ModuleList([MockResidualBlock(dim, OriginalDynamicErf(dim)) for _ in range(50)])
stable_blocks = nn.ModuleList([MockResidualBlock(dim, StableDynamicErf(dim)) for _ in range(50)])

for i in range(50):
    x_orig = orig_blocks[i](x_orig)
    x_stable = stable_blocks[i](x_stable)
    
    if (i + 1) in [1, 2, 5, 10, 20, 50]:
        print(f"Layer {i+1:02d}:")
        print(f"  - Original Residual Stream Variance : {x_orig.var().item():.3f}")
        print(f"  - Stable Residual Stream Variance   : {x_stable.var().item():.3f}")
        print("-" * 60)
