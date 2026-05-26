import torch
import torch.nn as nn

class FUE(nn.Module):
    def __init__(self, eps=1e-6, use_starrelu=False):
        super().__init__()
        self.eps = eps
        self.activation = nn.Sigmoid()

    def forward(self, x):
        z_bar = self.activation(x.mean(dim=1, keepdim=True)).clamp(min=self.eps)
        uncertainty = -z_bar * torch.log(z_bar)
        return x + x * (1 - uncertainty)
