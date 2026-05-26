import sys
import os
import math
import importlib
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def _load_ukan_kanlinear():
    kan_path = Path(__file__).resolve().parents[2] / "UKAN-EP" / "main" / "kannet" / "kannet.py"
    if not kan_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("brats23_ukan_kannet", kan_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "KANLinear", None)

KANLinear = _load_ukan_kanlinear()

def _load_unikan_skanlinear():
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    added_to_path = project_root_str not in sys.path
    if added_to_path:
        sys.path.insert(0, project_root_str)
    try:
        module = importlib.import_module("unikan.skan")
        return getattr(module, "SKANLinear_pure", None)
    except ModuleNotFoundError:
        return None
    finally:
        if added_to_path:
            sys.path.remove(project_root_str)

SKANLinear = _load_unikan_skanlinear()

def _load_project_settings():
    settings_path = Path(__file__).resolve().parents[2] / "settings.py"
    if not settings_path.exists():
        return None

    settings_dir = str(settings_path.parent)
    added_to_path = settings_dir not in sys.path
    if added_to_path:
        sys.path.insert(0, settings_dir)

    try:
        spec = importlib.util.spec_from_file_location("brats23_project_settings", settings_path)
        if spec is None or spec.loader is None:
            return None
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        return settings
    finally:
        if added_to_path:
            sys.path.remove(settings_dir)

def _setting_or_default(settings, name, default):
    return getattr(settings, name, default) if settings is not None else default

class StarReLU(nn.Module):
    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1),
            requires_grad=scale_learnable,
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1),
            requires_grad=bias_learnable,
        )

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

def _make_activation(default_kind, use_starrelu=False):
    if use_starrelu:
        return StarReLU()
    if default_kind == "relu":
        return nn.ReLU()
    if default_kind == "gelu":
        return nn.GELU()
    if default_kind == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation kind: {default_kind}")

class DynamicErf(nn.Module):
    def __init__(
        self,
        normalized_shape,
        channels_last=True,
        elementwise_affine=True,
        alpha_init_value=0.5,
        shift_init_value=0.0,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.channels_last = channels_last
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.channels_last:
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-6)
        else:
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-6)

        alpha = torch.clamp(self.alpha, min=1e-4, max=10.0)
        shift = torch.clamp(self.shift, min=-5.0, max=5.0)

        x_erf = torch.erf(alpha * x_norm + shift)

        if self.channels_last:
            erf_mean = x_erf.mean(-1, keepdim=True)
            erf_var = x_erf.var(-1, keepdim=True, unbiased=False)
            x_erf_norm = (x_erf - erf_mean) / torch.sqrt(erf_var + 1e-6)
        else:
            erf_mean = x_erf.mean(1, keepdim=True)
            erf_var = x_erf.var(1, keepdim=True, unbiased=False)
            x_erf_norm = (x_erf - erf_mean) / torch.sqrt(erf_var + 1e-6)

        if not self.elementwise_affine:
            return x_erf_norm
        if self.channels_last:
            return x_erf_norm * self.weight + self.bias
        view_shape = (1, self.normalized_shape[0]) + (1,) * (x.ndim - 2)
        weight = self.weight.view(view_shape)
        bias = self.bias.view(view_shape)
        return x_erf_norm * weight + bias

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, channels_last={self.channels_last}, "
            f"elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}, "
            f"shift_init_value={self.shift_init_value}"
        )

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

def _replace_norm_with_derf(module, alpha_init_value=0.5, shift_init_value=0.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicErf(
            normalized_shape=module.normalized_shape,
            channels_last=True,
            elementwise_affine=module.elementwise_affine,
            alpha_init_value=alpha_init_value,
            shift_init_value=shift_init_value,
        )
    elif isinstance(module, LayerNorm):
        module_output = DynamicErf(
            normalized_shape=module.normalized_shape[0],
            channels_last=module.data_format == "channels_last",
            elementwise_affine=True,
            alpha_init_value=alpha_init_value,
            shift_init_value=shift_init_value,
        )
    for name, child in module.named_children():
        module_output.add_module(
            name,
            _replace_norm_with_derf(
                child,
                alpha_init_value=alpha_init_value,
                shift_init_value=shift_init_value,
            ),
        )
    del module
    return module_output

def _norm3d(channels, norm_name="instance"):
    if isinstance(norm_name, str) and norm_name.lower() in {"batch", "batchnorm", "batchnorm3d"}:
        return nn.BatchNorm3d(channels)
    return nn.InstanceNorm3d(channels)

def _valid_group_count(*channels, requested=16):
    requested = max(1, int(requested))
    max_group = min(requested, *(int(channel) for channel in channels))
    for group in range(max_group, 0, -1):
        if all(int(channel) % group == 0 for channel in channels):
            return group
    return 1

def _part1by2(n: torch.Tensor):
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n

def _morton3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)

@torch.no_grad()
def _morton_perm_3d(d: int, h: int, w: int, device: torch.device):
    dd = torch.arange(d, device=device, dtype=torch.int32)
    hh = torch.arange(h, device=device, dtype=torch.int32)
    ww = torch.arange(w, device=device, dtype=torch.int32)
    try:
        zz, yy, xx = torch.meshgrid(dd, hh, ww, indexing="ij")
    except TypeError:
        zz, yy, xx = torch.meshgrid(dd, hh, ww)
    keys = _morton3d(zz.reshape(-1), yy.reshape(-1), xx.reshape(-1))
    perm = torch.argsort(keys)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    return perm.long(), inv.long()

def _conv_output_size(size, kernel_size, stride, padding=0, dilation=1):
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

def _feature_depths_from_input_size(input_size, num_stages=4):
    if input_size is None:
        return [64, 32, 16, 8]
    if isinstance(input_size, int):
        depth = input_size
    else:
        if len(input_size) < 1:
            raise ValueError("input_size must contain at least the depth dimension.")
        depth = int(input_size[0])

    feature_depths = []
    depth = _conv_output_size(depth, kernel_size=7, stride=2, padding=3)
    feature_depths.append(depth)
    for _ in range(1, num_stages):
        depth = _conv_output_size(depth, kernel_size=2, stride=2)
        feature_depths.append(depth)
    return feature_depths
