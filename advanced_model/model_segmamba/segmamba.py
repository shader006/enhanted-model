# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import importlib.util
import sys
import importlib
from pathlib import Path

import torch.nn as nn
import torch 

LOCAL_MAMBA_DIR = Path(__file__).resolve().parents[1] / "mamba"
if LOCAL_MAMBA_DIR.exists() and str(LOCAL_MAMBA_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAMBA_DIR))

SWINDER_DIR = Path(__file__).resolve().parents[2] / "Swin-DER"
if SWINDER_DIR.exists() and str(SWINDER_DIR) not in sys.path:
    sys.path.insert(0, str(SWINDER_DIR))

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrUpBlock
from monai.networks.nets.swin_unetr import get_window_size, window_partition, window_reverse
from mamba_ssm import Mamba
import torch.nn.functional as F 

try:
    from SwinDER.upsample.onsampling import Onsampling
except ModuleNotFoundError:
    Onsampling = None


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
        x = torch.erf(self.alpha * x + self.shift)
        if not self.elementwise_affine:
            return x
        if self.channels_last:
            return x * self.weight + self.bias
        view_shape = (1, self.normalized_shape[0]) + (1,) * (x.ndim - 2)
        weight = self.weight.view(view_shape)
        bias = self.bias.view(view_shape)
        return x * weight + bias

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, channels_last={self.channels_last}, "
            f"elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}, "
            f"shift_init_value={self.shift_init_value}"
        )


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

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
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

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, use_starrelu=False):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = _make_activation("gelu", use_starrelu=use_starrelu)
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles, use_starrelu=False) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = _make_activation("relu", use_starrelu=use_starrelu)

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = _make_activation("relu", use_starrelu=use_starrelu)

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = _make_activation("relu", use_starrelu=use_starrelu)

        self.proj4 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = _make_activation("relu", use_starrelu=use_starrelu)

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 * x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class TSMambaLayer(nn.Module):
    def __init__(self, dim, num_slices=None, mlp_ratio=2, use_starrelu=False):
        super().__init__()
        self.gsc = GSC(dim, use_starrelu=use_starrelu)
        self.tom = MambaLayer(dim=dim, num_slices=num_slices)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(dim, mlp_ratio * dim, use_starrelu=use_starrelu)

    def forward(self, x):
        x = self.gsc(x)
        x = self.tom(x)
        x = x + self.mlp(self.norm(x))
        return x


class GSCOnlyLayer(nn.Module):
    def __init__(self, dim, use_starrelu=False):
        super().__init__()
        self.gsc = GSC(dim, use_starrelu=use_starrelu)

    def forward(self, x):
        return self.gsc(x)

class FUE(nn.Module):
    def __init__(self, eps=1e-6, use_starrelu=False):
        super().__init__()
        self.eps = eps
        self.activation = nn.Sigmoid()

    def forward(self, x):
        z_bar = self.activation(x.mean(dim=1, keepdim=True)).clamp(min=self.eps)
        uncertainty = -z_bar * torch.log(z_bar)
        return x + x * (1 - uncertainty)


def _norm3d(channels, norm_name="instance"):
    if isinstance(norm_name, str) and norm_name.lower() in {"batch", "batchnorm", "batchnorm3d"}:
        return nn.BatchNorm3d(channels)
    return nn.InstanceNorm3d(channels)


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
    zz, yy, xx = torch.meshgrid(dd, hh, ww, indexing="ij")
    keys = _morton3d(zz.reshape(-1), yy.reshape(-1), xx.reshape(-1))
    perm = torch.argsort(keys)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    return perm.long(), inv.long()


class Pseudo3DBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
        bottleneck_ratio=4,
        res_block=True,
        use_starrelu=False,
        z_window_enabled=False,
        z_window_size=(4, 4, 4),
    ):
        super().__init__()
        hidden_channels = max(out_channels // bottleneck_ratio, 8)
        self.res_block = res_block
        activation = _make_activation("gelu", use_starrelu=use_starrelu)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _norm3d(hidden_channels, norm_name),
            activation,
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu", use_starrelu=use_starrelu),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu", use_starrelu=use_starrelu),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _norm3d(out_channels, norm_name),
        )
        self.act = _make_activation("gelu", use_starrelu=use_starrelu)

    def forward(self, x):
        out = self.conv(x)
        if self.res_block:
            out = out + self.proj(x)
        return self.act(out)


class TokenKANPseudo3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
        bottleneck_ratio=4,
        res_block=True,
        use_starrelu=False,
        z_window_enabled=False,
        z_window_size=(4, 4, 4),
        linear_cls=None,
        linear_name="KANLinear",
    ):
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(
                f"{linear_name} could not be loaded for TokenKANPseudo3DBlock."
            )

        self.res_block = res_block
        self.linear_cls = linear_cls
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.token_norm1 = nn.LayerNorm(in_channels)
        self.fc1 = self.linear_cls(in_channels, out_channels)
        self.token_norm2 = nn.LayerNorm(out_channels)
        self.mixer1 = Pseudo3DBottleneckBlock(
            out_channels,
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            res_block=False,
            use_starrelu=False,
        )
        self.token_norm3 = nn.LayerNorm(out_channels)
        self.fc2 = self.linear_cls(out_channels, out_channels)
        self.mixer2 = Pseudo3DBottleneckBlock(
            out_channels,
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            res_block=False,
            use_starrelu=False,
        )
        self.out_norm = _norm3d(out_channels, norm_name)
        self.act = _make_activation("gelu", use_starrelu=False)
        self.z_window_enabled = bool(z_window_enabled)
        self.z_window_size = tuple(int(size) for size in z_window_size)
        self._morton_cache = {}

    def _volume_to_tokens(self, x):
        b, c, d, h, w = x.shape
        if not self.z_window_enabled:
            tokens = x.reshape(b, c, d * h * w).transpose(1, 2).contiguous()
            return tokens, {"mode": "global", "spatial_shape": (d, h, w)}

        window_size = get_window_size((d, h, w), self.z_window_size)
        pad_d = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - w % window_size[2]) % window_size[2]
        x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        padded_shape = x_padded.shape[2:]
        windows = window_partition(x_padded.permute(0, 2, 3, 4, 1).contiguous(), window_size)
        perm, inv = self._get_morton_perm(window_size, x.device)
        tokens = windows[:, perm, :].contiguous()
        return tokens, {
            "mode": "z_window",
            "batch_size": b,
            "spatial_shape": (d, h, w),
            "padded_shape": padded_shape,
            "window_size": window_size,
            "inverse_perm": inv,
        }

    def _get_morton_perm(self, window_size, device):
        key = tuple(window_size)
        if key not in self._morton_cache:
            self._morton_cache[key] = _morton_perm_3d(*window_size, device=torch.device("cpu"))
        perm, inv = self._morton_cache[key]
        if perm.device != device:
            perm = perm.to(device=device)
            inv = inv.to(device=device)
        return perm, inv

    @staticmethod
    def _global_tokens_to_volume(tokens, spatial_shape):
        b, n, c = tokens.shape
        d, h, w = spatial_shape
        return tokens.transpose(1, 2).reshape(b, c, d, h, w).contiguous()

    @staticmethod
    def _window_tokens_to_volume(tokens, token_meta):
        inverse_perm = token_meta["inverse_perm"]
        batch_size = token_meta["batch_size"]
        padded_shape = token_meta["padded_shape"]
        spatial_shape = token_meta["spatial_shape"]
        window_size = token_meta["window_size"]
        windows = tokens[:, inverse_perm, :].contiguous()
        volume = window_reverse(windows, window_size, (batch_size, *padded_shape))
        volume = volume.permute(0, 4, 1, 2, 3).contiguous()
        d, h, w = spatial_shape
        return volume[:, :, :d, :h, :w].contiguous()

    @staticmethod
    def _tokens_to_volume(tokens, token_meta):
        if token_meta["mode"] == "global":
            return TokenKANPseudo3DBlock._global_tokens_to_volume(tokens, token_meta["spatial_shape"])
        if token_meta["mode"] == "z_window":
            return TokenKANPseudo3DBlock._window_tokens_to_volume(tokens, token_meta)
        raise ValueError(f"Unsupported token reconstruction mode: {token_meta['mode']}")

    def forward(self, x):
        residual = self.proj(x)
        tokens, token_meta = self._volume_to_tokens(x)
        tokens = self.token_norm1(tokens)
        tokens = self.fc1(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = self.token_norm2(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.mixer1(out)

        tokens, token_meta = self._volume_to_tokens(out)
        tokens = self.token_norm3(tokens)
        tokens = self.fc2(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape[0], tokens.shape[1], -1)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.mixer2(out)
        out = self.out_norm(out)
        if self.res_block:
            out = out + residual
        return self.act(out)


class TokenSKANPseudo3DBlock(TokenKANPseudo3DBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, linear_cls=SKANLinear, linear_name="SKANLinear_pure", **kwargs)


class Pseudo3DUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        upsample_kernel_size,
        norm_name="instance",
        res_block=True,
        upsample_mode="transconv",
        use_starrelu=False,
    ):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("Pseudo3DUpBlock only supports spatial_dims=3.")
        if upsample_mode == "onsampling":
            if Onsampling is None:
                raise ModuleNotFoundError("Onsampling is unavailable. Ensure Swin-DER is present in the project.")
            self.upsample = Onsampling(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                dyscope=True,
            )
        else:
            self.upsample = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
            )
        self.conv_block = Pseudo3DBottleneckBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=use_starrelu,
        )

    def forward(self, inp, skip):
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class TokenKANPseudo3DUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        upsample_kernel_size,
        norm_name="instance",
        res_block=True,
        upsample_mode="transconv",
        use_starrelu=False,
        z_window_enabled=False,
        z_window_size=(4, 4, 4),
    ):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("TokenKANPseudo3DUpBlock only supports spatial_dims=3.")
        if upsample_mode == "onsampling":
            if Onsampling is None:
                raise ModuleNotFoundError("Onsampling is unavailable. Ensure Swin-DER is present in the project.")
            self.upsample = Onsampling(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                dyscope=True,
            )
        else:
            self.upsample = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
            )
        self.conv_block = TokenKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=use_starrelu,
            z_window_enabled=z_window_enabled,
            z_window_size=z_window_size,
        )

    def forward(self, inp, skip):
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class TokenSKANPseudo3DUpBlock(TokenKANPseudo3DUpBlock):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        upsample_kernel_size,
        norm_name="instance",
        res_block=True,
        upsample_mode="transconv",
        use_starrelu=False,
        z_window_enabled=False,
        z_window_size=(4, 4, 4),
    ):
        nn.Module.__init__(self)
        if spatial_dims != 3:
            raise ValueError("TokenSKANPseudo3DUpBlock only supports spatial_dims=3.")
        if upsample_mode == "onsampling":
            if Onsampling is None:
                raise ModuleNotFoundError("Onsampling is unavailable. Ensure Swin-DER is present in the project.")
            self.upsample = Onsampling(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                dyscope=True,
            )
        else:
            self.upsample = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
            )
        self.conv_block = TokenSKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=use_starrelu,
            z_window_enabled=z_window_enabled,
            z_window_size=z_window_size,
        )


def _make_decoder_block(
    upsample_mode,
    spatial_dims,
    in_channels,
    out_channels,
    norm_name,
    res_block,
    use_starrelu=False,
):
    if upsample_mode == "onsampling":
        return Pseudo3DUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            upsample_mode=upsample_mode,
            use_starrelu=use_starrelu,
        )
    return UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=res_block,
    )


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


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], input_size=None,
                 mamba_stages=None, use_starrelu=False):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, groups=in_chans),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        num_slices_list = _feature_depths_from_input_size(input_size, num_stages=4)
        self.num_slices_list = num_slices_list
        valid_stage_indices = set(range(len(dims)))
        if mamba_stages is None:
            mamba_stage_set = valid_stage_indices
        else:
            mamba_stage_set = {int(stage_idx) for stage_idx in mamba_stages}
            invalid_indices = sorted(mamba_stage_set - valid_stage_indices)
            if invalid_indices:
                raise ValueError(f"mamba_stages contains invalid stage indices: {invalid_indices}")
        self.mamba_stages = sorted(mamba_stage_set)
        for i in range(4):
            block_cls = TSMambaLayer if i in mamba_stage_set else GSCOnlyLayer
            stage = nn.Sequential(
                *[
                    block_cls(dim=dims[i], num_slices=num_slices_list[i], use_starrelu=False)
                    if block_cls is TSMambaLayer
                    else block_cls(dim=dims[i], use_starrelu=use_starrelu)
                    for j in range(depths[i])
                ]
            )

            self.stages.append(stage)

        self.out_indices = out_indices

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=None,
        out_chans=None,
        depths=None,
        feat_size=None,
        drop_path_rate=None,
        layer_scale_init_value=None,
        hidden_size=None,
        norm_name = None,
        conv_block=None,
        res_block=None,
        spatial_dims=None,
        input_size=None,
        mamba_stages=None,
        starrelu_enabled=None,
        kan_enabled=None,
        skan_enabled=None,
        kan_z_window_enabled=None,
        kan_z_window_size=None,
        use_settings: bool = True,
    ) -> None:
        super().__init__()

        settings = _load_project_settings() if use_settings else None
        in_chans = in_chans if in_chans is not None else _setting_or_default(settings, "SEGMAMBA_IN_CHANS", 1)
        out_chans = out_chans if out_chans is not None else _setting_or_default(settings, "SEGMAMBA_OUT_CHANS", 13)
        depths = depths if depths is not None else _setting_or_default(settings, "SEGMAMBA_DEPTHS", [2, 2, 2, 2])
        feat_size = feat_size if feat_size is not None else _setting_or_default(settings, "SEGMAMBA_FEAT_SIZE", [48, 96, 192, 384])
        drop_path_rate = drop_path_rate if drop_path_rate is not None else _setting_or_default(settings, "SEGMAMBA_DROP_PATH_RATE", 0)
        layer_scale_init_value = (
            layer_scale_init_value
            if layer_scale_init_value is not None
            else _setting_or_default(settings, "SEGMAMBA_LAYER_SCALE_INIT_VALUE", 1e-6)
        )
        hidden_size = hidden_size if hidden_size is not None else _setting_or_default(settings, "SEGMAMBA_HIDDEN_SIZE", 768)
        norm_name = norm_name if norm_name is not None else _setting_or_default(settings, "SEGMAMBA_NORM_NAME", "instance")
        conv_block = conv_block if conv_block is not None else _setting_or_default(settings, "SEGMAMBA_CONV_BLOCK", True)
        res_block = res_block if res_block is not None else _setting_or_default(settings, "SEGMAMBA_RES_BLOCK", True)
        kan_enabled = (
            kan_enabled
            if kan_enabled is not None
            else _setting_or_default(settings, "SEGMAMBA_KAN", False)
        )
        skan_enabled = (
            skan_enabled
            if skan_enabled is not None
            else _setting_or_default(settings, "SEGMAMBA_SKAN", False)
        )
        kan_z_window_enabled = (
            kan_z_window_enabled
            if kan_z_window_enabled is not None
            else _setting_or_default(settings, "SEGMAMBA_KAN_Z_WINDOW", False)
        )
        kan_z_window_size = (
            kan_z_window_size
            if kan_z_window_size is not None
            else _setting_or_default(settings, "SEGMAMBA_KAN_Z_WINDOW_SIZE", (4, 4, 4))
        )
        spatial_dims = spatial_dims if spatial_dims is not None else _setting_or_default(settings, "SEGMAMBA_SPATIAL_DIMS", 3)
        input_size = input_size if input_size is not None else _setting_or_default(settings, "INPUT_SIZE", [128, 128, 128])
        mamba_stages = mamba_stages if mamba_stages is not None else _setting_or_default(settings, "SEGMAMBA_MAMBA_STAGES", [0, 1, 2, 3])
        starrelu_enabled = starrelu_enabled if starrelu_enabled is not None else _setting_or_default(settings, "SEGMAMBA_STARRELU", False)
        upsample_mode = "onsampling" if _setting_or_default(settings, "SEGMAMBA_ONSAMPLING", False) else "transconv"
        derf_norm_enabled = _setting_or_default(settings, "SEGMAMBA_DERF_NORM_ENABLED", False)
        derf_alpha_init_value = _setting_or_default(settings, "SEGMAMBA_DERF_ALPHA_INIT_VALUE", 0.5)
        derf_shift_init_value = _setting_or_default(settings, "SEGMAMBA_DERF_SHIFT_INIT_VALUE", 0.0)

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.input_size = input_size
        self.upsample_mode = upsample_mode
        self.mamba_stages = list(mamba_stages)
        self.starrelu_enabled = bool(starrelu_enabled)
        self.kan_enabled = bool(kan_enabled)
        self.skan_enabled = bool(skan_enabled)
        self.kan_z_window_enabled = bool(kan_z_window_enabled)
        self.kan_z_window_size = tuple(int(size) for size in kan_z_window_size)

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                input_size=input_size,
                                mamba_stages=mamba_stages,
                                use_starrelu=self.starrelu_enabled,
                              )
        self.encoder1 = Pseudo3DBottleneckBlock(
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.encoder2 = Pseudo3DBottleneckBlock(
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.encoder3 = Pseudo3DBottleneckBlock(
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.encoder4 = Pseudo3DBottleneckBlock(
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )

        use_token_kan = self.kan_enabled or self.skan_enabled
        if self.skan_enabled:
            late_encoder_block = TokenSKANPseudo3DBlock
        elif self.kan_enabled:
            late_encoder_block = TokenKANPseudo3DBlock
        else:
            late_encoder_block = Pseudo3DBottleneckBlock
        self.encoder5 = late_encoder_block(
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
            z_window_enabled=self.kan_z_window_enabled,
            z_window_size=self.kan_z_window_size,
        )
        self.bottleneck_downsample = nn.Sequential(
            nn.InstanceNorm3d(self.feat_size[3]),
            nn.Conv3d(self.feat_size[3], self.hidden_size, kernel_size=2, stride=2),
        )
        self.encoder6 = late_encoder_block(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
            z_window_enabled=self.kan_z_window_enabled,
            z_window_size=self.kan_z_window_size,
        )

        if use_token_kan:
            decoder_block_cls = TokenSKANPseudo3DUpBlock if self.skan_enabled else TokenKANPseudo3DUpBlock
            self.decoder5 = decoder_block_cls(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[3],
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                upsample_mode=upsample_mode,
                use_starrelu=self.starrelu_enabled,
                z_window_enabled=self.kan_z_window_enabled,
                z_window_size=self.kan_z_window_size,
            )
            self.decoder4 = decoder_block_cls(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[2],
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                upsample_mode=upsample_mode,
                use_starrelu=self.starrelu_enabled,
                z_window_enabled=self.kan_z_window_enabled,
                z_window_size=self.kan_z_window_size,
            )
        else:
            self.decoder5 = _make_decoder_block(
                upsample_mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[3],
                norm_name=norm_name,
                res_block=res_block,
                use_starrelu=self.starrelu_enabled,
            )
            self.decoder4 = _make_decoder_block(
                upsample_mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[2],
                norm_name=norm_name,
                res_block=res_block,
                use_starrelu=self.starrelu_enabled,
            )
        self.decoder3 = _make_decoder_block(
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.decoder2 = _make_decoder_block(
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.decoder1 = _make_decoder_block(
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=self.starrelu_enabled,
        )
        self.fue1 = FUE(use_starrelu=self.starrelu_enabled)
        self.fue2 = FUE(use_starrelu=self.starrelu_enabled)
        self.fue3 = FUE(use_starrelu=self.starrelu_enabled)
        self.fue4 = FUE(use_starrelu=self.starrelu_enabled)
        self.fue5 = FUE(use_starrelu=self.starrelu_enabled)
        self.fue6 = FUE(use_starrelu=self.starrelu_enabled)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)
        if derf_norm_enabled:
            converted = _replace_norm_with_derf(
                self,
                alpha_init_value=derf_alpha_init_value,
                shift_init_value=derf_shift_init_value,
            )
            self.__dict__.update(converted.__dict__)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.fue1(self.encoder1(x_in))  #z0
        enc2 = self.fue2(self.encoder2(outs[0])) #z1
        enc3 = self.fue3(self.encoder3(outs[1])) #z2
        enc4 = self.fue4(self.encoder4(outs[2])) #z3
        enc5 = self.fue5(self.encoder5(outs[3])) #z4
        enc_hidden = self.fue6(self.encoder6(self.bottleneck_downsample(outs[3])))
        dec3 = self.decoder5(enc_hidden, enc5)
        dec2 = self.decoder4(dec3, enc4)
        dec1 = self.decoder3(dec2, enc3)
        dec0 = self.decoder2(dec1, enc2)
        out = self.decoder1(dec0, enc1)
                
        return self.out(out)
    
