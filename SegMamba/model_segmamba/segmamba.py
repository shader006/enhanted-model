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
from pathlib import Path

import torch.nn as nn
import torch 

LOCAL_MAMBA_DIR = Path(__file__).resolve().parents[1] / "mamba"
if LOCAL_MAMBA_DIR.exists() and str(LOCAL_MAMBA_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAMBA_DIR))

from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 


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
        # 1. Standard normalization of input
        if self.channels_last:
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-6)
        else:
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-6)

        # 2. Clamped learnable parameters to ensure numeric stability and prevent explosion/saturation
        alpha = torch.clamp(self.alpha, min=1e-4, max=10.0)
        shift = torch.clamp(self.shift, min=-5.0, max=5.0)

        # 3. Dynamic Erf non-linear transformation
        x_erf = torch.erf(alpha * x_norm + shift)

        # 4. Re-normalization of Erf output to guarantee unit variance (1.0) and prevent scale decay
        if self.channels_last:
            erf_mean = x_erf.mean(-1, keepdim=True)
            erf_var = x_erf.var(-1, keepdim=True, unbiased=False)
            x_erf_norm = (x_erf - erf_mean) / torch.sqrt(erf_var + 1e-6)
        else:
            erf_mean = x_erf.mean(1, keepdim=True)
            erf_var = x_erf.var(1, keepdim=True, unbiased=False)
            x_erf_norm = (x_erf - erf_mean) / torch.sqrt(erf_var + 1e-6)

        # 5. Elementwise affine (weight and bias)
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
        )
    
    def mamba_forward(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)

        return x

    def forward(self, x):
        assert x.shape[1] == self.dim
        x_skip = x

        out_x_1 = self.mamba_forward(x)

        x_2 = rearrange(x, "b c d w h -> b c w d h")
        out_x_2 = self.mamba_forward(x_2)
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")

        x_3 = rearrange(x, "b c d w h -> b c h w d")
        out_x_3 = self.mamba_forward(x_3)
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")

        out = out_x_1 + out_x_2 + out_x_3
        out = out + x_skip
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.conv1 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.out_conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)

    def forward(self, x):
        return x + self.out_conv3(self.conv3(x) * self.conv1(x))

class TSMambaLayer(nn.Module):
    def __init__(self, dim, num_slices=None, mlp_ratio=2):
        super().__init__()
        self.gsc = GSC(dim)
        self.tom = MambaLayer(dim=dim, num_slices=num_slices)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(dim, mlp_ratio * dim)

    def forward(self, x):
        x = self.gsc(x)
        x = self.tom(x)
        x = x + self.mlp(self.norm(x))
        return x

class FUE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        z_bar = torch.sigmoid(x.mean(dim=1, keepdim=True)).clamp(min=self.eps)
        uncertainty = -z_bar * torch.log(z_bar)
        return x + x * (1 - uncertainty)


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
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], input_size=None):
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
        for i in range(4):
            stage = nn.Sequential(
                *[TSMambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
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
        spatial_dims = spatial_dims if spatial_dims is not None else _setting_or_default(settings, "SEGMAMBA_SPATIAL_DIMS", 3)
        input_size = input_size if input_size is not None else _setting_or_default(settings, "INPUT_SIZE", [128, 128, 128])
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

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                input_size=input_size,
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.bottleneck_downsample = nn.Sequential(
            nn.InstanceNorm3d(self.feat_size[3]),
            nn.Conv3d(self.feat_size[3], self.hidden_size, kernel_size=2, stride=2),
        )
        self.encoder6 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.fue1 = FUE()
        self.fue2 = FUE()
        self.fue3 = FUE()
        self.fue4 = FUE()
        self.fue5 = FUE()
        self.fue6 = FUE()
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
        enc1 = self.fue1(self.encoder1(x_in))
        enc2 = self.fue2(self.encoder2(outs[0]))
        enc3 = self.fue3(self.encoder3(outs[1]))
        enc4 = self.fue4(self.encoder4(outs[2]))
        enc5 = self.fue5(self.encoder5(outs[3]))
        enc_hidden = self.fue6(self.encoder6(self.bottleneck_downsample(enc5)))
        dec3 = self.decoder5(enc_hidden, enc5)
        dec2 = self.decoder4(dec3, enc4)
        dec1 = self.decoder3(dec2, enc3)
        dec0 = self.decoder2(dec1, enc2)
        out = self.decoder1(dec0, enc1)
                
        return self.out(out)
    
