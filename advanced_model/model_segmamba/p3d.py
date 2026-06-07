import sys
from pathlib import Path
import torch
import torch.nn as nn

ADVANCED_MODEL_DIR = Path(__file__).resolve().parents[1]
if str(ADVANCED_MODEL_DIR) in sys.path:
    sys.path.remove(str(ADVANCED_MODEL_DIR))
    sys.path.append(str(ADVANCED_MODEL_DIR))

from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from .utils import _make_activation, _norm3d, _load_project_settings, _setting_or_default

SWINDER_DIR = Path(__file__).resolve().parents[2] / "Swin-DER"
if SWINDER_DIR.exists() and str(SWINDER_DIR) not in sys.path:
    sys.path.insert(0, str(SWINDER_DIR))

try:
    from SwinDER.upsample.onsampling import Onsampling
except ModuleNotFoundError:
    Onsampling = None


class Pseudo3DBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
        bottleneck_ratio=4,
        res_block=True,
        morton_z_enabled=False,
    ):
        super().__init__()
        hidden_channels = max(out_channels // bottleneck_ratio, 8)
        self.res_block = res_block
        activation = _make_activation("gelu")
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
            _make_activation("gelu"),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu"),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _norm3d(out_channels, norm_name),
        )
        self.act = _make_activation("gelu")

    def forward(self, x):
        out = self.conv(x)
        if self.res_block:
            out = out + self.proj(x)
        return self.act(out)


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
        )

    def forward(self, inp, skip):
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class PWDWConv3D(nn.Module):
    def __init__(self, channels, norm_name="batch"):
        super().__init__()
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.pw_norm = _norm3d(channels, norm_name)
        self.pw_act = nn.ReLU(inplace=True)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True, groups=channels)
        self.dw_norm = _norm3d(channels, norm_name)
        self.dw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.pw_norm(x)
        x = self.pw_act(x)
        x = self.dwconv(x)
        x = self.dw_norm(x)
        return self.dw_act(x)


def _make_decoder_block(
    use_unet_3d_conv,
    upsample_mode,
    spatial_dims,
    in_channels,
    out_channels,
    norm_name,
    res_block,
):

    if use_unet_3d_conv:
        return UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
    return Pseudo3DUpBlock(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=res_block,
        upsample_mode=upsample_mode,
    )


def _make_encoder_block(
    use_unet_3d_conv,
    spatial_dims,
    in_channels,
    out_channels,
    norm_name,
    res_block,
):

    if use_unet_3d_conv:
        return UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
    return Pseudo3DBottleneckBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        norm_name=norm_name,
        res_block=res_block,
    )
