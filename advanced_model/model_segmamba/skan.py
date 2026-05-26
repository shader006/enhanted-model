import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    KANLinear,
    SKANLinear,
    _make_activation,
    _norm3d,
    _valid_group_count,
    _morton_perm_3d,
)
from .p3d import Pseudo3DBottleneckBlock, PWDWConv3D, Onsampling


def _make_group_kan_linear(linear_cls, in_features, out_features, grid_size=5, spline_order=3):
    if linear_cls is KANLinear:
        return linear_cls(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    return linear_cls(in_features, out_features)


def _make_groupkan_spatial_mixer(
    channels,
    norm_name="instance",
    bottleneck_ratio=4,
    spatial_mixer="pseudo3d",
):
    if str(spatial_mixer).lower() == "pwdw":
        return PWDWConv3D(channels, norm_name=norm_name)
    return Pseudo3DBottleneckBlock(
        channels,
        channels,
        norm_name=norm_name,
        bottleneck_ratio=bottleneck_ratio,
        res_block=False,
        use_starrelu=False,
    )


class TokenKANPseudo3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
        bottleneck_ratio=4,
        res_block=True,
        use_starrelu=False,
        morton_z_enabled=False,
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
        self.morton_z_enabled = bool(morton_z_enabled)
        self._morton_cache = {}

    def _volume_to_tokens(self, x):
        b, c, d, h, w = x.shape
        tokens = x.reshape(b, c, d * h * w).transpose(1, 2).contiguous()
        if not self.morton_z_enabled:
            return tokens, {"mode": "global", "spatial_shape": (d, h, w)}

        perm, inv = self._get_morton_perm((d, h, w), x.device)
        tokens = tokens[:, perm, :].contiguous()
        return tokens, {
            "mode": "morton_z",
            "spatial_shape": (d, h, w),
            "inverse_perm": inv,
        }

    def _get_morton_perm(self, spatial_shape, device):
        key = tuple(spatial_shape)
        if key not in self._morton_cache:
            self._morton_cache[key] = _morton_perm_3d(*spatial_shape, device=torch.device("cpu"))
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
    def _morton_tokens_to_volume(tokens, token_meta):
        inverse_perm = token_meta["inverse_perm"]
        spatial_shape = token_meta["spatial_shape"]
        tokens = tokens[:, inverse_perm, :].contiguous()
        return TokenKANPseudo3DBlock._global_tokens_to_volume(tokens, spatial_shape)

    @staticmethod
    def _tokens_to_volume(tokens, token_meta):
        if token_meta["mode"] == "global":
            return TokenKANPseudo3DBlock._global_tokens_to_volume(tokens, token_meta["spatial_shape"])
        if token_meta["mode"] == "morton_z":
            return TokenKANPseudo3DBlock._morton_tokens_to_volume(tokens, token_meta)
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


class GroupedKANActivation(nn.Module):
    def __init__(self, channels, group=16, linear_cls=None, linear_name="KANLinear"):
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for GroupedKANActivation.")

        self.channels = int(channels)
        self.group = _valid_group_count(self.channels, requested=group)
        self.channels_per_group = self.channels // self.group
        self.vectorized_lss = linear_cls is SKANLinear
        if self.vectorized_lss:
            self.weight = nn.Parameter(torch.empty(self.group, 2))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.weight[:, -1].zero_()
            self.register_buffer("log2", torch.tensor(math.log(2.0)), persistent=False)
        else:
            self.group_kan = nn.ModuleList([linear_cls(1, 1) for _ in range(self.group)])

    def forward(self, tokens):
        b, n, c = tokens.shape
        if self.vectorized_lss:
            grouped = tokens.view(b, n, self.group, self.channels_per_group)
            scale = self.weight[:, 0].view(1, 1, self.group, 1)
            bias = self.weight[:, 1].view(1, 1, self.group, 1)
            grouped = F.softplus(grouped * scale) - self.log2
            grouped = grouped + F.softplus(bias) - self.log2
            return grouped.reshape(b, n, c)

        group_outputs = []
        for group_idx, kan in enumerate(self.group_kan):
            start = group_idx * self.channels_per_group
            end = start + self.channels_per_group
            group_tokens = tokens[:, :, start:end].reshape(-1, 1)
            group_tokens = kan(group_tokens).view(b, n, self.channels_per_group)
            group_outputs.append(group_tokens)
        return torch.cat(group_outputs, dim=2)


class GroupedKANTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group=16,
        grid_size=5,
        spline_order=3,
        linear_cls=None,
        linear_name="KANLinear",
    ):
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for GroupedKANTransform.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.group = _valid_group_count(self.in_channels, self.out_channels, requested=group)
        self.in_per_group = self.in_channels // self.group
        self.out_per_group = self.out_channels // self.group
        self.vectorized_lss = linear_cls is SKANLinear
        if self.vectorized_lss:
            self.weight = nn.Parameter(torch.empty(self.group, self.out_per_group, self.in_per_group + 1))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.weight[:, :, -1].zero_()
            self.register_buffer("log2", torch.tensor(math.log(2.0)), persistent=False)
        else:
            self.group_kan = nn.ModuleList(
                [
                    _make_group_kan_linear(
                        linear_cls,
                        self.in_per_group,
                        self.out_per_group,
                        grid_size=grid_size,
                        spline_order=spline_order,
                    )
                    for _ in range(self.group)
                ]
            )

    def forward(self, tokens):
        b, n, _ = tokens.shape
        if self.vectorized_lss:
            grouped = tokens.view(b, n, self.group, self.in_per_group)
            input_weight = self.weight[:, :, :-1].view(1, 1, self.group, self.out_per_group, self.in_per_group)
            bias_weight = self.weight[:, :, -1].view(1, 1, self.group, self.out_per_group)
            grouped = grouped.unsqueeze(3)
            grouped = (F.softplus(grouped * input_weight) - self.log2).sum(dim=-1)
            grouped = grouped + F.softplus(bias_weight) - self.log2
            return grouped.reshape(b, n, self.out_channels)

        group_outputs = []
        for group_idx, kan in enumerate(self.group_kan):
            start = group_idx * self.in_per_group
            end = start + self.in_per_group
            group_tokens = tokens[:, :, start:end].reshape(b * n, self.in_per_group)
            group_tokens = kan(group_tokens).view(b, n, self.out_per_group)
            group_outputs.append(group_tokens)
        return torch.cat(group_outputs, dim=2)


class TokenGroupKANPseudo3DBlock(TokenKANPseudo3DBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
        bottleneck_ratio=4,
        res_block=True,
        use_starrelu=False,
        morton_z_enabled=False,
        active_group=16,
        channel_group=16,
        spatial_mixer="pseudo3d",
        linear_cls=None,
        linear_name="KANLinear",
    ):
        nn.Module.__init__(self)
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for TokenGroupKANPseudo3DBlock.")

        self.res_block = res_block
        self.linear_cls = linear_cls
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.token_norm1 = nn.LayerNorm(in_channels)
        self.gka1 = GroupedKANActivation(
            in_channels,
            group=active_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.fc1 = GroupedKANTransform(
            in_channels,
            out_channels,
            group=channel_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.spatial_mixer1 = _make_groupkan_spatial_mixer(
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            spatial_mixer=spatial_mixer,
        )
        self.token_norm2 = nn.LayerNorm(out_channels)
        self.gka2 = GroupedKANActivation(
            out_channels,
            group=active_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.fc2 = GroupedKANTransform(
            out_channels,
            out_channels,
            group=channel_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.spatial_mixer2 = _make_groupkan_spatial_mixer(
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            spatial_mixer=spatial_mixer,
        )
        self.out_norm = _norm3d(out_channels, norm_name)
        self.act = _make_activation("gelu", use_starrelu=False)
        self.morton_z_enabled = bool(morton_z_enabled)
        self._morton_cache = {}

    def forward(self, x):
        residual = self.proj(x)

        tokens, token_meta = self._volume_to_tokens(x)
        tokens = self.token_norm1(tokens)
        tokens = self.gka1(tokens)
        tokens = self.fc1(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.spatial_mixer1(out)

        tokens, token_meta = self._volume_to_tokens(out)
        tokens = self.token_norm2(tokens)
        tokens = self.gka2(tokens)
        tokens = self.fc2(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.spatial_mixer2(out)

        out = self.out_norm(out)
        if self.res_block:
            out = out + residual
        return self.act(out)


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
        morton_z_enabled=False,
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
            morton_z_enabled=morton_z_enabled,
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
        morton_z_enabled=False,
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
            morton_z_enabled=morton_z_enabled,
        )


class TokenGroupKANPseudo3DUpBlock(TokenKANPseudo3DUpBlock):
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
        morton_z_enabled=False,
        active_group=16,
        channel_group=16,
        spatial_mixer="pseudo3d",
        linear_cls=None,
        linear_name="KANLinear",
    ):
        nn.Module.__init__(self)
        if spatial_dims != 3:
            raise ValueError("TokenGroupKANPseudo3DUpBlock only supports spatial_dims=3.")
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
        self.conv_block = TokenGroupKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            use_starrelu=use_starrelu,
            morton_z_enabled=morton_z_enabled,
            active_group=active_group,
            channel_group=channel_group,
            spatial_mixer=spatial_mixer,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
