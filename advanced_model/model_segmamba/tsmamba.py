import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

from .utils import (
    LayerNorm,
    _make_activation,
    _norm3d,
    _morton_perm_3d,
    _feature_depths_from_input_size,
)


class GSC(nn.Module):
    def __init__(self, in_channles, mamba_impl="mamba1") -> None:
        super().__init__()
        self.mamba_impl = str(mamba_impl).lower()

        if self.mamba_impl == "mamba3":
            self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.norm = nn.InstanceNorm3d(in_channles)
            self.nonliner = nn.ReLU()

            self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.norm2 = nn.InstanceNorm3d(in_channles)
            self.nonliner2 = nn.ReLU()

            self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.norm3 = nn.InstanceNorm3d(in_channles)
            self.nonliner3 = nn.ReLU()

            self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.norm4 = nn.InstanceNorm3d(in_channles)
            self.nonliner4 = nn.ReLU()
        else:
            self.conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.conv1 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.out_conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)

    def forward(self, x):
        if self.mamba_impl == "mamba3":
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

            x = x1 + x2
            x = self.proj4(x)
            x = self.norm4(x)
            x = self.nonliner4(x)
            
            return x + x_residual
        else:
            return x + self.out_conv3(self.conv3(x) * self.conv1(x))



PROJECT_MAMBA_DIR = Path(__file__).resolve().parents[2] / "mamba"
LOCAL_MAMBA_DIR = Path(__file__).resolve().parents[1] / "mamba"
SEGMAMBA_MAMBA_DIR = Path(__file__).resolve().parents[2] / "SegMamba" / "mamba"

def import_mamba_from_dir(mamba_dir):
    import sys
    old_path = list(sys.path)
    old_mamba_modules = {}
    for k in list(sys.modules.keys()):
        if k.startswith("mamba_ssm"):
            old_mamba_modules[k] = sys.modules.pop(k)
    
    sys.path = [p for p in sys.path if "mamba" not in p.lower()]
    sys.path.insert(0, str(mamba_dir))
    
    try:
        from mamba_ssm import Mamba
        # Restore sys.path and sys.modules
        sys.path = old_path
        for k in list(sys.modules.keys()):
            if k.startswith("mamba_ssm"):
                sys.modules.pop(k, None)
        sys.modules.update(old_mamba_modules)
        return Mamba
    except Exception as e:
        sys.path = old_path
        sys.modules.update(old_mamba_modules)
        raise e

# Try to load the modified Mamba (with bimamba_type="v3")
LocalMamba = None
for candidate_dir in (LOCAL_MAMBA_DIR, SEGMAMBA_MAMBA_DIR, PROJECT_MAMBA_DIR):
    if candidate_dir.exists():
        try:
            LocalMamba = import_mamba_from_dir(candidate_dir)
            break
        except Exception:
            continue

# Now setup standard sys.path for Mamba2 and Mamba3 from PROJECT_MAMBA_DIR
for mamba_dir in (LOCAL_MAMBA_DIR, PROJECT_MAMBA_DIR):
    mamba_dir_str = str(mamba_dir)
    if mamba_dir.exists() and mamba_dir_str in sys.path:
        sys.path.remove(mamba_dir_str)
for mamba_dir in (LOCAL_MAMBA_DIR, PROJECT_MAMBA_DIR):
    if mamba_dir.exists():
        sys.path.insert(0, str(mamba_dir))

try:
    from mamba_ssm import Mamba2, Mamba3
except ImportError:
    for k in list(sys.modules.keys()):
        if k.startswith("mamba_ssm"):
            sys.modules.pop(k, None)
    if str(PROJECT_MAMBA_DIR) in sys.path:
        sys.path.remove(str(PROJECT_MAMBA_DIR))
    sys.path.insert(0, str(PROJECT_MAMBA_DIR))
    from mamba_ssm import Mamba2, Mamba3

SWINDER_DIR = Path(__file__).resolve().parents[2] / "Swin-DER"
if SWINDER_DIR.exists() and str(SWINDER_DIR) not in sys.path:
    sys.path.insert(0, str(SWINDER_DIR))

try:
    from SwinDER.upsample.onsampling import Onsampling
except ModuleNotFoundError:
    Onsampling = None


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = _make_activation("gelu")
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ParameterFreeIdentity(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x


class MambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        num_slices=None,
        mamba3_headdim=64,
        mamba3_chunk_size=64,
        mamba3_mimo_enabled=False,
        mamba3_mimo_rank=4,
        mamba3_rope_fraction=0.5,
        mamba3_outproj_norm_enabled=False,
        mamba_impl="mamba1",
        morton_z_enabled=False,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_impl = str(mamba_impl).lower()
        self.morton_z_enabled = bool(morton_z_enabled)
        self._morton_cache = {}
        if self.mamba_impl == "mamba3":
            self.mamba = Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=self._resolve_headdim(dim=dim, expand=expand, preferred_headdim=mamba3_headdim),
                chunk_size=self._resolve_chunk_size(mamba3_chunk_size, mamba3_mimo_enabled, mamba3_mimo_rank),
                rope_fraction=mamba3_rope_fraction,
                is_outproj_norm=mamba3_outproj_norm_enabled,
                is_mimo=mamba3_mimo_enabled,
                mimo_rank=mamba3_mimo_rank,
            )
        elif self.mamba_impl == "mamba2":
            self.mamba = Mamba2(
                d_model=dim,
                d_state=max(d_state, 16),
                expand=expand,
                headdim=self._resolve_headdim(dim=dim, expand=expand, preferred_headdim=mamba3_headdim),
                chunk_size=max(mamba3_chunk_size, 64),
                use_mem_eff_path=False,
            )
        elif self.mamba_impl in {"mamba", "mamba1", "simple"}:
            mamba_cls = LocalMamba
            if mamba_cls is None:
                try:
                    from mamba_ssm import Mamba as EnvMamba
                    mamba_cls = EnvMamba
                except ImportError:
                    raise ImportError("Could not import Mamba from any local path or environment.")
            import inspect
            sig = inspect.signature(mamba_cls.__init__)
            mamba_kwargs = {
                "d_model": dim,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            }
            if "bimamba_type" in sig.parameters:
                mamba_kwargs["bimamba_type"] = "none"
            if "nslices" in sig.parameters:
                mamba_kwargs["nslices"] = num_slices
            self.mamba = mamba_cls(**mamba_kwargs)
        else:
            raise ValueError(f"Unsupported mamba implementation: {mamba_impl}")

    @staticmethod
    def _resolve_headdim(dim, expand, preferred_headdim):
        d_inner = int(dim * expand)
        preferred_headdim = min(preferred_headdim, d_inner)
        for candidate in (preferred_headdim, 64, 32, 16, 8):
            if candidate <= d_inner and d_inner % candidate == 0 and candidate % 8 == 0:
                return candidate
        for candidate in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
            if candidate <= d_inner and d_inner % candidate == 0:
                return candidate
        gcd = math.gcd(d_inner, preferred_headdim)
        if gcd > 0:
            return gcd
        raise ValueError(f"Could not find a valid Mamba3 headdim for dim={dim}, expand={expand}")

    @staticmethod
    def _resolve_chunk_size(chunk_size, mimo_enabled, mimo_rank):
        chunk_size = max(1, int(chunk_size))
        if not mimo_enabled:
            return chunk_size
        return max(1, chunk_size // max(1, int(mimo_rank)))

    def _get_morton_perm(self, spatial_shape, device):
        key = tuple(spatial_shape)
        if key not in self._morton_cache:
            self._morton_cache[key] = _morton_perm_3d(*spatial_shape, device=torch.device("cpu"))
        perm, inv = self._morton_cache[key]
        if perm.device != device:
            perm = perm.to(device=device)
            inv = inv.to(device=device)
        return perm, inv

    def _mamba_forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        if self.morton_z_enabled:
            perm, inv = self._get_morton_perm(img_dims, x.device)
            x_flat = x_flat[:, perm, :].contiguous()

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        if self.morton_z_enabled:
            x_mamba = x_mamba[:, inv, :].contiguous()

        return x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

    def forward(self, x):
        x_skip = x
        if self.mamba_impl in {"mamba", "mamba1", "simple"}:
            from einops import rearrange
            out_x_1 = self._mamba_forward(x)

            x_2 = rearrange(x, "b c d w h -> b c w d h")
            out_x_2 = self._mamba_forward(x_2)
            out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")

            x_3 = rearrange(x, "b c d w h -> b c h w d")
            out_x_3 = self._mamba_forward(x_3)
            out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")

            out = out_x_1 + out_x_2 + out_x_3
            out = out + x_skip
            return out
        else:
            out = self._mamba_forward(x)
            out = out + x_skip
            return out


class TSMambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_slices=None,
        mlp_ratio=2,
        mamba3_d_state=16,
        mamba3_headdim=64,
        mamba3_chunk_size=64,
        mamba3_mimo_enabled=False,
        mamba3_mimo_rank=4,
        mamba3_rope_fraction=0.5,
        mamba3_outproj_norm_enabled=False,
        mamba_impl="mamba1",
        morton_z_enabled=False,
    ):
        super().__init__()
        self.gsc = GSC(dim, mamba_impl=mamba_impl)
        self.tom = MambaLayer(
            dim=dim,
            num_slices=num_slices,
            d_state=mamba3_d_state,
            mamba3_headdim=mamba3_headdim,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_mimo_enabled=mamba3_mimo_enabled,
            mamba3_mimo_rank=mamba3_mimo_rank,
            mamba3_rope_fraction=mamba3_rope_fraction,
            mamba3_outproj_norm_enabled=mamba3_outproj_norm_enabled,
            mamba_impl=mamba_impl,
            morton_z_enabled=morton_z_enabled,
        )
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(dim, mlp_ratio * dim)

    def forward(self, x):
        x = self.gsc(x)
        x = self.tom(x)
        x = x + self.mlp(self.norm(x))
        return x









class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        input_size=None,
        mamba_stages=None,
        mamba3_d_state=16,
        mamba3_headdim=64,
        mamba3_chunk_size=64,
        mamba3_mimo_enabled=False,
        mamba3_mimo_rank=4,
        mamba3_rope_fraction=0.5,
        mamba3_outproj_norm_enabled=False,
        mamba_impl="mamba1",
        morton_z_enabled=False,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, groups=in_chans),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
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
            block_cls = TSMambaLayer if i in mamba_stage_set else ParameterFreeIdentity
            stage = nn.Sequential(
                *[
                    block_cls(
                        dim=dims[i],
                        num_slices=num_slices_list[i],
                        mamba3_d_state=mamba3_d_state,
                        mamba3_headdim=mamba3_headdim,
                        mamba3_chunk_size=mamba3_chunk_size,
                        mamba3_mimo_enabled=mamba3_mimo_enabled,
                        mamba3_mimo_rank=mamba3_mimo_rank,
                        mamba3_rope_fraction=mamba3_rope_fraction,
                        mamba3_outproj_norm_enabled=mamba3_outproj_norm_enabled,
                        mamba_impl=mamba_impl,
                        morton_z_enabled=morton_z_enabled,
                    )
                    if block_cls is TSMambaLayer
                    else block_cls(dim=dims[i])
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
