__version__ = "1.0.1"

try:
    import triton.language as tl
    if not hasattr(tl, "make_tensor_descriptor") and hasattr(tl, "_experimental_make_tensor_descriptor"):
        tl.make_tensor_descriptor = tl._experimental_make_tensor_descriptor
except ImportError:
    pass

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
