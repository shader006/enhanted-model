import torch

from .sgd import SGD
from .msam import MSAM
from .adamW import AdamW
from .adamW_msam import AdamW_MSAM
try:
    from .sam import SAM
except ModuleNotFoundError:
    SAM = None

try:
    from .esam import ESAM
except ModuleNotFoundError:
    ESAM = None

try:
    from .looksam import LookSAM
except ModuleNotFoundError:
    LookSAM = None

try:
    from .adamW_sam import AdamW_SAM
except ModuleNotFoundError:
    AdamW_SAM = None

try:
    from utility.args import Args
except ModuleNotFoundError:
    Args = None

if Args is not None:
    Args.add_argument("--optimizer", type=str, help="optimizer name")
    Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
    Args.add_argument("--momentum", type=float, help="Momentum.")
    Args.add_argument("--rho", type=float, help="")
    Args.add_argument("--nesterov", type=bool, help="use normal nesterov momentum for sgd/sam")

    Args.add_argument("--grad_clip_norm", type=float, help="")


def getOptimizer(params) -> torch.nn.Module:
    if Args is None:
        raise RuntimeError("MSAM utility Args is unavailable in library-import mode.")

    # optimizer names and additional args which will be passed from Args.XX
    optimizerDict = {
        "SGD":        (SGD,        {"momentum": Args.momentum, "nesterov": Args.nesterov}),
        "MSAM":       (MSAM,       {"momentum": Args.momentum, "rho": Args.rho}),
        "AdamW":      (AdamW,      {}),
        "AdamW_MSAM": (AdamW_MSAM, {"rho": Args.rho}),
    }

    if SAM is not None:
        optimizerDict["SAM"] = (SAM, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho})
    if ESAM is not None:
        optimizerDict["ESAM"] = (ESAM, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho})
    if LookSAM is not None:
        optimizerDict["lookSAM"] = (LookSAM, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho})
    if AdamW_SAM is not None:
        optimizerDict["AdamW_SAM"] = (AdamW_SAM, {"rho": Args.rho})
    
    if Args.optimizer in optimizerDict:
        optimizer, additionalArgs = optimizerDict[Args.optimizer]
        return optimizer(params, lr = Args.learningRate, weight_decay=Args.weightDecay, **additionalArgs)
    else:
        raise RuntimeError(f"Optimizer '{Args.optimizer}' not found. Available optimizers: {', '.join(optimizerDict.keys())}")
