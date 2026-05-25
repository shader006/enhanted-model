import math
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

try:
    from MSAM.optimizer.msam import MSAM
except ModuleNotFoundError:
    MSAM = None

try:
    from lion_pytorch import Lion as ExternalLion
except ModuleNotFoundError:
    ExternalLion = None

try:
    from schedulefree import ScheduleFreeWrapper
except ModuleNotFoundError:
    ScheduleFreeWrapper = None


def _ensure_schedulefree_base_state(optimizer):
    """Initialize state entries expected by some optimizers before Schedule-Free adds its own keys."""
    optimizer_name = optimizer.__class__.__name__.lower()
    is_adam_like = optimizer_name in {"adam", "adamw"}
    is_lion_like = "lion" in optimizer_name

    if not (is_adam_like or is_lion_like):
        return

    for group in optimizer.param_groups:
        amsgrad = bool(group.get("amsgrad", False))
        for param in group["params"]:
            if param.grad is None:
                continue
            state = optimizer.state[param]
            if is_adam_like:
                state.setdefault("step", torch.zeros((), dtype=torch.float32, device=param.device))
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                if amsgrad and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            elif is_lion_like and "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)


class CompatibleScheduleFreeWrapper:
    """Compatibility wrapper around schedulefree that preserves Adam/AdamW/Lion state initialization."""

    def __init__(
        self,
        base,
        weight_decay_at_y=0.0,
        momentum=0.9,
        weight_lr_power=2,
        r=0,
    ):
        self.base = base
        self.weight_decay_at_y = weight_decay_at_y
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.momentum = momentum
        self.train_mode = False

    def add_param_group(self, param_group):
        return self.base.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.base.load_state_dict(state_dict)

    def state_dict(self):
        return self.base.state_dict()

    def zero_grad(self, set_to_none=True):
        return self.base.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1 - 1 / self.momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1 - self.momentum)
        self.train_mode = True

    @staticmethod
    def swap(x, y):
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        _ensure_schedulefree_base_state(self.base)

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)

                z = state["z"]

                if self.weight_decay_at_y != 0.0:
                    z.sub_(p, alpha=lr * self.weight_decay_at_y)
                    p.sub_(p, alpha=lr * self.weight_decay_at_y * (1 - self.momentum))

                p.lerp_(end=z, weight=1 - 1 / self.momentum)
                self.swap(z, p)

        self.base.step()

        for group in self.param_groups:
            weight_lr_power = self.weight_lr_power
            r = self.r
            k = group.get("k", 0)
            d = group.get("d", 1.0)
            lr = group["lr"] * d
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0))

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / weight_sum

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                z = state["z"]
                self.swap(z, p)
                p.lerp_(end=z, weight=ckp1)
                p.lerp_(end=state["z"], weight=1 - self.momentum)

            group["k"] = k + 1

        return loss

# Training settings
EPOCHS = 300
BATCH_SIZE = 1
INPUT_SIZE = [96, 96, 96]  # spatial input size for the model; adjust as needed for memory constraints and model architecture
VAL_EVERY = 1
NUM_GPUS = 1
DEVICE = "cuda:0"
ENV_TYPE = "pytorch"
AUGMENTATION = True
SEGMAMBA_AMP_ENABLED = False
SEGMAMBA_AMP_PRECISION = "fp32"  # choices: bf16, fp16, fp32
SEGMAMBA_MODALITY_DROPOUT_ENABLED = True
SEGMAMBA_MODALITY_DROPOUT_PROB = 0.3
SEGMAMBA_MODALITY_DROPOUT_MAX_CHANNELS = 1

# SegMamba model settings
SEGMAMBA_IN_CHANS = 4
SEGMAMBA_OUT_CHANS = 4
SEGMAMBA_DEPTHS = [1, 1, 1, 1]  # number of Mamba blocks in each encoder stage
SEGMAMBA_FEAT_SIZE = [48, 96, 192, 384]
SEGMAMBA_HIDDEN_SIZE = 768
SEGMAMBA_DROP_PATH_RATE = 0
SEGMAMBA_LAYER_SCALE_INIT_VALUE = 1e-6
SEGMAMBA_NORM_NAME = "instance"
SEGMAMBA_DERF_NORM_ENABLED = False
SEGMAMBA_DERF_ALPHA_INIT_VALUE = 0.5
SEGMAMBA_DERF_SHIFT_INIT_VALUE = 0.0
SEGMAMBA_3D_CONV = False
SEGMAMBA_RES_BLOCK = True
SEGMAMBA_KAN = False
SEGMAMBA_SKAN = True
SEGMAMBA_GROUPKAN = False
SEGMAMBA_GROUPKAN_ACTIVE_GROUP = 16
SEGMAMBA_GROUPKAN_CHANNEL_GROUP = 16
SEGMAMBA_GROUPKAN_SPATIAL_MIXER = "pseudo3d"  # choices: pseudo3d, pwdw
SEGMAMBA_KAN_MORTON_Z = True
SEGMAMBA_SPATIAL_DIMS = 3
SEGMAMBA_MAMBA_STAGES = [0,1, 2]
SEGMAMBA_STARRELU = False
SEGMAMBA_ONSAMPLING = True
ADVANCED_SEGMAMBA_MAMBA_IMPL = "mamba3"  # choices: mamba1, mamba2, mamba3
ADVANCED_SEGMAMBA_MAMBA3_MIMO_ENABLED = False

ADVANCED_SEGMAMBA_MAMBA3_MIMO_RANK = 4
ADVANCED_SEGMAMBA_MAMBA3_ROPE_FRACTION = 0.5 # choices: 0.5, 1.0
ADVANCED_SEGMAMBA_MAMBA3_D_STATE = 64
ADVANCED_SEGMAMBA_MAMBA3_HEADDIM = 64
ADVANCED_SEGMAMBA_MAMBA3_CHUNK_SIZE = 64
SEGMAMBA_VSS_DECODER_ENABLED = False

# Swin-DER model settings
SWINDER_ONSAMPLING = False
SWINDER_UPSAMPLE = "transconv"

# Reproducibility settings (shared across projects)
REPRO_SEED = 42
DETERMINISTIC = True
DETERMINISTIC_WARN_ONLY = True
SEGMAMBA_TF32_ENABLED = True
SEGMAMBA_AUGMENTER_BACKEND = "single-thread"
SEGMAMBA_TRAIN_PROCESS = 1
SEGMAMBA_VAL_PROCESS = 1
SEGMAMBA_AUGMENTER_SEED_COUNT = 32
SEGMAMBA_RAM_CACHE_ENABLED = True
SEGMAMBA_RAM_CACHE_PRELOAD = False
SEGMAMBA_GPU_TRANSFORMS_ENABLED = True
SEGMAMBA_CHANNELS_LAST_3D_ENABLED = True

# Optimizer settings
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULE_FREE_ENABLED = False
SCHEDULE_FREE_MOMENTUM = 0.9
SCHEDULE_FREE_WEIGHT_DECAY_AT_Y = 0.0
SEGMAMBA_LOSS_NAME = "DiceFocalTversky"  # choices: DiceCE, CE, DiceFocalTversky
SEGMAMBA_DICE_LOSS_WEIGHT = 1.0
SEGMAMBA_CE_LOSS_WEIGHT = 1.0
SEGMAMBA_FOCAL_TVERSKY_LOSS_WEIGHT = 1.0
SEGMAMBA_FOCAL_TVERSKY_ALPHA = 0.3
SEGMAMBA_FOCAL_TVERSKY_BETA = 0.7
SEGMAMBA_FOCAL_TVERSKY_GAMMA = 4.0 / 3.0
SEGMAMBA_LOSS_INCLUDE_BACKGROUND = False
SGD_MOMENTUM = 0.99
SGD_NESTEROV = True
LION_BETAS = (0.9, 0.99)
MSAM_MOMENTUM = 0.9
MSAM_NESTEROV = False
MSAM_RHO = 0.05

# Scheduler settings
SCHEDULER_NAME = "cosine_with_warmup"
T_MAX = 500
ETA_MIN = 1e-6
SCHEDULER_WARMUP = 0.05

# W&B settings
# Use "offline" on HPC nodes without internet. Set to "online" when network is available.
WANDB_MODE = "offline"
WANDB_PROJECT = "BTS_VAE_Model"
WANDB_RUN_NAME = None
WANDB_INIT_TIMEOUT = 30

# SegMamba logging settings
SEGMAMBA_WANDB_ENABLED = True
SEGMAMBA_WANDB_PROJECT = "SegMamba-BraTS23"
SEGMAMBA_WANDB_RUN_NAME = None
SEGMAMBA_LOG_EVERY_N_STEPS = 10
SEGMAMBA_BEST_MIN_DICE_DELTA = 1e-4
SEGMAMBA_BEST_MIN_HD95_DELTA = 1e-3


def build_optimizer(model):
    """Build optimizer for project training."""
    optimizer = None
    if OPTIMIZER_NAME == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER_NAME == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )
    elif OPTIMIZER_NAME == "Lion":
        lion_cls = getattr(torch.optim, "Lion", None) or ExternalLion
        if lion_cls is None:
            raise ModuleNotFoundError(
                "Lion optimizer is unavailable. Install lion-pytorch or use a PyTorch build that provides torch.optim.Lion."
            )
        optimizer = lion_cls(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=LION_BETAS,
        )
    elif OPTIMIZER_NAME == "MSAM":
        if MSAM is None:
            raise ModuleNotFoundError("MSAM optimizer is not importable from the current Python path.")
        optimizer = MSAM(
            params=model.parameters(),
            lr=LEARNING_RATE,
            momentum=MSAM_MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=MSAM_NESTEROV,
            rho=MSAM_RHO,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")

    if SCHEDULE_FREE_ENABLED:
        if ScheduleFreeWrapper is None:
            raise ModuleNotFoundError(
                "Schedule-Free is enabled but the 'schedulefree' package is unavailable. Install it with: pip install schedulefree"
            )
        optimizer = CompatibleScheduleFreeWrapper(
            optimizer,
            momentum=SCHEDULE_FREE_MOMENTUM,
            weight_decay_at_y=SCHEDULE_FREE_WEIGHT_DECAY_AT_Y,
        )

    return optimizer


def set_optimizer_train_mode(optimizer):
    """Switch optimizer to train mode when supported by the optimizer implementation."""
    if hasattr(optimizer, "train") and callable(optimizer.train):
        optimizer.train()


def set_optimizer_eval_mode(optimizer):
    """Switch optimizer to eval mode when supported by the optimizer implementation."""
    if hasattr(optimizer, "eval") and callable(optimizer.eval):
        optimizer.eval()


def _build_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _build_constant_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _build_poly_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1):
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init
        lr_range = lr_init - lr_end
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_lr, max_steps, exponent=0.9, current_step=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        progress = min(float(current_step) / float(max(1, self.max_steps)), 1.0)
        new_lr = self.initial_lr * (1 - progress) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


def build_scheduler(optimizer, num_training_steps=None, num_epochs=None):
    """Build LR scheduler paired with optimizer from project settings."""
    if SCHEDULE_FREE_ENABLED:
        return None
    if SCHEDULER_NAME in (None, "none", "None"):
        return None
    if SCHEDULER_NAME == "cosine_with_warmup":
        if num_training_steps is None:
            raise ValueError("SCHEDULER_NAME='cosine_with_warmup' requires num_training_steps.")
        warmup_steps = int(num_training_steps * SCHEDULER_WARMUP)
        return _build_cosine_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if SCHEDULER_NAME == "constant_with_warmup":
        if num_training_steps is None:
            raise ValueError("SCHEDULER_NAME='constant_with_warmup' requires num_training_steps.")
        warmup_steps = int(num_training_steps * SCHEDULER_WARMUP)
        return _build_constant_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    if SCHEDULER_NAME == "poly_with_warmup":
        if num_training_steps is None:
            raise ValueError("SCHEDULER_NAME='poly_with_warmup' requires num_training_steps.")
        warmup_steps = int(num_training_steps * SCHEDULER_WARMUP)
        return _build_poly_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if SCHEDULER_NAME == "poly":
        if num_training_steps is None:
            raise ValueError("SCHEDULER_NAME='poly' requires num_training_steps.")
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        return PolyLRScheduler(optimizer, initial_lr=lr, max_steps=num_training_steps)
    if SCHEDULER_NAME == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_MAX,
            eta_min=ETA_MIN,
        )
    raise ValueError(f"Unsupported scheduler: {SCHEDULER_NAME}")


def set_global_reproducibility(
    seed=REPRO_SEED,
    deterministic=DETERMINISTIC,
    deterministic_warn_only=DETERMINISTIC_WARN_ONLY,
    tf32_enabled=SEGMAMBA_TF32_ENABLED,
):
    """Set global random seeds and deterministic backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled

    # PyTorch deterministic kernels when available.
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        except TypeError:
            # Backward compatibility for older PyTorch without warn_only.
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def build_torch_generator(device="cpu", seed=REPRO_SEED):
    """Build a seeded torch.Generator for DataLoader shuffling and sampling."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def seed_worker(worker_id):
    """Seed each DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_swinder_upsample():
    """Return the effective Swin-DER decoder upsampling mode from settings."""
    return "onsampling" if SWINDER_ONSAMPLING else SWINDER_UPSAMPLE
