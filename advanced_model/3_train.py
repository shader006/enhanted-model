import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import numpy as np
import torch
import torch.nn as nn
import json
import sys
import re
import importlib.util
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRATS23_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != BASE_DIR
]
if BRATS23_DIR not in sys.path:
    sys.path.insert(0, BRATS23_DIR)
sys.path.append(BASE_DIR)

from light_training.dataloading.dataset import get_train_val_test_loader_from_split_json
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses import DiceCELoss

def _load_project_settings():
    settings_path = os.path.join(BRATS23_DIR, "settings.py")
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load settings from {settings_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = module
    spec.loader.exec_module(module)
    return module


settings = _load_project_settings()

settings.set_global_reproducibility()
set_determinism(settings.REPRO_SEED)

def parse_resume_checkpoint(argv):
    positional_args = [arg for arg in argv[1:] if not arg.startswith("--")]
    if not positional_args:
        return None
    if len(positional_args) > 1:
        raise ValueError(
            "Usage: python 3_train.py [path/to/model.pt]. "
            f"Received extra positional arguments: {positional_args}"
        )
    resume_path = os.path.abspath(positional_args[0])
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    if not resume_path.endswith(".pt"):
        raise ValueError(f"Resume checkpoint must be a .pt file: {resume_path}")
    return resume_path


def infer_run_name_from_checkpoint(checkpoint_path):
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    for i in range(len(parts) - 2):
        if parts[i] == "Log" and parts[i + 1] == "SegMamba":
            return parts[i + 2]
    return None


def infer_start_epoch_from_checkpoint(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"(?:^|_)ep(\d+)(?:_|\.|$)", filename)
    if match:
        return int(match.group(1)) + 1
    return 0


resume_checkpoint_path = parse_resume_checkpoint(sys.argv)
resume_start_epoch = infer_start_epoch_from_checkpoint(resume_checkpoint_path) if resume_checkpoint_path else 0
resume_run_name = infer_run_name_from_checkpoint(resume_checkpoint_path) if resume_checkpoint_path else None

_data_dir_candidates = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "fullres", "train")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "fullres", "train")),
]
data_dir = next((path for path in _data_dir_candidates if os.path.isdir(path)), _data_dir_candidates[0])
split_json_file = os.path.abspath(os.path.join(BASE_DIR, "..", "brats23_split_70_10_20.json"))
run_name = resume_run_name or settings.SEGMAMBA_WANDB_RUN_NAME or settings.WANDB_RUN_NAME or datetime.now().strftime("segmamba_%Y%m%d_%H%M%S")
run_root = os.path.join(BRATS23_DIR, "Log", "SegMamba", run_name)
logdir = os.path.join(run_root, "trainer")
wandb_dir = os.path.join(run_root, "wandb")
config_json_file = os.path.join(run_root, "config.json")
terminal_log_file = os.path.join(run_root, "terminal.log")

model_save_path = os.path.join(run_root, "checkpoints")
# augmentation = "nomirror"
augmentation = settings.AUGMENTATION
augmenter_backend = settings.SEGMAMBA_AUGMENTER_BACKEND  # options: single-thread | multi-thread | nondet-multiprocess

env = settings.ENV_TYPE
max_epoch = settings.EPOCHS
batch_size = settings.BATCH_SIZE
val_every = settings.VAL_EVERY
num_gpus = settings.NUM_GPUS
device = settings.DEVICE
roi_size = settings.INPUT_SIZE

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

def _to_loggable_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_loggable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_loggable_value(v) for k, v in value.items()}
    return str(value)

def build_config():
    settings_config = {
        name: _to_loggable_value(getattr(settings, name))
        for name in dir(settings)
        if name.isupper() and isinstance(getattr(settings, name), (str, int, float, bool, list, tuple, dict, type(None)))
    }
    return {
        "run_name": run_name,
        "paths": {
            "data_dir": data_dir,
            "split_json_file": split_json_file,
            "run_root": run_root,
            "checkpoint_dir": model_save_path,
            "wandb_dir": wandb_dir,
        },
        "training": {
            "env": env,
            "max_epoch": max_epoch,
            "start_epoch": resume_start_epoch,
            "batch_size": batch_size,
            "val_every": val_every,
            "num_gpus": num_gpus,
            "device": device,
            "roi_size": roi_size,
            "augmentation": augmentation,
            "augmenter_backend": augmenter_backend,
        },
        "resume": {
            "enabled": resume_checkpoint_path is not None,
            "checkpoint_path": resume_checkpoint_path,
            "start_epoch": resume_start_epoch,
        },
        "settings": settings_config,
    }

def save_config_snapshot(config):
    os.makedirs(run_root, exist_ok=True)
    with open(config_json_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def setup_terminal_logging():
    os.makedirs(run_root, exist_ok=True)
    log_handle = open(terminal_log_file, "a", encoding="utf-8", buffering=1)
    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.__stdout__, log_handle)
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.__stderr__, log_handle)
    return log_handle


def teardown_terminal_logging(log_handle):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_handle.close()

def init_wandb(config):
    if not settings.SEGMAMBA_WANDB_ENABLED:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed; continuing with console logs and checkpoints only.")
        return None

    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = settings.WANDB_MODE
    os.environ["WANDB_DIR"] = wandb_dir
    run = wandb.init(
        project=settings.SEGMAMBA_WANDB_PROJECT or settings.WANDB_PROJECT,
        name=run_name,
        config=config,
        dir=wandb_dir,
        mode=settings.WANDB_MODE,
        settings=wandb.Settings(init_timeout=settings.WANDB_INIT_TIMEOUT),
    )
    run.save(terminal_log_file, policy="live")
    return run

def scalar_value(value):
    return value.detach().float().item() if torch.is_tensor(value) else float(value)

def get_amp_dtype(precision):
    precision = str(precision).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"fp32", "float32", "none", "off"}:
        return None
    raise ValueError(f"Unsupported SEGMAMBA_AMP_PRECISION: {precision}")


class DiceFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=4.0 / 3.0,
        smooth=1e-6,
        lambda_dice=1.0,
        lambda_focal_tversky=1.0,
        include_background=False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.lambda_dice = lambda_dice
        self.lambda_focal_tversky = lambda_focal_tversky
        self.include_background = include_background

    def forward(self, logits, target):
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()

        probs = torch.softmax(logits.float(), dim=1)
        target_onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)

        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            target_onehot = target_onehot[:, 1:]

        reduce_dims = tuple(range(2, probs.ndim))
        true_pos = (probs * target_onehot).sum(dim=reduce_dims)
        false_pos = (probs * (1.0 - target_onehot)).sum(dim=reduce_dims)
        false_neg = ((1.0 - probs) * target_onehot).sum(dim=reduce_dims)

        dice_score = (2.0 * true_pos + self.smooth) / (
            2.0 * true_pos + false_pos + false_neg + self.smooth
        )
        tversky_score = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()
        # Clamp the focal term to avoid NaNs from fractional powers of tiny negative
        # values introduced by floating-point error near tversky_score == 1.
        focal_term = (1.0 - tversky_score).clamp(min=self.smooth, max=1.0)
        focal_tversky_loss = torch.pow(focal_term, 1.0 / self.gamma).mean()
        return self.lambda_dice * dice_loss + self.lambda_focal_tversky * focal_tversky_loss

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        self.augmenter_backend = augmenter_backend
        from model_segmamba.segmamba import SegMamba

        self.model = SegMamba()

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.best_mean_hd95 = float("inf")
        self.best_min_dice_delta = settings.SEGMAMBA_BEST_MIN_DICE_DELTA
        self.best_min_hd95_delta = settings.SEGMAMBA_BEST_MIN_HD95_DELTA
        self.mse = nn.MSELoss()
        self.train_process = settings.SEGMAMBA_TRAIN_PROCESS
        self.val_process = settings.SEGMAMBA_VAL_PROCESS
        self.seed = settings.REPRO_SEED
        self.augmenter_seeds = [
            settings.REPRO_SEED + i
            for i in range(max(settings.SEGMAMBA_AUGMENTER_SEED_COUNT, self.train_process, self.val_process))
        ]
        self.optimizer = settings.build_optimizer(self.model)
        self.scheduler_builder = settings.build_scheduler

        self.scheduler_type = settings.SCHEDULER_NAME
        self.warmup = settings.SCHEDULER_WARMUP
        self.cross = nn.CrossEntropyLoss()
        self.loss_name = str(settings.SEGMAMBA_LOSS_NAME).lower()
        if self.loss_name == "dicece":
            self.loss_fn = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                lambda_dice=settings.SEGMAMBA_DICE_LOSS_WEIGHT,
                lambda_ce=settings.SEGMAMBA_CE_LOSS_WEIGHT,
            )
        elif self.loss_name in {"dicefocaltversky", "dice_focal_tversky", "dft"}:
            self.loss_fn = DiceFocalTverskyLoss(
                alpha=settings.SEGMAMBA_FOCAL_TVERSKY_ALPHA,
                beta=settings.SEGMAMBA_FOCAL_TVERSKY_BETA,
                gamma=settings.SEGMAMBA_FOCAL_TVERSKY_GAMMA,
                lambda_dice=settings.SEGMAMBA_DICE_LOSS_WEIGHT,
                lambda_focal_tversky=settings.SEGMAMBA_FOCAL_TVERSKY_LOSS_WEIGHT,
                include_background=settings.SEGMAMBA_LOSS_INCLUDE_BACKGROUND,
            )
        elif self.loss_name == "ce":
            self.loss_fn = self.cross
        else:
            raise ValueError(f"Unsupported SEGMAMBA_LOSS_NAME: {settings.SEGMAMBA_LOSS_NAME}")
        self.wandb_run = None
        self.enable_tensorboard = False
        self.log_every_n_steps = max(int(settings.SEGMAMBA_LOG_EVERY_N_STEPS), 1)
        amp_dtype = get_amp_dtype(settings.SEGMAMBA_AMP_PRECISION)
        self.amp_enabled = bool(settings.SEGMAMBA_AMP_ENABLED and amp_dtype is not None and "cuda" in self.device)
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.float32
        if self.amp_enabled and self.amp_dtype == torch.bfloat16:
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                raise RuntimeError("SEGMAMBA_AMP_PRECISION='bf16' but this CUDA device does not support BF16.")
            self.grad_scaler = None
        elif self.amp_enabled and self.amp_dtype == torch.float16:
            self.grad_scaler = torch.amp.GradScaler("cuda")
        else:
            self.grad_scaler = None
        print(f"AMP enabled: {self.amp_enabled}, precision: {settings.SEGMAMBA_AMP_PRECISION}, grad_scaler: {self.grad_scaler is not None}")
        print(f"Loss: {settings.SEGMAMBA_LOSS_NAME}")

    def set_wandb_run(self, wandb_run):
        self.wandb_run = wandb_run

    def log(self, k, v, step):
        if self.local_rank != 0:
            return
        if self.wandb_run is not None:
            if k in {"training_loss", "lr"} and int(step) % self.log_every_n_steps != 0:
                return
            self.wandb_run.log({
                k: scalar_value(v),
                "trainer_step": int(step),
                "epoch": int(getattr(self, "epoch", -1)),
            })

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)

        if self.loss_name in {"dicece", "dicefocaltversky", "dice_focal_tversky", "dft"}:
            loss = self.loss_fn(pred, label.unsqueeze(1))
        else:
            loss = self.loss_fn(pred, label)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {self.global_step}: {loss.item()}. "
                f"AMP={self.amp_enabled}, precision={settings.SEGMAMBA_AMP_PRECISION}, loss={settings.SEGMAMBA_LOSS_NAME}"
            )

        self.log("training_loss", loss, step=self.global_step)

        return loss

    def convert_labels(self, labels):
        ## TC, WT and ET
        is_label_1 = labels == 1
        is_label_2 = labels == 2
        is_label_3 = labels == 3
        tc = is_label_1 | is_label_3
        wt = tc | is_label_2
        return torch.cat((tc, wt, is_label_3), dim=1).float()


    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()
        label.clamp_(0, 3)
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=(1.0, 1.0, 1.0)):
        gt = np.asarray(gt).astype(bool)
        pred = np.asarray(pred).astype(bool)

        spatial_rank = len(voxel_spacing)
        if gt.ndim == spatial_rank + 1:
            sample_metrics = [
                self.cal_metric(gt_i, pred_i, voxel_spacing=voxel_spacing)
                for gt_i, pred_i in zip(gt, pred)
            ]
            return np.nanmean(np.stack(sample_metrics, axis=0), axis=0)

        if gt.ndim != spatial_rank:
            gt = np.squeeze(gt)
            pred = np.squeeze(pred)

        if gt.ndim != spatial_rank:
            raise RuntimeError(
                f"HD95 expects {spatial_rank}D masks, got gt shape {gt.shape}, pred shape {pred.shape}."
            )

        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            hd95 = hausdorff_distance_95(pred, gt, voxel_spacing=voxel_spacing)
            return np.array([d, hd95])

        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 0.0])

        else:
            return np.array([0.0, 50.0])

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            class_metric = self.cal_metric(target_c, pred_c)
            dices.append(class_metric)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs

        tc = np.nanmean(dices[0][:, 0])
        wt = np.nanmean(dices[1][:, 0])
        et = np.nanmean(dices[2][:, 0])
        tc_hd95 = np.nanmean(dices[0][:, 1])
        wt_hd95 = np.nanmean(dices[1][:, 1])
        et_hd95 = np.nanmean(dices[2][:, 1])
        tc, wt, et = scalar_value(tc), scalar_value(wt), scalar_value(et)
        tc_hd95, wt_hd95, et_hd95 = scalar_value(tc_hd95), scalar_value(wt_hd95), scalar_value(et_hd95)

        print(f"dices is {tc, wt, et}")
        print(f"hd95 is {tc_hd95, wt_hd95, et_hd95}")

        mean_dice = scalar_value(np.nanmean([tc, wt, et]))
        mean_hd95 = scalar_value(np.nanmean([tc_hd95, wt_hd95, et_hd95]))
        if np.isnan(mean_hd95):
            mean_hd95 = float("inf")

        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)
        self.log("tc_hd95", tc_hd95, step=self.epoch)
        self.log("wt_hd95", wt_hd95, step=self.epoch)
        self.log("et_hd95", et_hd95, step=self.epoch)
        self.log("mean_hd95", mean_hd95, step=self.epoch)

        dice_improved = mean_dice > self.best_mean_dice + self.best_min_dice_delta
        dice_tied = abs(mean_dice - self.best_mean_dice) <= self.best_min_dice_delta
        hd95_improved = mean_hd95 < self.best_mean_hd95 - self.best_min_hd95_delta

        if dice_improved or (dice_tied and hd95_improved):
            best_reason = "dice" if dice_improved else "hd95_tiebreak"
            self.best_mean_dice = mean_dice
            self.best_mean_hd95 = mean_hd95
            save_new_model_and_delete_last(self.model,
                                            os.path.join(model_save_path,
                                            f"best_model_{best_reason}_dice{mean_dice:.4f}_hd95{mean_hd95:.4f}.pt"),
                                            delete_symbol="best_model")
            print(f"best model updated by {best_reason}")
        self.log("best_mean_dice", self.best_mean_dice, step=self.epoch)
        self.log("best_mean_hd95", self.best_mean_hd95, step=self.epoch)

        save_new_model_and_delete_last(self.model,
                                        os.path.join(model_save_path,
                                        f"final_model_dice{mean_dice:.4f}_hd95{mean_hd95:.4f}.pt"),
                                        delete_symbol="final_model")


        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_dice{mean_dice:.4f}_hd95{mean_hd95:.4f}.pt"))

        print(f"mean_dice is {mean_dice}, mean_hd95 is {mean_hd95}")

if __name__ == "__main__":
    experiment_config = build_config()
    terminal_log_handle = setup_terminal_logging()
    save_config_snapshot(experiment_config)

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)
    trainer.start_epoch = resume_start_epoch
    if resume_checkpoint_path:
        print(f"Resume checkpoint: {resume_checkpoint_path}")
        if resume_start_epoch > 0:
            print(f"Training will continue from epoch {resume_start_epoch} to {max_epoch - 1}.")
        else:
            print("Checkpoint filename has no epoch number; weights are loaded and training starts at epoch 0.")
        trainer.load_state_dict(resume_checkpoint_path, strict=True)
    trainer.set_wandb_run(init_wandb(experiment_config) if trainer.local_rank == 0 else None)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_split_json(data_dir, split_json_file)

    try:
        trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    finally:
        if trainer.wandb_run is not None:
            trainer.wandb_run.finish()
        teardown_terminal_logging(terminal_log_handle)
