import os
import sys
import torch
import numpy as np
import importlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRATS23_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)
sys.path.append(BRATS23_DIR)

print("Loading settings...")
import importlib.util
settings_path = os.path.join(BRATS23_DIR, "settings.py")
spec = importlib.util.spec_from_file_location("settings", settings_path)
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)
sys.modules["settings"] = settings

# Force enable GPU transforms for testing
settings.SEGMAMBA_GPU_TRANSFORMS_ENABLED = True
settings.SEGMAMBA_DERF_NORM_ENABLED = True
settings.ADVANCED_SEGMAMBA_MAMBA_IMPL = "mamba1"

print("Importing BraTSTrainer from 3_train...")
# Use importlib to bypass module name starting with digit
train_module = importlib.import_module("advanced_model.3_train")
train_module.settings.SEGMAMBA_GPU_TRANSFORMS_ENABLED = True
train_module.settings.SEGMAMBA_DERF_NORM_ENABLED = True
train_module.settings.ADVANCED_SEGMAMBA_MAMBA_IMPL = "mamba1"
import sys
sys.modules["settings"] = train_module.settings

# Explicitly import segmamba first so it's loaded in sys.modules
import model_segmamba.segmamba

def apply_segmamba_monkeypatch():
    for name, mod in list(sys.modules.items()):
        if name.endswith("segmamba") and hasattr(mod, "_load_project_settings") and not hasattr(mod._load_project_settings, "_is_monkeypatched"):
            print(f"[*] Monkeypatching _load_project_settings in module: {name}")
            orig = mod._load_project_settings
            def make_custom_load(original_load):
                def custom_load():
                    print(f"[*] custom_load called for namespace of {name}")
                    s = original_load()
                    if s is not None:
                        print("[*] Customizing loaded settings: setting ADVANCED_SEGMAMBA_MAMBA_IMPL to mamba1")
                        s.SEGMAMBA_GPU_TRANSFORMS_ENABLED = True
                        s.SEGMAMBA_DERF_NORM_ENABLED = True
                        s.ADVANCED_SEGMAMBA_MAMBA_IMPL = "mamba1"
                    else:
                        print("[!] original_load returned None!")
                    return s
                custom_load._is_monkeypatched = True
                return custom_load
            mod._load_project_settings = make_custom_load(orig)

# Apply monkeypatch immediately
apply_segmamba_monkeypatch()

BraTSTrainer = train_module.BraTSTrainer

# Determine available GPUs and set device/num_gpus
gpu_count = torch.cuda.device_count()
print(f"Detected CUDA devices count: {gpu_count}")
if gpu_count > 0:
    device = "cuda"
    num_gpus = 1
else:
    device = "cpu"
    num_gpus = 1
    print("[*] Falling back to CPU for testing since CUDA is not available or NVML cannot be initialized in this terminal.")

print("Initializing BraTSTrainer...")
trainer = BraTSTrainer(
    env_type="pytorch",
    max_epochs=1,
    batch_size=1,
    device=device,
    val_every=1,
    num_gpus=num_gpus,
    logdir="./scratch/logs/"
)

# Mocking loss function and model to speed up
print("Trainer initialized successfully.")
print("gpu_transforms_enabled:", trainer.gpu_transforms_enabled)
print("gpu_transform_fn:", trainer.gpu_transform_fn)

# Generate dummy batch
print(f"\nGenerating dummy batch on {device}...")
dummy_batch = {
    "data": torch.randn(1, 4, 32, 32, 32, device=device), # smaller size for faster execution
    "seg": torch.randint(0, 4, (1, 1, 32, 32, 32), device=device).float()
}

print("Dummy batch shapes:")
print("  - data:", dummy_batch["data"].shape)
print("  - seg:", dummy_batch["seg"].shape)

print("\nRunning trainer.training_step(dummy_batch)...")
try:
    # Just run the transform part first to see if ValueError is gone
    if trainer.gpu_transform_fn is not None:
        print("[*] Applying GPU transform manually inside script first to test...")
        
        any_key = next(iter(trainer.gpu_transform_fn.keys))
        batch_size = dummy_batch[any_key].shape[0]
        
        squeezed_batch = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in dummy_batch.items()
        }
        print("  - Squeezed data shape:", squeezed_batch["data"].shape)
        print("  - Squeezed seg shape:", squeezed_batch["seg"].shape)
        
        # Squeeze rand_affine keys if not on GPU (MONAI RandAffined with rotate might fail on CPU without device arg, but let's see)
        transformed = trainer.gpu_transform_fn(squeezed_batch)
        print("[+] RandAffined transformation successfully applied without ValueError!")
        
        final_batch = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in transformed.items()
        }
        print("  - Restored data shape:", final_batch["data"].shape)
        print("  - Restored seg shape:", final_batch["seg"].shape)
        
    print("\nRunning full trainer.training_step()...")
    # Actually training_step will do forward pass of SegMamba
    loss = trainer.training_step(dummy_batch)
    print(f"[+] SUCCESS: training_step completed successfully! Loss = {loss.item():.4f}")
    
except Exception as e:
    import traceback
    print("[!] Error occurred:")
    traceback.print_exc()
