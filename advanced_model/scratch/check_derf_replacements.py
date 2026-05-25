import os
import sys
import torch

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BRATS23_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    
    # Ensure correct sys.path order
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != BASE_DIR
    ]
    if BRATS23_DIR not in sys.path:
        sys.path.insert(0, BRATS23_DIR)
    sys.path.append(BASE_DIR)

    print("[*] Importing segmamba module...")
    import model_segmamba.segmamba as segmamba_module
    import torch.nn as nn

    # --- MONKEYPATCHING TO FORCE ENABLE DERF NORM ---
    orig_load_settings = segmamba_module._load_project_settings

    def mock_load_settings():
        settings = orig_load_settings()
        if settings is not None:
            settings.SEGMAMBA_DERF_NORM_ENABLED = True
            settings.SEGMAMBA_DERF_ALPHA_INIT_VALUE = 0.5
            settings.SEGMAMBA_DERF_SHIFT_INIT_VALUE = 0.0
            print("[*] [MockSettings] Overrode SEGMAMBA_DERF_NORM_ENABLED = True")
        else:
            print("[!] [MockSettings] Original settings.py not found!")
        return settings

    segmamba_module._load_project_settings = mock_load_settings
    print("[*] Monkeypatched _load_project_settings successfully.")

    # Now load SegMamba and DynamicErf classes
    from model_segmamba.segmamba import SegMamba, DynamicErf

    print("\nInitializing SegMamba with Derf enabled...")
    model = SegMamba()

    print("\nScanning all modules in SegMamba to locate normalization layers:")
    derf_count = 0
    remaining_layernorms = []
    instancenorms = []
    rmsnorms = []

    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        class_fullname = f"{module.__class__.__module__}.{class_name}"
        
        # 1. Check for DynamicErf
        if isinstance(module, DynamicErf) or class_name == "DynamicErf":
            derf_count += 1
            # print(f"  [DynamicErf] {name}")
        # 2. Check for remaining LayerNorms (both PyTorch & Custom)
        elif "LayerNorm" in class_name:
            remaining_layernorms.append((name, class_fullname))
        # 3. Check for InstanceNorms (which we preserve)
        elif "InstanceNorm" in class_name:
            instancenorms.append((name, class_fullname))
        # 4. Check for RMSNorms (from mamba_ssm Triton kernels, which we also preserve)
        elif "RMSNorm" in class_name:
            rmsnorms.append((name, class_fullname))

    print(f"\n[*] SCAN SUMMARY:")
    print(f"  - Total DynamicErf layers successfully replaced: {derf_count}")
    print(f"  - Total remaining (non-replaced) LayerNorm layers: {len(remaining_layernorms)}")
    print(f"  - Total preserved InstanceNorm3d layers: {len(instancenorms)}")
    print(f"  - Total preserved RMSNorm layers (Mamba Triton): {len(rmsnorms)}")
    
    if len(remaining_layernorms) > 0:
        print("\n[!] WARNING: The following LayerNorms were NOT replaced:")
        for name, fullname in remaining_layernorms:
            print(f"  - {name}: {fullname}")
    else:
        print("\n[+] SUCCESS: ALL LayerNorm layers (both PyTorch and Custom) were COMPLETELY and SUCCESSFULLY replaced by DynamicErf!")

if __name__ == "__main__":
    main()
