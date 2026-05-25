import os
import sys
import time
import torch

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BRATS23_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != BASE_DIR
    ]
    if BRATS23_DIR not in sys.path:
        sys.path.insert(0, BRATS23_DIR)
    sys.path.append(os.path.join(BRATS23_DIR, "advanced_model"))

    # Load real settings as a base
    print("Loading base settings...")
    import importlib.util
    settings_path = os.path.join(BRATS23_DIR, "settings.py")
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    import model_segmamba.segmamba as segmamba_mod
    
    global active_mock_settings
    active_mock_settings = None
    
    def mock_load():
        return active_mock_settings
        
    segmamba_mod._load_project_settings = mock_load
    SegMamba = segmamba_mod.SegMamba

    class MockSettings:
        def __init__(self, base_settings, vss_decoder, onsampling, vss_mamba3=False):
            for attr in dir(base_settings):
                if attr.isupper():
                    setattr(self, attr, getattr(base_settings, attr))
            self.SEGMAMBA_VSS_DECODER_ENABLED = vss_decoder
            self.SEGMAMBA_ONSAMPLING = onsampling
            self.SEGMAMBA_VSS_MAMBA3_ENABLED = vss_mamba3

    print("=" * 60)
    print("MOCK TEST: SEGMAMBA DECODER BENCHMARK")
    print("=" * 60)

    device = torch.device("cpu")
    
    # 1. Config A: Standard Decoder + Onsampling
    print("\n[Config A] Initializing standard decoder + onsampling...")
    mock_a = MockSettings(settings, vss_decoder=False, onsampling=True, vss_mamba3=False)
    active_mock_settings = mock_a
    t0 = time.time()
    model_a = SegMamba().to(device)
    init_time_a = time.time() - t0
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    print(f"Standard Decoder init successful in {init_time_a:.4f}s. Parameters: {params_a:,}")

    # 2. Config B: VSS Decoder + Onsampling (TSMamba fallback)
    print("\n[Config B] Initializing VSS Decoder + onsampling (TSMamba)...")
    mock_b = MockSettings(settings, vss_decoder=True, onsampling=True, vss_mamba3=False)
    active_mock_settings = mock_b
    t0 = time.time()
    model_b = SegMamba().to(device)
    init_time_b = time.time() - t0
    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"VSS Decoder (TSMamba) init successful in {init_time_b:.4f}s. Parameters: {params_b:,}")

    # 3. Config C: Pure VSSM3Block + Onsampling (Mamba 3)
    print("\n[Config C] Initializing VSS Decoder + onsampling (Pure VSSM3Block)...")
    mock_c = MockSettings(settings, vss_decoder=True, onsampling=True, vss_mamba3=True)
    active_mock_settings = mock_c
    t0 = time.time()
    model_c = SegMamba().to(device)
    init_time_c = time.time() - t0
    params_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    print(f"VSS Decoder (Pure VSSM3Block) init successful in {init_time_c:.4f}s. Parameters: {params_c:,}")

    print("\n" + "=" * 120)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 120)
    print(f"{'Metric':<30} | {'Config A: Standard':<26} | {'Config B: VSS (TSMamba)':<26} | {'Config C: VSS (Mamba 3)':<26}")
    print("-" * 120)
    print(f"{'Decoder Blocks type':<30} | {'Pseudo3DUpBlock':<26} | {'VSSUpBlock (TSMamba)':<26} | {'VSSUpBlock (VSSM3Block)':<26}")
    print(f"{'Total Trainable Parameters':<30} | {params_a:<26,} | {params_b:<26,} | {params_c:<26,}")
    print(f"{'Parameter Diff vs Baseline':<30} | {'Baseline':<26} | {f'{params_b - params_a:+,} ({((params_b - params_a)/params_a)*100:+.2f}%)':<26} | {f'{params_c - params_a:+,} ({((params_c - params_a)/params_a)*100:+.2f}%)':<26}")
    print(f"{'Initialization Time':<30} | {f'{init_time_a:.4f}s':<26} | {f'{init_time_b:.4f}s':<26} | {f'{init_time_c:.4f}s':<26}")
    print("=" * 120)

if __name__ == "__main__":
    main()
