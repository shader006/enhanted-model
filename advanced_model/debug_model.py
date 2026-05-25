import os
import sys
import time
import torch

def main():
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

    print("Loading settings...")
    import importlib.util
    settings_path = os.path.join(BRATS23_DIR, "settings.py")
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    print("Importing SegMamba...")
    from model_segmamba.segmamba import SegMamba

    print("Initializing SegMamba model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    start_time = time.time()
    model = SegMamba().to(device)
    print(f"Model initialized and moved to device in {time.time() - start_time:.4f} seconds.")

    # Create dummy input: batch_size=1, channels=4, spatial_dims=(96, 96, 96)
    input_shape = (1, 4, 96, 96, 96)
    print(f"Generating dummy input of shape {input_shape}...")
    x = torch.randn(input_shape, dtype=torch.float32, device=device)

    print("Running forward pass...")
    try:
        start_time = time.time()
        pred = model(x)
        torch.cuda.synchronize()
        print(f"Forward pass completed in {time.time() - start_time:.4f} seconds!")
        print("Output shape:", pred.shape)
    except Exception as e:
        print("Error during forward pass:", e)
        import traceback
        traceback.print_exc()
        return

    print("Running backward pass...")
    try:
        start_time = time.time()
        loss = pred.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"Backward pass completed in {time.time() - start_time:.4f} seconds!")
    except Exception as e:
        print("Error during backward pass:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
