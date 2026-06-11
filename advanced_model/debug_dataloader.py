import os
import sys
import time

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

    print("Importing dataset & trainer modules...")
    from light_training.dataloading.dataset import get_train_val_test_loader_from_split_json
    from light_training.dataloading.base_data_loader import DataLoaderMultiProcess
    from light_training.augment.multi_processor import create_limited_len_augmenter
    from light_training.augment.train_augment import get_train_transforms

    data_dir = "/home/cuc.buithi/BRATS/data/fullres/train"
    split_json_file = os.path.abspath(os.path.join(BASE_DIR, "..", "brats23_split_70_10_20.json"))

    print("Loading dataset from split JSON...")
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_split_json(data_dir, split_json_file)

    print(f"Creating DataLoaderMultiProcess with batch_size={settings.BATCH_SIZE}...")
    train_loader = DataLoaderMultiProcess(train_ds, 
                                          batch_size=settings.BATCH_SIZE,
                                          patch_size=settings.INPUT_SIZE,
                                          print_time=True)

    print("Getting transforms...")
    tr_transforms = get_train_transforms(
        patch_size=settings.INPUT_SIZE,
        mirror_axes=[0, 1, 2],
        modality_dropout_prob=0.3 if settings.SEGMAMBA_MODALITY_DROPOUT_ENABLED else 0.0,
        modality_dropout_max_channels=settings.SEGMAMBA_MODALITY_DROPOUT_MAX_CHANNELS
    )

    print(f"Creating augmenter with mode={settings.SEGMAMBA_AUGMENTER_BACKEND}...")
    data_generator = create_limited_len_augmenter(
        mode=settings.SEGMAMBA_AUGMENTER_BACKEND,
        my_imaginary_length=250,
        data_loader=train_loader,
        transform=tr_transforms,
        num_processes=1,
        num_cached=6,
        seeds=[1234],
        pin_memory=True,
        wait_time=0.02,
    )

    print("Calling next(data_generator) to fetch the first batch...")
    try:
        start_time = time.time()
        batch = next(data_generator)
        print(f"Successfully fetched a batch in {time.time() - start_time:.4f} seconds!")
        print("Keys in batch:", batch.keys())
        print("Data shape in batch:", batch["data"].shape)
    except Exception as e:
        print("Error occurred during next():", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
