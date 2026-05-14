from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len


class _LimitedLenAdapter:
    def __init__(self, my_imaginary_length, augmenter):
        self.len = my_imaginary_length
        self.augmenter = augmenter

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.augmenter)

    def __getattr__(self, name):
        return getattr(self.augmenter, name)


def create_limited_len_augmenter(
    mode,
    my_imaginary_length,
    data_loader,
    transform,
    num_processes=12,
    num_cached=6,
    seeds=None,
    pin_memory=True,
    wait_time=0.02,
):
    mode = str(mode).lower()

    if mode in {"single", "single-thread", "singlethread"}:
        augmenter = SingleThreadedAugmenter(data_loader=data_loader, transform=transform)
        return _LimitedLenAdapter(my_imaginary_length, augmenter)

    if mode in {"multi", "multi-thread", "multithread"}:
        augmenter = MultiThreadedAugmenter(
            data_loader,
            transform,
            num_processes,
            num_cached,
            seeds,
            pin_memory=pin_memory,
        )
        return _LimitedLenAdapter(my_imaginary_length, augmenter)

    if mode in {"nondet", "nondet-multiprocess", "multiprocess"}:
        return LimitedLenWrapper(
            my_imaginary_length,
            data_loader=data_loader,
            transform=transform,
            num_processes=num_processes,
            num_cached=num_cached,
            seeds=seeds,
            pin_memory=pin_memory,
            wait_time=wait_time,
        )

    raise ValueError(
        f"Unsupported augmenter mode: {mode}. "
        "Use one of: single-thread, multi-thread, nondet-multiprocess."
    )