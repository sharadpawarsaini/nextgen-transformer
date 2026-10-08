import os
import re

AUGMENTER_FALLBACK = """try:
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
except ImportError:
    try:
        from batchgenerators.dataloading import MultiThreadedAugmenter
    except ImportError:
        try:
            from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter as MultiThreadedAugmenter
        except ImportError:
            class MultiThreadedAugmenter:
                def __init__(self, dataloader, transform, num_processes=1, num_cached_per_queue=1, seeds=None, pin_memory=False):
                    self.generator = dataloader
                    self.transform = transform
                def __iter__(self):
                    return self
                def __next__(self):
                    item = next(self.generator)
                    return self.transform(**item) if callable(self.transform) else item"""

SLIM_FALLBACK = """try:
    from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
except ImportError:
    try:
        from batchgenerators.dataloading import SlimDataLoaderBase
    except ImportError:
        class SlimDataLoaderBase:
            def __init__(self, data, batch_size, number_of_threads_in_multithreading=1):
                self._data = data
                self.batch_size = batch_size
                self.number_of_threads_in_multithreading = number_of_threads_in_multithreading"""

files_augmenter = [
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\default_data_augmentation.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_insaneDA.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_insaneDA2.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_moreDA.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_noDA.py"
]

for filepath in files_augmenter:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace any MultiThreadedAugmenter import block
    pattern = r"(try:\s+from batchgenerators\.dataloading.*?\nexcept ImportError:.*?\n\s+from batchgenerators\.dataloading import MultiThreadedAugmenter|from batchgenerators\.dataloading import MultiThreadedAugmenter)"
    new_content = re.sub(pattern, AUGMENTER_FALLBACK, content, flags=re.DOTALL)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Updated {os.path.basename(filepath)}")

# Update dataset_loading.py
dataset_loading_path = r"c:\Users\shara\Desktop\nnFormer\nnformer\training\dataloading\dataset_loading.py"
with open(dataset_loading_path, "r", encoding="utf-8") as f:
    content = f.read()

pattern = r"(try:\s+from batchgenerators\.dataloading.*?\nexcept ImportError:.*?\n\s+from batchgenerators\.dataloading import SlimDataLoaderBase|from batchgenerators\.dataloading import SlimDataLoaderBase)"
new_content = re.sub(pattern, SLIM_FALLBACK, content, flags=re.DOTALL)

with open(dataset_loading_path, "w", encoding="utf-8") as f:
    f.write(new_content)
print("Updated dataset_loading.py")
