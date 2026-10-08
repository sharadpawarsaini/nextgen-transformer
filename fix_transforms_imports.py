import os

files_to_fix = [
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\default_data_augmentation.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_moreDA.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_insaneDA.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_insaneDA2.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\data_augmentation_noDA.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\custom_transforms.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\downsampling.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\pyramid_augmentations.py",
]

TRANSFORM_IMPORTS_HEADER = """try:
    from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
except ImportError:
    from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform

try:
    from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
except ImportError:
    from batchgenerators.transforms import SpatialTransform, MirrorTransform

try:
    from batchgenerators.transforms.color_transforms import GammaTransform
except ImportError:
    from batchgenerators.transforms import GammaTransform

try:
    from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
except ImportError:
    from batchgenerators.transforms import Compose, AbstractTransform
"""

for filepath in files_to_fix:
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if "from batchgenerators.transforms import" in line:
            # Add robust header block once
            new_lines.append(TRANSFORM_IMPORTS_HEADER + "\n")
            # If line ends with backslash, skip multi-line import continuation
            if line.strip().endswith("\\"):
                # look ahead to skip wrapped imports
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith("GammaTransform") or lines[j].strip().startswith("SegChannelSelectionTransform") or lines[j].strip().startswith("SpatialTransform") or lines[j].strip().startswith("Compose") or lines[j].strip().startswith("AbstractTransform")):
                    j += 1
        elif any(token in line for token in ["DataChannelSelectionTransform", "SegChannelSelectionTransform", "SpatialTransform,", "GammaTransform,", "MirrorTransform,", "Compose"]) and "try:" not in line and "except" not in line and "def " not in line and "class " not in line and "=" not in line:
            # Skip old import lines handled by header
            continue
        else:
            new_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Fixed imports in:", os.path.basename(filepath))
