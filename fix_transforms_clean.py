import os
import glob

TRANSFORMS_FALLBACK = """try:
    from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
    from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
    from batchgenerators.transforms.color_transforms import GammaTransform
    from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
except ImportError:
    from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, GammaTransform, MirrorTransform, Compose, AbstractTransform
"""

py_files = glob.glob(r"c:\Users\shara\Desktop\nnFormer\nnformer\training\data_augmentation\*.py")

for filepath in py_files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "from batchgenerators.transforms import" in content:
        # Split into lines
        lines = content.splitlines()
        new_lines = []
        skip = False
        replaced = False
        for line in lines:
            if "from batchgenerators.transforms import" in line:
                if not replaced:
                    new_lines.append(TRANSFORMS_FALLBACK)
                    replaced = True
                if line.endswith("\\"):
                    skip = True
                continue
            if skip:
                if line.endswith("\\"):
                    continue
                else:
                    skip = False
                    continue
            new_lines.append(line)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print("Successfully updated:", os.path.basename(filepath))
