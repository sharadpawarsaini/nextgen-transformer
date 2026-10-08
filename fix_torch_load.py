import os

def safe_torch_load(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace simple torch.load calls with safe_torch_load helper or inline weights_only=False
    # Check if safe_torch_load helper already defined
    if "def safe_torch_load" not in content:
        helper = """import torch

def safe_torch_load(file_path, map_location=None):
    try:
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location=map_location)
"""
        # Insert helper at top after imports
        content = helper + content

    # Replace torch.load calls with safe_torch_load
    content = content.replace("torch.load(i, map_location=torch.device('cpu'))", "safe_torch_load(i, map_location=torch.device('cpu'))")
    content = content.replace("torch.load(fname, map_location=torch.device('cpu'))", "safe_torch_load(fname, map_location=torch.device('cpu'))")
    content = content.replace("torch.load(fname)", "safe_torch_load(fname)")
    content = content.replace("torch.load(pretrain_weight, map_location='cpu')", "safe_torch_load(pretrain_weight, map_location='cpu')")
    content = content.replace("torch.load('/home/xychen/jsguo/weight/gelunorm_former_skip_global_shift.model', map_location='cpu')", "safe_torch_load('/home/xychen/jsguo/weight/gelunorm_former_skip_global_shift.model', map_location='cpu')")
    content = content.replace("torch.load(\"/home/xychen/jsguo/weight/tumor_pretrain.model\", map_location='cpu')", "safe_torch_load(\"/home/xychen/jsguo/weight/tumor_pretrain.model\", map_location='cpu')")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated torch.load in:", os.path.basename(filepath))

files = [
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\model_restore.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\run\load_pretrained_weights.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\network_training\network_trainer.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\network_training\network_trainer_synapse.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\network_training\nnFormerTrainerV2_nnformer_acdc.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\network_training\nnFormerTrainerV2_nnformer_synapse.py",
    r"c:\Users\shara\Desktop\nnFormer\nnformer\training\network_training\nnFormerTrainerV2_nnformer_tumor.py",
]

for f in files:
    if os.path.exists(f):
        safe_torch_load(f)
