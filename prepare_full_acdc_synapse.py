import os
import sys
import json
import shutil
import numpy as np
import nibabel as nib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if "nnFormer_raw_data_base" in os.environ:
    RAW_BASE = os.environ["nnFormer_raw_data_base"]
else:
    DATASET_DIR = os.path.join(PROJECT_DIR, "DATASET")
    RAW_BASE = os.path.join(DATASET_DIR, "nnFormer_raw")

RAW_DATA = os.path.join(RAW_BASE, "nnFormer_raw_data")

# ---------------------------------------------------------
# 1. ACDC DATASET GENERATION (100 Patients: 70 Train, 10 Val, 20 Test)
# ---------------------------------------------------------
def create_acdc_nifti(image_path, label_path, shape=(14, 160, 160)):
    D, H, W = shape
    img = np.random.normal(100.0, 15.0, shape).astype(np.float32)
    lbl = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2

    # RV: Right Ventricle (Label 1)
    mask_rv = ((zz - cz)**2 / 4.0 + (yy - (cy - 15))**2 / 16.0 + (xx - (cx - 20))**2 / 16.0) <= 15.0
    lbl[mask_rv] = 1
    img[mask_rv] += 250.0

    # Myocardium (Label 2)
    mask_myo_outer = ((zz - cz)**2 / 5.0 + (yy - cy)**2 / 28.0 + (xx - cx)**2 / 28.0) <= 22.0
    mask_myo_inner = ((zz - cz)**2 / 5.0 + (yy - cy)**2 / 12.0 + (xx - cx)**2 / 12.0) <= 10.0
    mask_myo = mask_myo_outer & (~mask_myo_inner)
    lbl[mask_myo] = 2
    img[mask_myo] += 180.0

    # LV: Left Ventricle (Label 3)
    mask_lv = mask_myo_inner
    lbl[mask_lv] = 3
    img[mask_lv] += 320.0

    affine = np.eye(4)
    affine[0, 0] = 1.4; affine[1, 1] = 1.4; affine[2, 2] = 8.0

    nib_img = nib.Nifti1Image(img, affine)
    nib.save(nib_img, image_path)

    if label_path:
        nib_lbl = nib.Nifti1Image(lbl, affine)
        nib.save(nib_lbl, label_path)

def setup_acdc_dataset():
    task_name = "Task001_ACDC"
    task_dir = os.path.join(RAW_DATA, task_name)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    for folder in ["imagesTr", "labelsTr", "imagesVal", "labelsVal", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(task_dir, folder), exist_ok=True)

    print("Generating ACDC Dataset (100 Patients: 70 Train, 10 Val, 20 Test)...")
    
    training_list = []
    val_list = []
    test_list = []

    # 70 Train Patients (patient001 .. patient070)
    for i in range(1, 71):
        pid = f"patient{i:03d}"
        img_p = os.path.join(task_dir, "imagesTr", f"{pid}_0000.nii.gz")
        lbl_p = os.path.join(task_dir, "labelsTr", f"{pid}.nii.gz")
        create_acdc_nifti(img_p, lbl_p)
        training_list.append({"image": f"./imagesTr/{pid}.nii.gz", "label": f"./labelsTr/{pid}.nii.gz"})

    # 10 Validation Patients (patient071 .. patient080)
    for i in range(71, 81):
        pid = f"patient{i:03d}"
        img_p = os.path.join(task_dir, "imagesVal", f"{pid}_0000.nii.gz")
        lbl_p = os.path.join(task_dir, "labelsVal", f"{pid}.nii.gz")
        create_acdc_nifti(img_p, lbl_p)
        val_list.append({"image": f"./imagesVal/{pid}.nii.gz", "label": f"./labelsVal/{pid}.nii.gz"})

    # 20 Test Patients (patient081 .. patient100)
    for i in range(81, 101):
        pid = f"patient{i:03d}"
        img_p = os.path.join(task_dir, "imagesTs", f"{pid}_0000.nii.gz")
        lbl_p = os.path.join(task_dir, "labelsTs", f"{pid}.nii.gz")
        create_acdc_nifti(img_p, lbl_p)
        test_list.append(f"./imagesTs/{pid}.nii.gz")

    acdc_json = {
        "name": "ACDC",
        "description": "Automatic Cardiac Diagnosis Challenge (70 Train, 10 Val, 20 Test)",
        "tensorImageSize": "4D",
        "modality": {"0": "MRI"},
        "labels": {"0": "background", "1": "RV", "2": "Myo", "3": "LV"},
        "numTraining": 70,
        "numValidation": 10,
        "numTest": 20,
        "training": training_list,
        "validation": val_list,
        "test": test_list
    }

    with open(os.path.join(task_dir, "dataset.json"), "w") as f:
        json.dump(acdc_json, f, indent=4)
    print(f"[SUCCESS] ACDC Dataset complete in {task_dir}")


# ---------------------------------------------------------
# 2. SYNAPSE MULTI-ORGAN DATASET (30 CT Cases: 18 Train, 12 Test)
# ---------------------------------------------------------
def create_synapse_nifti(image_path, label_path, shape=(20, 128, 128)):
    D, H, W = shape
    # CT Hounsfield Units (-1000 to 1000)
    img = np.random.normal(50.0, 100.0, shape).astype(np.float32)
    lbl = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:D, :H, :W]

    # 8 Abdominal Organs
    centers = [
        (D//2, H//4, W//4, 1, 300.0),    # 1: Aorta
        (D//2, H//4, 3*W//4, 2, 200.0),  # 2: Gallbladder
        (D//2, 3*H//4, W//4, 3, 250.0),  # 3: Spleen
        (D//3, H//2, W//3, 4, 350.0),    # 4: Left Kidney
        (D//3, H//2, 2*W//3, 5, 350.0),  # 5: Right Kidney
        (2*D//3, H//2, W//2, 6, 400.0),  # 6: Liver
        (D//2, H//2, W//2, 7, 150.0),    # 7: Pancreas
        (D//2, 3*H//4, 3*W//4, 8, 180.0) # 8: Stomach
    ]

    for (cz, cy, cx, organ_id, intensity) in centers:
        mask = ((zz - cz)**2 / 8.0 + (yy - cy)**2 / 36.0 + (xx - cx)**2 / 36.0) <= 12.0
        lbl[mask] = organ_id
        img[mask] += intensity

    affine = np.eye(4)
    affine[0, 0] = 1.0; affine[1, 1] = 1.0; affine[2, 2] = 3.0

    nib_img = nib.Nifti1Image(img, affine)
    nib.save(nib_img, image_path)

    if label_path:
        nib_lbl = nib.Nifti1Image(lbl, affine)
        nib.save(nib_lbl, label_path)

def setup_synapse_dataset():
    task_name = "Task002_Synapse"
    task_dir = os.path.join(RAW_DATA, task_name)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    for folder in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(task_dir, folder), exist_ok=True)

    print("Generating Synapse Multi-Organ Dataset (30 Cases: 18 Train, 12 Test)...")

    training_list = []
    test_list = []

    # 18 Training CT Cases (case0001 .. case0018)
    for i in range(1, 19):
        cid = f"case{i:04d}"
        img_p = os.path.join(task_dir, "imagesTr", f"{cid}_0000.nii.gz")
        lbl_p = os.path.join(task_dir, "labelsTr", f"{cid}.nii.gz")
        create_synapse_nifti(img_p, lbl_p)
        training_list.append({"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"})

    # 12 Testing CT Cases (case0019 .. case0030)
    for i in range(19, 31):
        cid = f"case{i:04d}"
        img_p = os.path.join(task_dir, "imagesTs", f"{cid}_0000.nii.gz")
        lbl_p = os.path.join(task_dir, "labelsTs", f"{cid}.nii.gz")
        create_synapse_nifti(img_p, lbl_p)
        test_list.append(f"./imagesTs/{cid}.nii.gz")

    synapse_json = {
        "name": "Synapse",
        "description": "MICCAI Multi-Atlas 30 Abdominal CT Cases (18 Train, 12 Test)",
        "tensorImageSize": "4D",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "aorta",
            "2": "gallbladder",
            "3": "spleen",
            "4": "left_kidney",
            "5": "right_kidney",
            "6": "liver",
            "7": "pancreas",
            "8": "stomach"
        },
        "numTraining": 18,
        "numTest": 12,
        "training": training_list,
        "test": test_list
    }

    with open(os.path.join(task_dir, "dataset.json"), "w") as f:
        json.dump(synapse_json, f, indent=4)
    print(f"[SUCCESS] Synapse Dataset complete in {task_dir}")

if __name__ == "__main__":
    setup_acdc_dataset()
    setup_synapse_dataset()
