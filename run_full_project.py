"""
nnFormer Full End-to-End Pipeline Execution Script
Runs environment configuration, data generation/preprocessing,
nnFormer 3D training, model inference, and metric evaluation.
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "DATASET")
RAW_BASE = os.path.join(DATASET_DIR, "nnFormer_raw")
RAW_DATA = os.path.join(RAW_BASE, "nnFormer_raw_data")
CROPPED_DATA = os.path.join(RAW_BASE, "nnFormer_cropped_data")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "nnFormer_preprocessed")
RESULTS_DIR = os.path.join(DATASET_DIR, "nnFormer_trained_models")

# Set required environment variables before importing any nnformer modules
os.environ["nnFormer_raw_data_base"] = RAW_BASE
os.environ["nnFormer_preprocessed"] = PREPROCESSED_DIR
os.environ["RESULTS_FOLDER"] = RESULTS_DIR
os.environ["nnFormer_def_n_proc"] = "2"
os.environ["nnFormer_n_proc_DA"] = "1"
os.environ["MAX_NUM_EPOCHS"] = "100"
os.environ["NUM_BATCHES_PER_EPOCH"] = "10"
os.environ["NUM_VAL_BATCHES_PER_EPOCH"] = "5"

TASK_ID = 1
TASK_NAME = f"Task{TASK_ID:03d}_ACDC"
TASK_DIR = os.path.join(RAW_DATA, TASK_NAME)


def print_step(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n", flush=True)


def create_sample_nifti(image_path, label_path, shape=(14, 160, 160)):
    """Generate a realistic 3D volumetric image and ground-truth segmentation."""
    D, H, W = shape
    # Create background image with simulated MRI contrast
    img = np.random.normal(100.0, 15.0, shape).astype(np.float32)
    lbl = np.zeros(shape, dtype=np.uint8)

    # Coordinates
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2

    # Structure 1: RV (Right Ventricle, label 1)
    mask_rv = ((zz - cz)**2 / 4.0 + (yy - (cy - 15))**2 / 16.0 + (xx - (cx - 20))**2 / 16.0) <= 15.0
    lbl[mask_rv] = 1
    img[mask_rv] += 250.0

    # Structure 2: Myocardium (label 2)
    mask_myo_outer = ((zz - cz)**2 / 5.0 + (yy - cy)**2 / 28.0 + (xx - cx)**2 / 28.0) <= 22.0
    mask_myo_inner = ((zz - cz)**2 / 5.0 + (yy - cy)**2 / 12.0 + (xx - cx)**2 / 12.0) <= 10.0
    mask_myo = mask_myo_outer & (~mask_myo_inner)
    lbl[mask_myo] = 2
    img[mask_myo] += 180.0

    # Structure 3: LV (Left Ventricle blood pool, label 3)
    mask_lv = mask_myo_inner
    lbl[mask_lv] = 3
    img[mask_lv] += 320.0

    affine = np.eye(4)
    affine[0, 0] = 1.4
    affine[1, 1] = 1.4
    affine[2, 2] = 8.0

    nib_img = nib.Nifti1Image(img, affine)
    nib.save(nib_img, image_path)

    if label_path:
        nib_lbl = nib.Nifti1Image(lbl, affine)
        nib.save(nib_lbl, label_path)


def step1_prepare_dataset():
    print_step("STEP 1: Setting Up Directories & Sample Decathlon Dataset")
    if os.path.isfile(os.path.join(TASK_DIR, "dataset.json")):
        print(f"Dataset already prepared in {TASK_DIR}. Skipping generation!")
        return
    for d in [
        os.path.join(TASK_DIR, "imagesTr"),
        os.path.join(TASK_DIR, "labelsTr"),
        os.path.join(TASK_DIR, "imagesTs"),
        os.path.join(TASK_DIR, "labelsTs"),
        CROPPED_DATA,
        PREPROCESSED_DIR,
        RESULTS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    print("Generating sample 3D volumes (shape 14x160x160)...")
    create_sample_nifti(
        os.path.join(TASK_DIR, "imagesTr", "patient001_0000.nii.gz"),
        os.path.join(TASK_DIR, "labelsTr", "patient001.nii.gz")
    )
    create_sample_nifti(
        os.path.join(TASK_DIR, "imagesTr", "patient002_0000.nii.gz"),
        os.path.join(TASK_DIR, "labelsTr", "patient002.nii.gz")
    )
    create_sample_nifti(
        os.path.join(TASK_DIR, "imagesTs", "patient003_0000.nii.gz"),
        os.path.join(TASK_DIR, "labelsTs", "patient003.nii.gz")
    )

    dataset_json = {
        "name": "ACDC",
        "description": "Automatic Cardiac Diagnosis Challenge Demo",
        "tensorImageSize": "4D",
        "reference": "https://www.creatis.insa-lyon.fr/Challenge/acdc/",
        "licence": "see ACDC website",
        "release": "1.0",
        "modality": {"0": "MRI"},
        "labels": {
            "0": "background",
            "1": "RV",
            "2": "Myo",
            "3": "LV"
        },
        "numTraining": 2,
        "numTest": 1,
        "training": [
            {"image": "./imagesTr/patient001.nii.gz", "label": "./labelsTr/patient001.nii.gz"},
            {"image": "./imagesTr/patient002.nii.gz", "label": "./labelsTr/patient002.nii.gz"}
        ],
        "test": [
            "./imagesTs/patient003.nii.gz"
        ]
    }

    with open(os.path.join(TASK_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Dataset successfully prepared in {TASK_DIR}")


def step2_preprocess():
    print_step("STEP 2: Experiment Planning & Preprocessing (nnFormer_plan_and_preprocess)")
    plans_file = os.path.join(PREPROCESSED_DIR, TASK_NAME, "nnFormerPlansv2.1_plans_3D.pkl")
    if os.path.isfile(plans_file):
        print(f"Preprocessed plans already exist at {plans_file}. Skipping redundant preprocessing to run fast!")
        return
    cmd = [
        sys.executable,
        "-m", "nnformer.experiment_planning.nnFormer_plan_and_preprocess",
        "-t", str(TASK_ID),
        "-tl", "2",
        "-tf", "2",
        "--verify_dataset_integrity"
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ, check=True)
    print("Preprocessing completed successfully!")


def step3_train():
    print_step("STEP 3: Network Training (nnFormer_train)")
    cmd = [
        sys.executable,
        "-m", "nnformer.run.run_training",
        "3d_fullres",
        "nnFormerTrainerV2_nnformer_acdc",
        str(TASK_ID),
        "0"
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=os.environ, check=True)
    print("Training step completed successfully!")


def step4_predict():
    print_step("STEP 4: Model Inference (nnFormer_predict)")
    input_folder = os.path.join(TASK_DIR, "imagesTs")
    output_folder = os.path.join(TASK_DIR, "inferTs", "nnformer_acdc")
    os.makedirs(output_folder, exist_ok=True)

    chk = "model_best"
    chk_path = os.path.join(RESULTS_DIR, "nnFormer", "3d_fullres", TASK_NAME,
                            "nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1", "fold_0", "model_best.model")
    if not os.path.isfile(chk_path):
        chk = "model_final_checkpoint"
    print(f"Using checkpoint: {chk}")

    cmd = [
        sys.executable,
        "-m", "nnformer.inference.predict_simple",
        "-i", input_folder,
        "-o", output_folder,
        "-m", "3d_fullres",
        "-t", str(TASK_ID),
        "-f", "0",
        "-tr", "nnFormerTrainerV2_nnformer_acdc",
        "-chk", chk,
        "--disable_tta",
        "--num_threads_preprocessing", "1",
        "--num_threads_nifti_save", "1"
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=os.environ, check=True)
    print(f"Inference complete! Output saved to: {output_folder}")
    return output_folder


def dice_score(pred, gt):
    intersection = 2.0 * np.sum((pred > 0) & (gt > 0))
    total = np.sum(pred > 0) + np.sum(gt > 0)
    if total == 0:
        return 1.0
    return intersection / total


def step5_evaluate(infer_folder):
    print_step("STEP 5: Evaluating Predictions & Calculating Metrics")
    pred_file = os.path.join(infer_folder, "patient003.nii.gz")
    gt_file = os.path.join(TASK_DIR, "labelsTs", "patient003.nii.gz")

    if not os.path.isfile(pred_file):
        print(f"Prediction file not found at: {pred_file}")
        return

    pred_nib = nib.load(pred_file)
    gt_nib = nib.load(gt_file)
    pred_data = pred_nib.get_fdata()
    gt_data = gt_nib.get_fdata()

    print(f"Prediction shape: {pred_data.shape}, Unique labels: {np.unique(pred_data)}")
    print(f"Ground truth shape: {gt_data.shape}, Unique labels: {np.unique(gt_data)}")

    labels_map = {1: "Right Ventricle (RV)", 2: "Myocardium (Myo)", 3: "Left Ventricle (LV)"}
    print("\nDice Similarity Coefficient per Class:")
    print("-" * 50)
    for c, name in labels_map.items():
        score = dice_score(pred_data == c, gt_data == c)
        print(f"  {name:<25}: Dice = {score:.4f}")
    print("-" * 50)


def main():
    step1_prepare_dataset()
    step2_preprocess()
    step3_train()
    infer_folder = step4_predict()
    step5_evaluate(infer_folder)
    print_step("nnFormer Full Pipeline Executed Successfully!")


if __name__ == "__main__":
    main()
