@echo off
set "PROJECT_DIR=%~dp0"
set "nnFormer_raw_data_base=%PROJECT_DIR%DATASET\nnFormer_raw"
set "nnFormer_preprocessed=%PROJECT_DIR%DATASET\nnFormer_preprocessed"
set "RESULTS_FOLDER=%PROJECT_DIR%DATASET\nnFormer_trained_models"

echo nnFormer environment variables configured:
echo nnFormer_raw_data_base = %nnFormer_raw_data_base%
echo nnFormer_preprocessed  = %nnFormer_preprocessed%
echo RESULTS_FOLDER        = %RESULTS_FOLDER%

call "%PROJECT_DIR%.venv\Scripts\activate.bat"
cmd /k
