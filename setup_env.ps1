$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:nnFormer_raw_data_base = Join-Path $ScriptDir "DATASET\nnFormer_raw"
$env:nnFormer_preprocessed = Join-Path $ScriptDir "DATASET\nnFormer_preprocessed"
$env:RESULTS_FOLDER = Join-Path $ScriptDir "DATASET\nnFormer_trained_models"

Write-Host "nnFormer environment variables configured:" -ForegroundColor Green
Write-Host "nnFormer_raw_data_base = $env:nnFormer_raw_data_base"
Write-Host "nnFormer_preprocessed  = $env:nnFormer_preprocessed"
Write-Host "RESULTS_FOLDER        = $env:RESULTS_FOLDER"

& (Join-Path $ScriptDir ".venv\Scripts\Activate.ps1")
