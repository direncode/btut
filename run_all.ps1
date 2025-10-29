# BTUT Auto Runner
Write-Host "Initializing BTUT Demo Environment..." -ForegroundColor Cyan

py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run all demos
py btut_grok_test.py --demo all

# Run diagnostics and scaling
py btut_diagnostics.py --gamma 1.45 --cA_SH 0.40 --alpha 0.60 --tau 0.30 --N 300000 --iters 20 --reps 5
py btut_scaling_test.py

# Generate GIF animations
py btut_animate.py --mode converge
py btut_animate.py --mode phase

# Ensure results folder exists
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipFolder = "results_zip"
if (-not (Test-Path $zipFolder)) {
    New-Item -ItemType Directory -Path $zipFolder | Out-Null
}

# Zip results
Compress-Archive -Path plots\* -DestinationPath "$zipFolder\BTUT_results_$timestamp.zip" -Force
Write-Host "All tasks complete. Results stored in results_zip." -ForegroundColor Green



