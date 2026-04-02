$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$port = 8503
$gpuPy = if ($env:PYTORCH_GPU_PYTHON) { $env:PYTORCH_GPU_PYTHON } else { "E:\Anaconda\envs\myPytorch\python.exe" }
$demoPy = if (Test-Path $gpuPy) { $gpuPy } else { "python" }
Write-Host "Starting Streamlit demo on port $port using: $demoPy"
& $demoPy -m streamlit run app/demo_streamlit.py --server.port $port
