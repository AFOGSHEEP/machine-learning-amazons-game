@echo off
cd /d "%~dp0.."

if "%PYTORCH_GPU_PYTHON%"=="" set "PYTORCH_GPU_PYTHON=E:\Anaconda\envs\myPytorch\python.exe"

if not exist "%PYTORCH_GPU_PYTHON%" (
  echo ERROR: Python not found: %PYTORCH_GPU_PYTHON%
  exit /b 1
)

"%PYTORCH_GPU_PYTHON%" scripts\check_cuda.py || exit /b 1

"%PYTORCH_GPU_PYTHON%" -m src.main train-dqn-pool --episodes 3000 --device cuda --log-csv results/train_dqn_opponent_pool_gpu.csv
