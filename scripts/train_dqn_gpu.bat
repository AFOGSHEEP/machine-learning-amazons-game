@echo off
REM Train DQN on GPU. Set PYTORCH_GPU_PYTHON if your CUDA env is not at the default path.
cd /d "%~dp0.."

if "%PYTORCH_GPU_PYTHON%"=="" set "PYTORCH_GPU_PYTHON=E:\Anaconda\envs\myPytorch\python.exe"

if not exist "%PYTORCH_GPU_PYTHON%" (
  echo ERROR: Python not found: %PYTORCH_GPU_PYTHON%
  echo Set PYTORCH_GPU_PYTHON to your conda env python ^(with torch+cuda^).
  exit /b 1
)

"%PYTORCH_GPU_PYTHON%" scripts\check_cuda.py || exit /b 1

echo Using: %PYTORCH_GPU_PYTHON%
"%PYTORCH_GPU_PYTHON%" -m src.main train-dqn --episodes 2000 --device cuda --use-per --n-step 3 --log-csv results/train_dqn_rainbow_gpu.csv
