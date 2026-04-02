@echo off
cd /d "%~dp0.."
set PORT=8503

REM Prefer CUDA env: Anaconda base often has torch+cpu only.
if "%PYTORCH_GPU_PYTHON%"=="" set "PYTORCH_GPU_PYTHON=E:\Anaconda\envs\myPytorch\python.exe"
if exist "%PYTORCH_GPU_PYTHON%" (
  set "DEMO_PY=%PYTORCH_GPU_PYTHON%"
) else (
  set "DEMO_PY=python"
)

echo Starting Streamlit demo on port %PORT% using:
echo   %DEMO_PY%
"%DEMO_PY%" -m streamlit run app/demo_streamlit.py --server.port %PORT%
