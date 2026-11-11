@echo off
setlocal enabledelayedexpansion
title Spectra Segment - GPU Health Check

set "ROOT=%~dp0..\"
set "CHECKS=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "NVIDIA_DIR=%ROOT%backend"
set "HAS_AMD_VENV=0"
set "HAS_NVIDIA_VENV=0"
set "HAS_AMD_GPU=0"
set "HAS_NVIDIA_GPU=0"

if exist "%AMD_DIR%\.venv\Scripts\python.exe" set "HAS_AMD_VENV=1"
if exist "%NVIDIA_DIR%\.venv\Scripts\python.exe" set "HAS_NVIDIA_VENV=1"

if "%HAS_AMD_VENV%"=="0" if "%HAS_NVIDIA_VENV%"=="0" (
    echo [FAIL] No backend virtual environment found.
    echo        Run install_amd.bat or install.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)

for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"`) do (
    echo %%G | findstr /i "AMD Radeon" >nul && set "HAS_AMD_GPU=1"
    echo %%G | findstr /i "NVIDIA" >nul && set "HAS_NVIDIA_GPU=1"
)

if defined SPECTRA_CHECK_BACKEND (
    if /i "%SPECTRA_CHECK_BACKEND%"=="amd" goto :amd
    if /i "%SPECTRA_CHECK_BACKEND%"=="nvidia" goto :nvidia
    if /i "%SPECTRA_CHECK_BACKEND%"=="cpu" goto :cpu
    echo [FAIL] SPECTRA_CHECK_BACKEND must be amd, nvidia or cpu.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 2
)

if "%HAS_AMD_GPU%"=="1" if "%HAS_NVIDIA_GPU%"=="1" if "%HAS_AMD_VENV%"=="1" if "%HAS_NVIDIA_VENV%"=="1" (
    echo [WARN] Both AMD and NVIDIA display adapters were detected.
    echo        Use checks\check_gpu_amd.bat or checks\check_gpu_nvidia.bat
    echo        so the correct PyTorch environment is tested.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 2
)

if "%HAS_AMD_GPU%"=="1" if "%HAS_AMD_VENV%"=="1" goto :amd
if "%HAS_NVIDIA_GPU%"=="1" if "%HAS_NVIDIA_VENV%"=="1" goto :nvidia

if "%HAS_AMD_GPU%"=="1" (
    echo [WARN] AMD adapter detected, but backend_amd_gpu\.venv is missing.
)
if "%HAS_NVIDIA_GPU%"=="1" (
    echo [WARN] NVIDIA adapter detected, but backend\.venv is missing.
)
if "%HAS_NVIDIA_VENV%"=="1" (
    echo [INFO] No matching GPU environment; running a CPU smoke test.
    goto :cpu
)

echo [FAIL] No usable backend environment was found for this machine.
echo        Run install.bat for CPU/NVIDIA, or install_amd.bat for AMD.
if not defined SPECTRA_NO_PAUSE pause >nul
exit /b 1

:amd
if "%HAS_AMD_VENV%"=="0" (
    echo [FAIL] AMD adapter detected, but backend_amd_gpu\.venv is missing.
    echo        Run install_amd.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)
call "%CHECKS%check_gpu_amd.bat" %*
exit /b %ERRORLEVEL%

:nvidia
if "%HAS_NVIDIA_VENV%"=="0" (
    echo [FAIL] NVIDIA/CPU backend selected, but backend\.venv is missing.
    echo        Run install.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)
call "%CHECKS%check_gpu_nvidia.bat" %*
exit /b %ERRORLEVEL%

:cpu
if "%HAS_NVIDIA_VENV%"=="0" (
    echo [FAIL] CPU mode uses backend\.venv, which is missing.
    echo        Run install.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)
set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "SPECTRA_FORCE_CPU=1"
"%NVIDIA_DIR%\.venv\Scripts\python.exe" -X utf8 "%CHECKS%check_gpu_health.py" --backend cpu %*
set "RC=%ERRORLEVEL%"
if not defined SPECTRA_NO_PAUSE pause >nul
exit /b %RC%
