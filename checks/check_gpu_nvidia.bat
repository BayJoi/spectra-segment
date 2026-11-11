@echo off
setlocal
title Spectra Segment - NVIDIA GPU Health Check

set "ROOT=%~dp0..\"
set "CHECKS=%~dp0"
set "PY=%ROOT%backend\.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo [FAIL] No CPU/NVIDIA virtual environment found at:
    echo        %ROOT%backend\.venv
    echo        Run install.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)

echo Using CPU/NVIDIA Python: %PY%
echo [WARN] NVIDIA hardware has NOT been tested by this project.
echo [WARN] This run is only evidence for the exact machine and wheel used here.
echo.
"%PY%" -X utf8 "%CHECKS%check_gpu_health.py" --backend nvidia %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo [ OK ] NVIDIA-path health check passed for this machine.
    echo       That is not a project-wide NVIDIA validation.
) else (
    echo [FAIL] NVIDIA-path health check reported failures. See the log above / checks\*.log
)
if not defined SPECTRA_NO_PAUSE pause >nul
exit /b %RC%
