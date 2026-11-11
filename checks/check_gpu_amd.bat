@echo off
setlocal
title Spectra Segment - AMD GPU Health Check

set "ROOT=%~dp0..\"
set "CHECKS=%~dp0"
set "PY=%ROOT%backend_amd_gpu\.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo [FAIL] No AMD ROCm virtual environment found at:
    echo        %ROOT%backend_amd_gpu\.venv
    echo        Run install_amd.bat first.
    if not defined SPECTRA_NO_PAUSE pause >nul
    exit /b 1
)

echo Using AMD Python: %PY%
echo.
"%PY%" -X utf8 "%CHECKS%check_gpu_health.py" --backend amd %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo [ OK ] AMD health check passed. Log saved beside this script.
) else (
    echo [FAIL] AMD health check reported failures. See the log above / checks\*.log
)
if not defined SPECTRA_NO_PAUSE pause >nul
exit /b %RC%
