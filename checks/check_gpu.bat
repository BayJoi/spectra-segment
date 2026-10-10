@echo off
setlocal
title Spectra Segment - GPU Health Check

set "ROOT=%~dp0..\"
set "CHECKS=%~dp0"
set "PYTHONPATH=%ROOT%"

set "PY="
if exist "%ROOT%backend_amd_gpu\.venv\Scripts\python.exe" set "PY=%ROOT%backend_amd_gpu\.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%backend\.venv\Scripts\python.exe" set "PY=%ROOT%backend\.venv\Scripts\python.exe"
if not defined PY (
    echo [FAIL] No project venv found.
    echo        Run install.bat or install_amd.bat first.
    pause
    exit /b 1
)

echo Using python: %PY%
echo.

"%PY%" -X utf8 "%CHECKS%check_gpu_health.py" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [ OK ] Health check passed. Log saved in checks\ folder.
) else (
    echo [FAIL] Health check reported failures. See log above / checks\*.log
)
pause
exit /b %RC%
