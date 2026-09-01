@echo off
setlocal enabledelayedexpansion

title Spectra Segment - Installer (CPU)

set "ROOT=%~dp0"
set "TEMP=%ROOT%\.tmp"
set "TMP=%ROOT%\.tmp"
if not exist "%ROOT%\.tmp" mkdir "%ROOT%\.tmp" 2>nul
set "TOOLS_DIR=%ROOT%tools"
set "UV_DIR=%TOOLS_DIR%\uv-cpu"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_CACHE=%ROOT%backend\.uv\cache"
set "UV_PYTHON=%ROOT%tools\python"
set "VENV_DIR=%ROOT%backend\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "MODELS_DIR=%ROOT%backend\model_weights"
set "PY_VERSION=3.13.14"
rem uv version is resolved to the latest stable GitHub release at download time

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                       Installer  (CPU)                     ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 5 ]  Ensuring Bun"
set "BUN_DIR_OVERRIDE=%ROOT%tools\bun-cpu"
call "%ROOT%frontend\bun_env.cmd"
if !errorlevel! neq 0 (
    echo   [WARN]  Bun is not available - frontend install will be skipped.
    echo           Download Bun from https://bun.com/docs/install
)

call :section "[ 2 / 5 ]  Portable uv"
if exist "%UV_EXE%" (
    echo   [INFO]  uv already downloaded.
    goto :skip_uv_download
)

echo   [INFO]  Resolving latest stable uv release...
if not exist "%UV_DIR%" mkdir "%UV_DIR%"

powershell -NoProfile -Command ^
    "$ErrorActionPreference='Stop'; $rel = Invoke-RestMethod -UserAgent 'spectra-segment-installer' -Uri 'https://api.github.com/repos/astral-sh/uv/releases/latest'; $ver = $rel.tag_name.TrimStart('v'); $base = 'https://github.com/astral-sh/uv/releases/download/' + $ver; $wc = New-Object System.Net.WebClient; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $wc.DownloadFile($base + '/uv-x86_64-pc-windows-msvc.zip', '%UV_DIR%\uv.zip'); $wc.DownloadFile($base + '/uv-x86_64-pc-windows-msvc.zip.sha256', '%UV_DIR%\uv.zip.sha256'); $dl = (Get-FileHash -Algorithm SHA256 -Path '%UV_DIR%\uv.zip').Hash.ToLower(); $expect = ((Get-Content '%UV_DIR%\uv.zip.sha256' -Raw) -split '\s+')[0].ToLower(); if ($dl -ne $expect) { Remove-Item '%UV_DIR%\uv.zip' -ErrorAction SilentlyContinue; Remove-Item '%UV_DIR%\uv.zip.sha256' -ErrorAction SilentlyContinue; Write-Host ('[FAIL] uv checksum mismatch: ' + $ver); exit 1 }; Remove-Item '%UV_DIR%\uv.zip.sha256' -ErrorAction SilentlyContinue; Write-Host ('[ OK ] uv v' + $ver + ' downloaded and checksum verified')"
if !errorlevel! neq 0 (
    echo   [FAIL]  Failed to download uv.
    pause
    exit /b 1
)

echo   [INFO]  Extracting uv...
powershell -NoProfile -Command ^
    "Expand-Archive -Path '%UV_DIR%\uv.zip' -DestinationPath '%UV_DIR%' -Force"
if !errorlevel! neq 0 (
    echo   [FAIL]  Failed to extract uv.
    pause
    exit /b 1
)

if exist "%UV_DIR%\uv-x86_64-pc-windows-msvc\uv.exe" (
    move /y "%UV_DIR%\uv-x86_64-pc-windows-msvc\uv.exe" "%UV_DIR%\uv.exe" >nul 2>&1
    move /y "%UV_DIR%\uv-x86_64-pc-windows-msvc\uvw.exe" "%UV_DIR%\uvw.exe" >nul 2>&1
    move /y "%UV_DIR%\uv-x86_64-pc-windows-msvc\uvx.exe" "%UV_DIR%\uvx.exe" >nul 2>&1
    rmdir /s /q "%UV_DIR%\uv-x86_64-pc-windows-msvc" >nul 2>&1
)
del "%UV_DIR%\uv.zip" >nul 2>&1
echo   [ OK ]  uv ready.

:skip_uv_download

call :section "[ 3 / 5 ]  Configuring portable environment"
set "UV_CACHE_DIR=%UV_CACHE%"
set "UV_PYTHON_CACHE_DIR=%UV_CACHE%"
set "UV_NO_CONFIG=1"
if not exist "%UV_CACHE%" mkdir "%UV_CACHE%"
set "UV_PYTHON_INSTALL_DIR=%UV_PYTHON%"
set "UV_PYTHON_INSTALL_BIN=0"
if not exist "%UV_PYTHON%" mkdir "%UV_PYTHON%"
echo   [INFO]  Cache: %UV_CACHE%
echo   [INFO]  Python: %UV_PYTHON%

call :section "[ 4 / 5 ]  Virtual environment"
if exist "%VENV_PYTHON%" (
    echo   [INFO]  Virtual environment already exists.
    goto :skip_venv
)
echo   [INFO]  Creating virtual environment with Python %PY_VERSION%...
"%UV_EXE%" venv --python %PY_VERSION% --seed "%VENV_DIR%"
if !errorlevel! neq 0 (
    echo   [FAIL]  Failed to create virtual environment.
    pause
    exit /b 1
)
echo   [ OK ]  Virtual environment created.

:skip_venv

call :section "[ 5 / 5 ]  Detecting GPU and installing dependencies"
set "GPU_TYPE=cpu"
set "GPU_NAME="

for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
    set "GPU_TYPE=nvidia"
    set "GPU_NAME=%%G"
)

if "!GPU_TYPE!"=="nvidia" (
    echo   [INFO]  NVIDIA GPU detected: !GPU_NAME!
) else (
    echo   [INFO]  No NVIDIA GPU detected - using CPU.
)

if "!GPU_TYPE!"=="nvidia" (
    echo   [INFO]  Installing CUDA 13.0 PyTorch...
    "%UV_EXE%" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --python "%VENV_PYTHON%" --cache-dir "%UV_CACHE%"
    if !errorlevel! neq 0 (
        echo   [WARN]  CUDA install failed, falling back to CPU...
        "%UV_EXE%" pip install torch torchvision --python "%VENV_PYTHON%" --cache-dir "%UV_CACHE%"
    )
) else (
    echo   [INFO]  Installing CPU PyTorch...
    "%UV_EXE%" pip install torch torchvision --python "%VENV_PYTHON%" --cache-dir "%UV_CACHE%"
)

echo   [INFO]  Installing Python packages...
if exist "%ROOT%backend\requirements.txt" (
    "%UV_EXE%" pip install -r "%ROOT%backend\requirements.txt" --python "%VENV_PYTHON%" --cache-dir "%UV_CACHE%"
    if !errorlevel! neq 0 (
        echo   [FAIL]  Failed to install Python packages.
        pause
        exit /b 1
    )
)

echo   [INFO]  Installing SAM 3 CLIP text-encoder dependency...
set "CLIP_SRC=%ROOT%tools\clip_src"
if not exist "%CLIP_SRC%" mkdir "%CLIP_SRC%" 2>nul
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $wc=New-Object System.Net.WebClient; $zip=Join-Path '%TEMP%' 'clip.zip'; $wc.DownloadFile('https://codeload.github.com/ultralytics/CLIP/zip/refs/heads/main',$zip); Expand-Archive -Path $zip -DestinationPath '%CLIP_SRC%' -Force; Remove-Item $zip -ErrorAction SilentlyContinue" >nul 2>&1
set "CLIP_DL_ERR=!errorlevel!"
set "CLIP_DIR="
if exist "%CLIP_SRC%" for /d %%d in ("%CLIP_SRC%\*") do set "CLIP_DIR=%%d"
if "!CLIP_DL_ERR!"=="0" if defined CLIP_DIR (
    "%UV_EXE%" pip install --python "%VENV_PYTHON%" "!CLIP_DIR!" --cache-dir "%UV_CACHE%"
    if !errorlevel! neq 0 (
        echo   [WARN]  Failed to install CLIP - SAM 3 mode will be unavailable.
    )
) else (
    echo   [WARN]  Failed to download CLIP - SAM 3 mode will be unavailable.
)

set "WEB_DIR=%ROOT%backend\web"
echo   [INFO]  Preparing frontend workspace at backend\web ...
call "%ROOT%frontend\use_web.bat" "%WEB_DIR%"
if errorlevel 1 (
    echo   [FAIL]  Could not set up frontend workspace.
    pause
    exit /b 1
)
if defined BUN_EXE (
    echo   [INFO]  Installing Bun dependencies...
    if exist "%WEB_DIR%\package.json" (
        pushd "%WEB_DIR%"
        "!BUN_EXE!" install
        set "BUN_INSTALL_ERR=!errorlevel!"
        popd
        if !BUN_INSTALL_ERR! neq 0 (
            echo   [FAIL]  Failed to install Bun dependencies.
            pause
            exit /b 1
        )
        echo   [ OK ]  Bun dependencies installed.
    )
) else (
    echo   [WARN]  Skipping frontend install - Bun not available.
    echo           Download Bun from https://bun.com/docs/install
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
cd /d "%ROOT%"

echo(
echo  +------------------------------------------------------------+
echo  ^|  Installation complete                                     ^|
echo  ^|  Everything is portable in this folder.                    ^|
echo  ^|  Run start.bat to launch the application.                  ^|
echo  +------------------------------------------------------------+
echo(
echo   [INFO]  Layout:
echo   [INFO]    tools\uv-cpu\      - uv binary (CPU)
echo   [INFO]    tools\bun-cpu\     - Bun (stable, latest)
echo   [INFO]    tools\python\      - Python %PY_VERSION%
echo   [INFO]    backend\.venv\     - Virtual environment
echo   [INFO]    backend\.uv\cache\ - Package cache
echo   [INFO]    backend\web\       - Frontend workspace ^(isolated^)
echo(
pause

endlocal
exit /b 0

:section
echo(
echo   ================================================================
echo    %~1
echo   ----------------------------------------------------------------
echo(
exit /b 0
