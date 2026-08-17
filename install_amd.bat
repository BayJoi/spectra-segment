@echo off
setlocal enabledelayedexpansion

title Spectra Segment - AMD ROCm Installer

set "ROOT=%~dp0"
set "TEMP=%ROOT%\.tmp"
set "TMP=%ROOT%\.tmp"
if not exist "%ROOT%\.tmp" mkdir "%ROOT%\.tmp" 2>nul
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "VENV_DIR=%AMD_DIR%\.venv"
set "UV_CACHE=%AMD_DIR%\.uv_cache_amd"
set "UV_CACHE_DIR=%UV_CACHE%"
set "UV_PYTHON_CACHE_DIR=%UV_CACHE%"
set "UV_NO_CONFIG=1"
set "UV_EXE=%ROOT%tools\uv-amd\uv.exe"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                      Installer  (AMD)                      ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 8 ]  Preflight"
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [INFO]  AMD venv already exists at %VENV_DIR%
    echo           To reinstall, first run: cleanup_amd.bat
    goto :done
)

echo   [INFO]  AMD ROCm SDK check...
where rocm-sdk >nul 2>&1
if errorlevel 1 (
    echo   [INFO]  rocm-sdk CLI not found - optional.
    echo           The rocm-sdk-devel pip package provides portable HIP DLLs.
    echo           System ROCm SDK is only needed for advanced ROCm development.
)

call :section "[ 2 / 8 ]  GPU detection"
set "AMD_GFX="

if defined AMD_GFX_OVERRIDE set "AMD_GFX=%AMD_GFX_OVERRIDE%"& echo   [INFO]  Using user-specified arch: !AMD_GFX!& goto :gpu_detected

py -3 "%AMD_DIR%\detect_amd_gpu.py" >"%AMD_DIR%\.gpu_arch.txt" 2>nul
if exist "%AMD_DIR%\.gpu_arch.txt" for /f "usebackq delims=" %%a in ("%AMD_DIR%\.gpu_arch.txt") do if not defined AMD_GFX set "AMD_GFX=%%a"
del "%AMD_DIR%\.gpu_arch.txt" 2>nul
if defined AMD_GFX goto :gpu_detected

python "%AMD_DIR%\detect_amd_gpu.py" >"%AMD_DIR%\.gpu_arch.txt" 2>nul
if exist "%AMD_DIR%\.gpu_arch.txt" for /f "usebackq delims=" %%a in ("%AMD_DIR%\.gpu_arch.txt") do if not defined AMD_GFX set "AMD_GFX=%%a"
del "%AMD_DIR%\.gpu_arch.txt" 2>nul
if defined AMD_GFX goto :gpu_detected

set "DEV_ID="
powershell -NoProfile -Command "try{((Get-WmiObject Win32_VideoController|?{$_.PNPDeviceID -match 'VEN_1002'}).PNPDeviceID -replace '.*DEV_([0-9A-F]{4}).*','$1')}catch{}" >"%AMD_DIR%\.gpu_id.txt" 2>nul
if exist "%AMD_DIR%\.gpu_id.txt" for /f "usebackq delims=" %%a in ("%AMD_DIR%\.gpu_id.txt") do set "DEV_ID=%%a"
del "%AMD_DIR%\.gpu_id.txt" 2>nul
if defined DEV_ID if /i "!DEV_ID!"=="744c" set "AMD_GFX=gfx1100"
if defined DEV_ID if /i "!DEV_ID!"=="73bf" set "AMD_GFX=gfx1030"
if defined DEV_ID if /i "!DEV_ID!"=="73ff" set "AMD_GFX=gfx1032"
if defined DEV_ID if /i "!DEV_ID!"=="731f" set "AMD_GFX=gfx1010"
if defined DEV_ID if /i "!DEV_ID!"=="1681" set "AMD_GFX=gfx1035"
if defined DEV_ID if /i "!DEV_ID!"=="163f" set "AMD_GFX=gfx1033"
if defined DEV_ID if /i "!DEV_ID!"=="15bf" set "AMD_GFX=gfx1103"
if defined DEV_ID if /i "!DEV_ID!"=="150e" set "AMD_GFX=gfx1150"
if defined DEV_ID if /i "!DEV_ID!"=="7590" set "AMD_GFX=gfx1200"
if defined DEV_ID if /i "!DEV_ID!"=="7550" set "AMD_GFX=gfx1201"
if defined AMD_GFX echo   [INFO]  Matched device maps to !AMD_GFX!& goto :gpu_detected

echo   [FAIL]  GPU detection failed.
echo           Set AMD_GFX_OVERRIDE=gfxXXXX to skip detection, e.g.:
echo             set AMD_GFX_OVERRIDE=gfx1100 ^&^& install_amd.bat
echo           Common values: gfx1100 (RDNA3), gfx1030 (RDNA2), gfx1010 (RDNA1)
pause
exit /b 1

:gpu_detected
echo   [ OK ]  AMD GPU arch: !AMD_GFX!

set "TORCH_INDEX_URL=https://rocm.nightlies.amd.com/whl-multi-arch/"
set "ROCM_SDK_PKG=rocm-sdk-devel"
if "!AMD_GFX!"=="gfx942" (
    set "TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/"
    set "ROCM_SDK_PKG=rocm[devel,libraries]"
)
if "!AMD_GFX!"=="gfx950" (
    set "TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/"
    set "ROCM_SDK_PKG=rocm[devel,libraries]"
)

call :section "[ 3 / 8 ]  Python + uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%tools\python"
set "UV_PYTHON_INSTALL_BIN=0"
if exist "%UV_EXE%" (
    echo   [ OK ]  uv found ^(project tools\uv-amd^)
) else (
    echo   [INFO]  Resolving latest stable uv release...
    if not exist "%ROOT%tools\uv-amd" mkdir "%ROOT%tools\uv-amd" 2>nul
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $rel = Invoke-RestMethod -UserAgent 'spectra-segment-installer' -Uri 'https://api.github.com/repos/astral-sh/uv/releases/latest'; $ver = $rel.tag_name.TrimStart('v'); $base = 'https://github.com/astral-sh/uv/releases/download/' + $ver; $wc = New-Object System.Net.WebClient; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $wc.DownloadFile($base + '/uv-x86_64-pc-windows-msvc.zip', '%ROOT%tools\uv-amd\uv.zip'); $wc.DownloadFile($base + '/uv-x86_64-pc-windows-msvc.zip.sha256', '%ROOT%tools\uv-amd\uv.zip.sha256'); $dl = (Get-FileHash -Algorithm SHA256 -Path '%ROOT%tools\uv-amd\uv.zip').Hash.ToLower(); $expect = ((Get-Content '%ROOT%tools\uv-amd\uv.zip.sha256' -Raw) -split '\s+')[0].ToLower(); if ($dl -ne $expect) { Remove-Item '%ROOT%tools\uv-amd\uv.zip' -ErrorAction SilentlyContinue; Remove-Item '%ROOT%tools\uv-amd\uv.zip.sha256' -ErrorAction SilentlyContinue; Write-Host ('[FAIL] uv checksum mismatch: ' + $ver); exit 1 }; Remove-Item '%ROOT%tools\uv-amd\uv.zip.sha256' -ErrorAction SilentlyContinue; Write-Host ('[ OK ] uv v' + $ver + ' downloaded and checksum verified')"
    if errorlevel 1 (
        echo   [FAIL]  uv download or checksum verification failed. Install manually: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    powershell -NoProfile -Command "Expand-Archive -Path '%ROOT%tools\uv-amd\uv.zip' -DestinationPath '%ROOT%tools\uv-amd' -Force" >nul
    del "%ROOT%tools\uv-amd\uv.zip" 2>nul
    if exist "%ROOT%tools\uv-amd\uv-x86_64-pc-windows-msvc\uv.exe" (
        move /y "%ROOT%tools\uv-amd\uv-x86_64-pc-windows-msvc\uv.exe" "%ROOT%tools\uv-amd\uv.exe" >nul 2>&1
        move /y "%ROOT%tools\uv-amd\uv-x86_64-pc-windows-msvc\uvw.exe" "%ROOT%tools\uv-amd\uvw.exe" >nul 2>&1
        move /y "%ROOT%tools\uv-amd\uv-x86_64-pc-windows-msvc\uvx.exe" "%ROOT%tools\uv-amd\uvx.exe" >nul 2>&1
        rmdir /s /q "%ROOT%tools\uv-amd\uv-x86_64-pc-windows-msvc" >nul 2>&1
    )
    if not exist "%UV_EXE%" (
        echo   [FAIL]  uv extraction failed
        pause
        exit /b 1
    )
    echo   [ OK ]  uv downloaded to %ROOT%tools\uv-amd
)

call :section "[ 4 / 8 ]  Virtual environment"
echo   [INFO]  Creating Python 3.12.13 virtual environment...
"%UV_EXE%" python install 3.12.13 --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  uv could not install Python 3.12.13
    echo           Try: "%UV_EXE%" python install 3.12.13
    pause
    exit /b 1
)
"%UV_EXE%" venv --python 3.12.13 "%VENV_DIR%" --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  uv venv creation failed
    pause
    exit /b 1
)
echo   [ OK ]  venv created at %VENV_DIR%

call :section "[ 5 / 8 ]  ROCm SDK + PyTorch"
echo   [INFO]  Installing ROCm SDK (%ROCM_SDK_PKG%)...
"%UV_EXE%" pip install --python "%VENV_DIR%" "%ROCM_SDK_PKG%" --index-url "%TORCH_INDEX_URL%" --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [WARN]  ROCm SDK pip package install failed - continuing without it.
)

set "USE_DEVICE_EXTRAS=1"
if "!AMD_GFX!"=="gfx942" set "USE_DEVICE_EXTRAS="
if "!AMD_GFX!"=="gfx950" set "USE_DEVICE_EXTRAS="

set "TORCH_PKG=torch"
set "TORCHVISION_PKG=torchvision"
set "TORCHAUDIO_PKG=torchaudio"
if defined USE_DEVICE_EXTRAS (
    set "TORCH_PKG=torch[device-!AMD_GFX!]"
    set "TORCHVISION_PKG=torchvision[device-!AMD_GFX!]"
)

echo   [INFO]  Installing ROCm PyTorch (this may take a while)...
"%UV_EXE%" pip install --python "%VENV_DIR%" ^
    "!TORCH_PKG!" ^
    "!TORCHVISION_PKG!" ^
    "!TORCHAUDIO_PKG!" ^
    --index-url "%TORCH_INDEX_URL%" ^
    --cache-dir "%UV_CACHE%" ^
    2>&1
if errorlevel 1 (
    echo   [FAIL]  ROCm PyTorch installation failed
    echo           Try: %VENV_DIR%\Scripts\python.exe -m pip install "!TORCH_PKG!" --index-url %TORCH_INDEX_URL%
    pause
    exit /b 1
)
echo   [ OK ]  ROCm PyTorch installed

call :section "[ 6 / 8 ]  Python requirements"
echo   [INFO]  Installing packages from requirements.txt...
"%UV_EXE%" pip install --python "%VENV_DIR%" -r "%AMD_DIR%\requirements.txt" --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [WARN]  Some packages failed to install. Check requirements.txt.
)
echo   [ OK ]  Python requirements installed

call :section "[ 7 / 8 ]  ROCm runtime + verification"
echo   [INFO]  Initializing ROCm SDK...
where rocm-sdk >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%a in ('rocm-sdk init 2^>nul') do set "ROCM_INIT=%%a"
    if defined ROCM_INIT if exist "!ROCM_INIT!" call "!ROCM_INIT!"
    for /f "delims=" %%a in ('rocm-sdk path --root 2^>nul') do set "HIP_PATH=%%a"
    if defined HIP_PATH (
        set "ROCM_PATH=!HIP_PATH!"
        echo   [ OK ]  HIP runtime initialized  ^(HIP_PATH=!HIP_PATH!^)
    ) else (
        echo   [WARN]  Could not determine ROCm SDK root path
    )
) else (
    echo   [INFO]  rocm-sdk CLI not used - portable HIP DLLs from pip package handle runtime
)

echo   [INFO]  Checking installation...
"%VENV_DIR%\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); v = getattr(torch.version, 'hip', None); print('HIP:', v)" 2>&1
if errorlevel 1 (
    echo   [FAIL]  PyTorch verification failed
    pause
    exit /b 1
)
echo   [ OK ]  Installation verified

call :section "[ 8 / 8 ]  Frontend"
set "WEB_DIR=%AMD_DIR%\web"
if exist "%ROOT%\frontend\package.json" (
    echo   [INFO]  Preparing frontend workspace at backend_amd_gpu\web ...
    call "%ROOT%frontend\use_web.bat" "%WEB_DIR%"
    if errorlevel 1 (
        echo   [FAIL]  Could not set up frontend workspace.
        pause
        exit /b 1
    )
    if not exist "%WEB_DIR%\node_modules\vite\package.json" (
        echo   [INFO]  Installing frontend dependencies ^(bun^)...
        set "BUN_DIR_OVERRIDE=%ROOT%tools\bun-amd"
        call "%ROOT%frontend\bun_env.cmd"
        if errorlevel 1 (
            echo   [WARN]  Bun not available - skipping frontend install.
            echo           Download Bun from https://bun.com/docs/install
        ) else (
            pushd "%WEB_DIR%"
            "!BUN_EXE!" install 2>&1
            if errorlevel 1 (
                echo   [FAIL]  bun install failed
                popd
            ) else (
                echo   [ OK ]  Frontend dependencies installed
                popd
                echo   [INFO]  Running bun run build...
                pushd "%ROOT%\frontend"
                "!BUN_EXE!" run build 2>&1
                if errorlevel 1 (
                    echo   [FAIL]  Frontend build failed
                ) else (
                    echo   [ OK ]  Frontend build complete
                )
                popd
            )
        )
    ) else (
        echo   [INFO]  node_modules already installed
    )
)

:done
echo(
echo  +------------------------------------------------------------+
echo  ^|  AMD setup complete.                                       ^|
echo  ^|  To start: start_amd.bat                                   ^|
echo  ^|  To clean: cleanup_amd.bat                                 ^|
echo  +------------------------------------------------------------+
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
