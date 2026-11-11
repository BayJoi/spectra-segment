@echo off
setlocal enabledelayedexpansion

title Spectra Segment - AMD ROCm Installer

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "VENV_DIR=%AMD_DIR%\.venv"
set "UV_NO_CONFIG=1"
set "UV_EXE=%ROOT%tools\uv-amd\uv.exe"

set "TEMP=%ROOT%\.tmp"
set "TMP=%ROOT%\.tmp"
if not exist "%TEMP%" mkdir "%TEMP%" 2>nul

set "ROOT_LEN=0"
for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "$env:ROOT.Length"`) do set "ROOT_LEN=%%L"
set "UV_CACHE=%AMD_DIR%\.uv_cache_amd"
if !ROOT_LEN! gtr 50 set "UV_CACHE=%ROOT%.uv-amd"
if not exist "%UV_CACHE%" mkdir "%UV_CACHE%" 2>nul
set "UV_CACHE_DIR=%UV_CACHE%"
set "UV_PYTHON_CACHE_DIR=%UV_CACHE%"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                      Installer  (AMD)                      ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 8 ]  Preflight"
if !ROOT_LEN! gtr 50 (
    echo   [INFO]  Long project path - using the short project cache:
    echo           %UV_CACHE%
)
set "WEB_DIR=%AMD_DIR%\web"
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [INFO]  AMD venv already exists at %VENV_DIR%
    if exist "%WEB_DIR%\node_modules\vite\package.json" (
        echo           Frontend workspace ready.
    ) else (
        echo   [WARN]  Frontend workspace is incomplete - continuing setup.
    )
)

echo   [INFO]  ROCm runtime check...
echo           The ROCm PyTorch wheels bundle the HIP runtime and math
echo           libraries (rocm-sdk-core / rocm-sdk-libraries), so no system-wide
echo           ROCm install is required.

call :section "[ 2 / 8 ]  Python + uv"
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

call :section "[ 3 / 8 ]  Virtual environment"
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [INFO]  Virtual environment already exists - keeping it.
    goto :venv_ready
)
echo   [INFO]  Creating Python 3.12.13 virtual environment...
"%UV_EXE%" python install 3.12.13 --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  uv could not install Python 3.12.13
    echo           Try: "%UV_EXE%" python install 3.12.13
    pause
    exit /b 1
)
"%UV_EXE%" venv --clear --python 3.12.13 --seed "%VENV_DIR%" --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  uv venv creation failed
    pause
    exit /b 1
)
echo   [ OK ]  venv created at %VENV_DIR%

:venv_ready

call :section "[ 4 / 8 ]  GPU detection and per-card profile"
set "AMD_GFX="
set "HARDWARE_CFG=%AMD_DIR%\spectra_amd_hardware.cfg"
set "LAUNCHER_CFG=%AMD_DIR%\spectra_launcher.cfg"

echo   [INFO]  Reading AMD hardware and choosing the safest supported profile...
"%VENV_DIR%\Scripts\python.exe" "%AMD_DIR%\detect_amd_gpu.py" --write-config "%HARDWARE_CFG%" --write-launcher-config "%LAUNCHER_CFG%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  No Windows-supported AMD GPU was detected.
    echo           Set AMD_GFX_OVERRIDE to an exact target only if you know the card, e.g.:
    echo             set AMD_GFX_OVERRIDE=gfx1100 ^&^& install_amd.bat
    pause
    exit /b 1
)
if not exist "%HARDWARE_CFG%" (
    echo   [FAIL]  Hardware configuration was not written.
    pause
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%a in ("%HARDWARE_CFG%") do set "%%a=%%b"
if not defined AMD_GFX (
    echo   [FAIL]  Detection returned no gfx target.
    pause
    exit /b 1
)
echo   [ OK ]  !AMD_GPU_NAME! - !AMD_GFX! (!AMD_FAMILY!)
echo   [ OK ]  Support: !AMD_GPU_SUPPORT!   Recommended profile: !AMD_PROFILE!
echo   [INFO]  VRAM: !AMD_VRAM_MB! MB   FP16: !AMD_FP16!   Detector cap: !AMD_DETECT_MAX_SIDE! px   Matte cap: !SPECTRA_MATTE_INPUT_SIZE! px
if /i "!AMD_GPU_SUPPORT!"=="build" (
    echo   [WARN]  This target is build-only in TheRock's Windows matrix. It is not hardware-tested here.
)
echo   [INFO]  Settings are stored in %HARDWARE_CFG%

:gpu_detected
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://nightly.repo.amd.com/rocm/whl-next/"

call :section "[ 5 / 8 ]  ROCm PyTorch"

set "TORCH_PKG=torch[device-!AMD_GFX!]"
set "TORCHVISION_PKG=torchvision[device-!AMD_GFX!]"
set "TORCHAUDIO_PKG=torchaudio"

echo   [INFO]  Resolving and installing the newest ROCm nightly wheels...
echo           Existing torch, torchvision and torchaudio builds are replaced.
"%UV_EXE%" pip install --python "%VENV_DIR%" ^
    "!TORCH_PKG!" ^
    "!TORCHVISION_PKG!" ^
    "!TORCHAUDIO_PKG!" ^
    --index-url "%TORCH_INDEX_URL%" ^
    --cache-dir "%UV_CACHE%" ^
    --refresh ^
    --upgrade ^
    --prerelease=allow ^
    2>&1
if errorlevel 1 (
    echo   [WARN]  uv could not install ROCm PyTorch - retrying with pip...
    "%VENV_DIR%\Scripts\python.exe" -m pip --version >nul 2>&1
    if errorlevel 1 "%VENV_DIR%\Scripts\python.exe" -m ensurepip --upgrade >nul 2>&1
    if not exist "%AMD_DIR%\pip_cache" mkdir "%AMD_DIR%\pip_cache" 2>nul
    "%VENV_DIR%\Scripts\python.exe" -m pip install ^
        "!TORCH_PKG!" ^
        "!TORCHVISION_PKG!" ^
        "!TORCHAUDIO_PKG!" ^
        --index-url "%TORCH_INDEX_URL%" ^
        --cache-dir "%AMD_DIR%\pip_cache" ^
        --disable-pip-version-check ^
        --pre ^
        --upgrade ^
        2>&1
    if errorlevel 1 (
        echo   [FAIL]  ROCm PyTorch installation failed
        echo           Try: %VENV_DIR%\Scripts\python.exe -m pip install "!TORCH_PKG!" --index-url %TORCH_INDEX_URL%
        echo           If both attempts fail with a long-path error ^(WinError 206^), move this
        echo           folder somewhere shorter, e.g. C:\spectra-segment, and run the installer again.
        pause
        exit /b 1
    )
)
echo   [ OK ]  ROCm PyTorch installed

call :section "[ 6 / 8 ]  Python requirements"
echo   [INFO]  Installing packages from requirements.txt...
"%UV_EXE%" pip install --python "%VENV_DIR%" -r "%AMD_DIR%\requirements.txt" --cache-dir "%UV_CACHE%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  Some packages failed to install. Check requirements.txt.
    pause
    exit /b 1
)
echo   [ OK ]  Python requirements installed

echo   [INFO]  Checking package compatibility...
"%UV_EXE%" pip check --python "%VENV_DIR%" 2>&1
if errorlevel 1 (
    echo   [FAIL]  Installed packages have an incompatible dependency tree.
    echo           Re-run install_amd.bat after the nightly index settles.
    pause
    exit /b 1
)
echo   [ OK ]  Package dependency tree is compatible

echo   [INFO]  Installing SAM 3 CLIP text-encoder dependency...
set "CLIP_SRC=%ROOT%tools\clip_src"
if not exist "%CLIP_SRC%" mkdir "%CLIP_SRC%" 2>nul
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $wc=New-Object System.Net.WebClient; $zip=Join-Path '%TEMP%' 'clip.zip'; $wc.DownloadFile('https://codeload.github.com/ultralytics/CLIP/zip/refs/heads/main',$zip); Expand-Archive -Path $zip -DestinationPath '%CLIP_SRC%' -Force; Remove-Item $zip -ErrorAction SilentlyContinue" >nul 2>&1
set "CLIP_DL_ERR=!errorlevel!"
set "CLIP_DIR="
if exist "%CLIP_SRC%" for /d %%d in ("%CLIP_SRC%\*") do set "CLIP_DIR=%%d"
if "!CLIP_DL_ERR!"=="0" if defined CLIP_DIR (
    "%UV_EXE%" pip install --python "%VENV_DIR%" "!CLIP_DIR!" --cache-dir "%UV_CACHE%"
    if !errorlevel! neq 0 (
        if not exist "%AMD_DIR%\pip_cache" mkdir "%AMD_DIR%\pip_cache" 2>nul
        "%VENV_DIR%\Scripts\python.exe" -m pip install --cache-dir "%AMD_DIR%\pip_cache" "!CLIP_DIR!" 2>&1
        if !errorlevel! neq 0 (
            echo   [WARN]  Failed to install CLIP - SAM 3 mode will be unavailable.
        )
    )
) else (
    echo   [WARN]  Failed to download CLIP - SAM 3 mode will be unavailable.
)

call :section "[ 7 / 8 ]  Verification"
echo   [INFO]  Checking installation...
"%VENV_DIR%\Scripts\python.exe" -c "import sys, torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('HIP:', getattr(torch.version, 'hip', None)); print('GPU available:', torch.cuda.is_available()); print('Arch:', torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else 'none'); sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1
if errorlevel 1 (
    echo   [FAIL]  PyTorch cannot see the AMD GPU. Check the gfx target and the wheel index.
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
