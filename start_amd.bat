@echo off
setlocal enabledelayedexpansion

title Spectra Segment - AMD ROCm Launcher

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "VENV_PYTHON=%AMD_DIR%\.venv\Scripts\python.exe"
set "MODEL_WEIGHTS=%AMD_DIR%\model_weights"

set "RC_ARG="
set "CLI_MODE="
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="-reconfigure" set "RC_ARG=-Reconfigure"
if /i "%~1"=="--reconfigure" set "RC_ARG=-Reconfigure"
if /i "%~1"=="/reconfigure" set "RC_ARG=-Reconfigure"
if /i "%~1"=="-cpu" set "CLI_MODE=-ForceCpu"
if /i "%~1"=="--cpu" set "CLI_MODE=-ForceCpu"
if /i "%~1"=="/cpu" set "CLI_MODE=-ForceCpu"
if /i "%~1"=="-gpu" set "CLI_MODE=-ForceGpu"
if /i "%~1"=="--gpu" set "CLI_MODE=-ForceGpu"
if /i "%~1"=="/gpu" set "CLI_MODE=-ForceGpu"
shift
goto :parse_args
:args_done

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                    Launcher (AMD ROCm)                     ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 6 ]  Environment"
if not exist "%VENV_PYTHON%" (
    echo   [FAIL]  AMD venv not found at %AMD_DIR%\.venv
    echo           Run install_amd.bat first.
    pause
    exit /b 1
)
"%VENV_PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL]  AMD Python environment is present but cannot start.
    pause
    exit /b 1
)
echo   [ OK ]  AMD Python environment is ready.

call :section "[ 2 / 6 ]  GPU"
set "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
set "MIOPEN_DEBUG_DISABLE_FIND_DB=0"
set "PYTORCH_TUNABLEOP_CACHE_DIR=%MODEL_WEIGHTS%\tunableop_cache"
set "HSA_ENABLE_SDMA=0"
if not defined TORCH_BLAS_PREFER_HIPBLASLT set "TORCH_BLAS_PREFER_HIPBLASLT=1"
if not defined DISABLE_ADDMM_CUDA_LT set "DISABLE_ADDMM_CUDA_LT=1"
set "MIMALLOC_PURGE_DELAY=0"

set "HARDWARE_CFG=%AMD_DIR%\spectra_amd_hardware.cfg"
set "LAUNCHER_CFG=%AMD_DIR%\spectra_launcher.cfg"
if not exist "%MODEL_WEIGHTS%" mkdir "%MODEL_WEIGHTS%" 2>nul
if defined AMD_GFX_OVERRIDE (
    "%VENV_PYTHON%" "%AMD_DIR%\detect_amd_gpu.py" --gfx "%AMD_GFX_OVERRIDE%" --write-config "%HARDWARE_CFG%" --write-launcher-config "%LAUNCHER_CFG%" >nul 2>&1
)
if not exist "%HARDWARE_CFG%" (
    "%VENV_PYTHON%" "%AMD_DIR%\detect_amd_gpu.py" --write-config "%HARDWARE_CFG%" --write-launcher-config "%LAUNCHER_CFG%" >nul 2>&1
)
set "AMD_GFX="
set "AMD_VRAM=0"
if exist "%HARDWARE_CFG%" for /f "usebackq tokens=1,* delims==" %%a in ("%HARDWARE_CFG%") do set "%%a=%%b"
if defined AMD_VRAM_MB set "AMD_VRAM=%AMD_VRAM_MB%"
if not defined AMD_GFX goto :no_amd_gpu

echo   [INFO]  GPU: !AMD_GPU_NAME! - !AMD_GFX! - !AMD_FAMILY!
echo   [INFO]  Profile: !AMD_PROFILE! - detector !AMD_DETECT_MAX_SIDE! px - matte !SPECTRA_MATTE_INPUT_SIZE! px
if /i not "!AMD_GPU_SUPPORT!"=="release" echo   [WARN]  GPU support: !AMD_GPU_SUPPORT!
if defined AMD_AOTRITON if "!AMD_AOTRITON!"=="1" set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
if defined AMD_QUANTIZE (
    if not defined SAM2_QUANTIZE set "SAM2_QUANTIZE=!AMD_QUANTIZE!"
    if not defined SAM3_QUANTIZE set "SAM3_QUANTIZE=!AMD_QUANTIZE!"
)
if defined AMD_FP16 if "!AMD_FP16!"=="0" set "SPECTRA_DETECTOR_FP16=0"
if defined AMD_FP16 if "!AMD_FP16!"=="0" set "SPECTRA_MATTE_FP16=0"
if defined AMD_DETECT_MAX_SIDE set "SPECTRA_DETECT_MAX_SIDE=!AMD_DETECT_MAX_SIDE!"
if defined AMD_HSA_OVERRIDE if not defined HSA_OVERRIDE_GFX_VERSION (
    set "HSA_OVERRIDE_GFX_VERSION=!AMD_HSA_OVERRIDE!"
    echo   [INFO]  HSA override: !HSA_OVERRIDE_GFX_VERSION!
)
goto :gpu_configured

:no_amd_gpu
echo   [WARN]  No supported AMD GPU detected - using CPU.

:gpu_configured
call :section "[ 3 / 6 ]  Memory"
set "FORCE_CPU_ARG="
set "FORCE_GPU_ARG="
set "STORED_MODE="
if exist "%LAUNCHER_CFG%" for /f "usebackq tokens=1,* delims==" %%a in ("%LAUNCHER_CFG%") do if /i "%%a"=="vram_mode" set "STORED_MODE=%%b"
if defined RC_ARG (
    if defined AMD_GFX (
        choice /c GC /n /m "  [INFO]  AMD GPU detected - use (G)PU or (C)PU? "
        if errorlevel 2 (set "FORCE_CPU_ARG=-ForceCpu") else (set "FORCE_GPU_ARG=-ForceGpu")
    ) else (
        set "FORCE_CPU_ARG=-ForceCpu"
    )
) else if /i "!STORED_MODE!"=="cpu" (
    set "FORCE_CPU_ARG=-ForceCpu"
) else if defined AMD_GFX (
    set "FORCE_GPU_ARG=-ForceGpu"
) else (
    set "FORCE_CPU_ARG=-ForceCpu"
)
if not exist "%ROOT%launcher" mkdir "%ROOT%launcher" 2>nul
if defined CLI_MODE (
    set "FORCE_CPU_ARG="
    set "FORCE_GPU_ARG="
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%launcher\setup_memory.ps1" -ConfigPath "%AMD_DIR%\spectra_launcher.cfg" -Backend AMD -DetectedVramMb "!AMD_VRAM!" %RC_ARG% %FORCE_CPU_ARG% %FORCE_GPU_ARG% %CLI_MODE%
if errorlevel 1 (
    echo   [FAIL]  Memory profile setup failed.
    pause
    exit /b 1
)
if exist "%AMD_DIR%\spectra_launcher.cfg.env" (
    for /f "usebackq delims=" %%a in ("%AMD_DIR%\spectra_launcher.cfg.env") do set "%%a"
    del "%AMD_DIR%\spectra_launcher.cfg.env" 2>nul
)
call :section "[ 4 / 6 ]  Caches"
if not exist "%MODEL_WEIGHTS%" mkdir "%MODEL_WEIGHTS%" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%launcher\prepare_caches.ps1" -ModelWeights "%MODEL_WEIGHTS%"
if errorlevel 1 (
    echo   [FAIL]  Could not prepare model caches.
    pause
    exit /b 1
)

set "HF_HOME=%MODEL_WEIGHTS%\hf_cache"
set "HF_HUB_CACHE=%MODEL_WEIGHTS%\hf_cache\hub"
set "HUGGINGFACE_HUB_CACHE=%MODEL_WEIGHTS%\hf_cache\hub"
set "TRANSFORMERS_CACHE=%MODEL_WEIGHTS%\hf_cache\hub"
set "HF_MODULES_CACHE=%MODEL_WEIGHTS%\hf_cache\modules"
set "HF_DATASETS_CACHE=%MODEL_WEIGHTS%\hf_datasets"
set "SENTENCE_TRANSFORMERS_HOME=%MODEL_WEIGHTS%\sentence_transformers"
set "TORCH_HOME=%MODEL_WEIGHTS%\torch_cache"
set "TRITON_CACHE_DIR=%MODEL_WEIGHTS%\triton"
set "NUMBA_CACHE_DIR=%MODEL_WEIGHTS%\numba"
set "ULTRALYTICS_HOME=%MODEL_WEIGHTS%"
set "YOLO_CONFIG_DIR=%MODEL_WEIGHTS%"
set "MPLCONFIGDIR=%MODEL_WEIGHTS%\matplotlib"
set "XDG_CACHE_HOME=%MODEL_WEIGHTS%\cache"
set "XDG_CONFIG_HOME=%MODEL_WEIGHTS%\config"
set "XDG_DATA_HOME=%MODEL_WEIGHTS%\data"
set "TMPDIR=%MODEL_WEIGHTS%\tmp"
set "TEMP=%MODEL_WEIGHTS%\tmp"
set "TMP=%MODEL_WEIGHTS%\tmp"
set "MIOPEN_USER_DB_PATH=%MODEL_WEIGHTS%\miopen"
set "MIOPEN_CUSTOM_CACHE_DIR=%MODEL_WEIGHTS%\miopen\cache"
set "PYTORCH_TUNABLEOP_CACHE_DIR=%MODEL_WEIGHTS%\tunableop_cache"

call :section "[ 5 / 6 ]  Backend"
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and $p.CommandLine -like '*main:app*') { Stop-Process -Id $_ -Force } }; exit 0"
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and ($p.CommandLine -like '*run dev*' -or $p.CommandLine -like '*vite*')) { Stop-Process -Id $_ -Force } }; exit 0"

if not exist "%AMD_DIR%\logs" mkdir "%AMD_DIR%\logs"
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "LAUNCH_STAMP=%%t"
if not defined LAUNCH_STAMP set "LAUNCH_STAMP=%RANDOM%"
set "SPECTRA_LOG_FILE=%AMD_DIR%\logs\backend_%LAUNCH_STAMP%.log"
set "BACKEND_LAUNCH_LOG=%AMD_DIR%\logs\launcher_%LAUNCH_STAMP%.log"
echo   [INFO]  Starting backend (port 8000)...
start /B "Spectra-Segment-Backend-AMD" cmd /c ""%AMD_DIR%\run.cmd" > "%BACKEND_LAUNCH_LOG%" 2>&1"

echo   [WAIT]  Waiting for backend at http://127.0.0.1:8000/health ...
set "CURL_OK="
where curl >nul 2>&1
if not errorlevel 1 set "CURL_OK=1"
set "RETRIES=0"
:health_loop
timeout /t 2 /nobreak >nul
set /a RETRIES+=1
if !RETRIES! geq 30 (
    echo   [FAIL]  Backend did not start ^(waited up to 60 seconds^)
    if exist "%BACKEND_LAUNCH_LOG%" (
        echo   --- %BACKEND_LAUNCH_LOG% ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%BACKEND_LAUNCH_LOG%' -Tail 15"
    )
    if exist "%SPECTRA_LOG_FILE%" (
        echo   --- %SPECTRA_LOG_FILE% ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%SPECTRA_LOG_FILE%' -Tail 15"
    )
    echo           Check: %SPECTRA_LOG_FILE%
    pause
    exit /b 1
)
if defined CURL_OK (
    curl -s -o nul --max-time 2 http://127.0.0.1:8000/health >nul 2>&1
    if not errorlevel 1 goto :backend_ready
) else (
    for /f "tokens=*" %%a in ('powershell -NoProfile -Command "try { $null = (New-Object System.Net.WebClient).DownloadString('http://127.0.0.1:8000/health'); 'ok' } catch { 'no' }" 2^>nul') do (
        if "%%a"=="ok" goto :backend_ready
    )
)
goto :health_loop

:backend_ready
echo   [ OK ]  Backend is ready

call :section "[ 6 / 6 ]  Frontend"
set "WEB_DIR=%AMD_DIR%\web"
set "BUN_DIR_OVERRIDE=%ROOT%tools\bun-amd"
call "%ROOT%frontend\bun_env.cmd"
if errorlevel 1 (
    echo   [WARN]  Bun not available - download failed.
    echo           Download Bun from https://bun.com/docs/install
    echo   [WARN]  Frontend not started
    goto :amd_done
)
call "%ROOT%frontend\use_web.bat" "%WEB_DIR%"
if errorlevel 1 (
    echo   [FAIL]  Could not set up frontend workspace at %WEB_DIR%
    echo           Backend is running - use stop_amd.bat to stop it.
    goto :amd_done
)
if not exist "%WEB_DIR%\node_modules\vite\package.json" (
    echo   [INFO]  Installing frontend dependencies ^(bun install^)...
    pushd "%WEB_DIR%"
    call "%BUN_EXE%" install
    popd
)
echo   [INFO]  Starting frontend via project Bun ^(port 3000^)...
pushd "%ROOT%\frontend"
start /B "Spectra-Segment-Frontend" cmd /c ""!BUN_EXE!" "%ROOT%frontend\node_modules\vite\bin\vite.js" --host 127.0.0.1 --port 3000"
popd

:amd_done

echo(
echo  +------------------------------------------------------------+
echo  ^|  Spectra Segment (AMD) is starting up.                     ^|
echo  ^|  Backend:  http://127.0.0.1:8000                           ^|
echo  ^|  Frontend: http://127.0.0.1:3000                           ^|
echo  +------------------------------------------------------------+
echo(
echo   [INFO]  Logs: %SPECTRA_LOG_FILE%
echo(
endlocal
exit /b 0

:section
echo(
echo   %~1
exit /b 0
