@echo off
setlocal enabledelayedexpansion

title Spectra Segment - AMD ROCm Launcher

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "VENV_PYTHON=%AMD_DIR%\.venv\Scripts\python.exe"
set "MODEL_WEIGHTS=%AMD_DIR%\model_weights"

set "RC_ARG="
if /i "%~1"=="-reconfigure" set "RC_ARG=-Reconfigure"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                  Launcher (AMD GPU / CPU)                  ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 7 ]  Preflight"
if not exist "%VENV_PYTHON%" (
    echo   [FAIL]  AMD venv not found at %AMD_DIR%\.venv
    echo           Run install_amd.bat first.
    pause
    exit /b 1
)

call :section "[ 2 / 7 ]  Configuring AMD GPU"
set "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
set "MIOPEN_DEBUG_DISABLE_FIND_DB=0"
set "PYTORCH_TUNABLEOP_CACHE_DIR=%MODEL_WEIGHTS%\tunableop_cache"
set "HSA_ENABLE_SDMA=0"
set "MIMALLOC_PURGE_DELAY=0"
if defined AMD_GFX_OVERRIDE (
    set "AMD_GFX=%AMD_GFX_OVERRIDE%"
    echo   [INFO]  AMD GPU arch: !AMD_GFX!  ^(from AMD_GFX_OVERRIDE^)
    goto :gpu_configured
)

if not exist "%MODEL_WEIGHTS%" mkdir "%MODEL_WEIGHTS%"
set "AMD_VRAM=0"
"%VENV_PYTHON%" "%AMD_DIR%\detect_amd_gpu.py" >"%MODEL_WEIGHTS%\.gpu_arch.txt" 2>nul
if exist "%MODEL_WEIGHTS%\.gpu_arch.txt" (
    set "AMD_GFX="
    set "AMD_VRAM=0"
    set "LN=0"
    for /f "usebackq delims=" %%a in ("%MODEL_WEIGHTS%\.gpu_arch.txt") do (
        set /a LN+=1
        if !LN!==1 (set "AMD_GFX=%%a") else if !LN!==2 (set "AMD_VRAM=%%a") else (if not "%%a"=="" set "AMD_HSA_OVERRIDE_AUTO=%%a")
    )
)
del "%MODEL_WEIGHTS%\.gpu_arch.txt" 2>nul
if defined AMD_GFX (
    echo   [INFO]  AMD GPU arch: !AMD_GFX!
    set "IS_LEGACY_SDP=1"
    if "!AMD_GFX:~0,6!"=="gfx110" set "IS_LEGACY_SDP="
    if "!AMD_GFX:~0,6!"=="gfx115" set "IS_LEGACY_SDP="
    if "!AMD_GFX:~0,5!"=="gfx12" set "IS_LEGACY_SDP="
    if defined IS_LEGACY_SDP (
        echo   [INFO]  RDNA2/older - Flash/MemEff SDP unsupported, using Math SDP
        set "TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0"
        set "TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0"
        set "TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED=1"
    ) else (
        set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
    )
) else (
    echo   [INFO]  AMD GPU not detected - starting anyway ^(CPU fallback^)
)

:gpu_configured

if defined AMD_GFX (
    if not defined HSA_OVERRIDE_GFX_VERSION (
        if defined AMD_HSA_OVERRIDE (
            set "HSA_OVERRIDE_GFX_VERSION=!AMD_HSA_OVERRIDE!"
            echo   [INFO]  HSA_OVERRIDE_GFX_VERSION=!HSA_OVERRIDE_GFX_VERSION! ^(user AMD_HSA_OVERRIDE^)
        )
        if not defined HSA_OVERRIDE_GFX_VERSION if defined AMD_HSA_OVERRIDE_AUTO (
            set "HSA_OVERRIDE_GFX_VERSION=!AMD_HSA_OVERRIDE_AUTO!"
            echo   [INFO]  HSA_OVERRIDE_GFX_VERSION=!HSA_OVERRIDE_GFX_VERSION! ^(wheel kernel family mapping^)
        )
        if not defined HSA_OVERRIDE_GFX_VERSION (
            set "GFX5=!AMD_GFX:~0,5!"
            set "AMD_HSA_MAP="
            if "!GFX5!"=="gfx101" set "AMD_HSA_MAP=10.1.0"
            if "!GFX5!"=="gfx103" set "AMD_HSA_MAP=10.3.0"
            if "!GFX5!"=="gfx110" set "AMD_HSA_MAP=11.0.0"
            if defined AMD_HSA_MAP (
                set "HSA_OVERRIDE_GFX_VERSION=!AMD_HSA_MAP!"
                echo   [INFO]  HSA_OVERRIDE_GFX_VERSION=!HSA_OVERRIDE_GFX_VERSION! ^(wheel kernel family mapping^)
            ) else (
                set "GFX4=!AMD_GFX:~0,4!"
                if "!GFX4!"=="gfx9" echo   [WARN]  !AMD_GFX! has no kernels in therock multi-arch wheels - CPU fallback expected.
            )
        )
    )
)

call :section "[ 3 / 7 ]  Memory profile"
set "FORCE_CPU_ARG="
if defined AMD_GFX (
    choice /c GC /n /m "  [INFO]  AMD GPU detected - use (G)PU or (C)PU? "
    if errorlevel 2 set "FORCE_CPU_ARG=-ForceCpu"
) else (
    echo   [INFO]  No AMD GPU detected - using CPU mode.
    set "FORCE_CPU_ARG=-ForceCpu"
)
if not exist "%ROOT%launcher" mkdir "%ROOT%launcher" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%launcher\setup_memory.ps1" -ConfigPath "%AMD_DIR%\spectra_launcher.cfg" -Backend AMD -DetectedVramMb "!AMD_VRAM!" %RC_ARG% %FORCE_CPU_ARG%
if errorlevel 1 (
    echo   [FAIL]  Memory profile setup failed.
    pause
    exit /b 1
)
if exist "%AMD_DIR%\spectra_launcher.cfg.env" (
    for /f "usebackq delims=" %%a in ("%AMD_DIR%\spectra_launcher.cfg.env") do set "%%a"
    del "%AMD_DIR%\spectra_launcher.cfg.env" 2>nul
)
if defined SPECTRA_VRAM_MODE (
    echo   [INFO]  Active memory profile: !SPECTRA_VRAM_MODE!
    if defined SPECTRA_CPU_THREADS echo   [INFO]  CPU threads: !SPECTRA_CPU_THREADS!
    if defined SPECTRA_CPU_RAM_MB echo   [INFO]  CPU RAM budget: !SPECTRA_CPU_RAM_MB! MB
    if "!SPECTRA_VRAM_MODE!"=="cpu" echo   [INFO]  CPU mode: forcing CPU inference
)

call :section "[ 4 / 7 ]  Preparing model caches"
md "%MODEL_WEIGHTS%\hf_cache" 2>nul
md "%MODEL_WEIGHTS%\torch_cache" 2>nul
md "%MODEL_WEIGHTS%\cache" 2>nul
md "%MODEL_WEIGHTS%\config" 2>nul
md "%MODEL_WEIGHTS%\data" 2>nul
md "%MODEL_WEIGHTS%\matplotlib" 2>nul
md "%MODEL_WEIGHTS%\tunableop_cache" 2>nul
md "%MODEL_WEIGHTS%\miopen\cache" 2>nul
md "%MODEL_WEIGHTS%\tmp" 2>nul
md "%MODEL_WEIGHTS%\hf_datasets" 2>nul
md "%MODEL_WEIGHTS%\sentence_transformers" 2>nul
md "%MODEL_WEIGHTS%\triton" 2>nul
md "%MODEL_WEIGHTS%\numba" 2>nul

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

call :section "[ 5 / 7 ]  Initializing ROCm SDK"
where rocm-sdk >nul 2>&1
if not errorlevel 1 (
    echo   [INFO]  Initializing ROCm SDK...
    for /f "delims=" %%a in ('rocm-sdk init 2^>nul') do set "ROCM_INIT=%%a"
    if defined ROCM_INIT if exist "!ROCM_INIT!" call "!ROCM_INIT!"
    for /f "delims=" %%a in ('rocm-sdk path --root 2^>nul') do set "HIP_PATH=%%a"
    if defined HIP_PATH (
        set "ROCM_PATH=!HIP_PATH!"
        echo   [INFO]  ROCm SDK initialized ^(HIP_PATH=!HIP_PATH!^)
    ) else (
        echo   [WARN]  Could not determine ROCm SDK root path
    )
) else (
    echo   [INFO]  rocm-sdk CLI not found - portable HIP DLLs from pip package are sufficient.
)

call :section "[ 6 / 7 ]  Starting backend"
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*main:app*') { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 taskkill /F /PID %%p >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*run dev*' -or $c.CommandLine -like '*vite*') { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 taskkill /F /PID %%p >nul 2>&1
)

if not exist "%AMD_DIR%\logs" mkdir "%AMD_DIR%\logs"
echo   [INFO]  Starting backend (port 8000)...
start /B "Spectra-Segment-Backend-AMD" cmd /c ""%AMD_DIR%\run.cmd" > "%AMD_DIR%\backend_launch.log" 2>&1"

echo   [WAIT]  Waiting for backend at http://127.0.0.1:8000/health ...
set "RETRIES=0"
:health_loop
timeout /t 2 /nobreak >nul
set /a RETRIES+=1
if !RETRIES! gtr 30 (
    echo   [FAIL]  Backend did not start within 60 seconds
    if exist "%AMD_DIR%\backend_launch.log" (
        echo   --- %AMD_DIR%\backend_launch.log ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%AMD_DIR%\backend_launch.log' -Tail 15"
    )
    if exist "%AMD_DIR%\logs\backend.log" (
        echo   --- %AMD_DIR%\logs\backend.log ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%AMD_DIR%\logs\backend.log' -Tail 15"
    )
    echo           Check: %AMD_DIR%\logs\backend.log
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('powershell -NoProfile -Command "try { $null = (New-Object System.Net.WebClient).DownloadString('http://127.0.0.1:8000/health'); 'ok' } catch { 'no' }" 2^>nul') do (
    if "%%a"=="ok" goto :backend_ready
)
goto :health_loop

:backend_ready
echo   [ OK ]  Backend is ready

call :section "[ 7 / 7 ]  Starting frontend"
set "WEB_DIR=%AMD_DIR%\web"
set "BUN_DIR_OVERRIDE=%ROOT%tools\bun-amd"
call "%ROOT%frontend\bun_env.cmd"
if errorlevel 1 (
    echo   [WARN]  Bun not available - download failed.
    echo           Download Bun from https://bun.com/docs/install
    echo   [WARN]  Frontend not started
    goto :amd_done
)
call "%ROOT%frontend\use_web.bat" "%WEB_DIR%" >nul 2>&1
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
start /B "Spectra-Segment-Frontend" cmd /c ""!BUN_EXE!" run dev --host 127.0.0.1 --port 3000"
popd

:amd_done

echo(
echo  +------------------------------------------------------------+
echo  ^|  Spectra Segment (AMD) is starting up.                     ^|
echo  ^|  Backend:  http://127.0.0.1:8000                           ^|
echo  ^|  Frontend: http://127.0.0.1:3000                           ^|
echo  +------------------------------------------------------------+
echo(
echo   [INFO]  Logs: %AMD_DIR%\logs\backend.log
echo(
endlocal
exit /b 0

:section
echo(
echo   ================================================================
echo    %~1
echo   ----------------------------------------------------------------
echo(
exit /b 0
