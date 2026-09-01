@echo off
setlocal enabledelayedexpansion

title Spectra Segment - AMD Cleanup

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "VENV_DIR=%AMD_DIR%\.venv"
set "UV_CACHE=%AMD_DIR%\.uv_cache_amd"
set "TOOLS_DIR=%AMD_DIR%\tools"
set "MODELS_DIR=%AMD_DIR%\model_weights"
set "LOGS_DIR=%AMD_DIR%\logs"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                        Cleanup  (AMD)                      ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(
echo    Protected: checks\ folder (health-check script + logs) is never removed.

call :section "[ 1 / 3 ]  Reviewing items to remove"
echo   The following will be permanently removed:
echo(

set "HAS_ITEMS=0"

if exist "%VENV_DIR%" (
    echo    [DIR]  backend_amd_gpu\.venv\  ^(virtual environment^)
    set "HAS_ITEMS=1"
)
if exist "%UV_CACHE%" (
    echo    [DIR]  backend_amd_gpu\.uv_cache_amd\  ^(uv package cache^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\pip_cache" (
    echo    [DIR]  backend_amd_gpu\pip_cache\  ^(pip cache^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\hf_cache" (
    echo    [DIR]  model_weights\hf_cache\  ^(HuggingFace cache^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\torch_cache" (
    echo    [DIR]  model_weights\torch_cache\  ^(PyTorch cache^)
    set "HAS_ITEMS=1"
)
if exist "%TOOLS_DIR%" (
    echo    [DIR]  backend_amd_gpu\tools\  ^(legacy uv from older installs^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\tunableop_cache" (
    echo    [DIR]  model_weights\tunableop_cache\  ^(ROCm TunableOp cache^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\matplotlib" (
    echo    [DIR]  model_weights\matplotlib\  ^(matplotlib cache^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\cache" (
    echo    [DIR]  model_weights\cache\  ^(XDG cache^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\config" (
    echo    [DIR]  model_weights\config\  ^(XDG config^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\data" (
    echo    [DIR]  model_weights\data\  ^(XDG data^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\tmp" (
    echo    [DIR]  model_weights\tmp\  ^(temp files^)
    set "HAS_ITEMS=1"
)
if exist "%LOGS_DIR%" (
    echo    [DIR]  logs\  ^(log files^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\.gpu_arch.txt" (
    echo    [FILE] backend_amd_gpu\.gpu_arch.txt  ^(GPU detection temp^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\.gpu_id.txt" (
    echo    [FILE] backend_amd_gpu\.gpu_id.txt  ^(GPU detection temp^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\.gpu_arch.txt" (
    echo    [FILE] model_weights\.gpu_arch.txt  ^(GPU detection temp^)
    set "HAS_ITEMS=1"
)
if exist "%MODELS_DIR%\.gpu_id.txt" (
    echo    [FILE] model_weights\.gpu_id.txt  ^(GPU detection temp^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\backend_launch.log" (
    echo    [FILE] backend_amd_gpu\backend_launch.log  ^(launch output^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\spectra_launcher.cfg" (
    echo    [FILE] backend_amd_gpu\spectra_launcher.cfg  ^(memory profile^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\spectra_launcher.cfg.env" (
    echo    [FILE] backend_amd_gpu\spectra_launcher.cfg.env  ^(transient env from launch^)
    set "HAS_ITEMS=1"
)
if exist "%AMD_DIR%\web" (
    echo    [DIR]  backend_amd_gpu\web\  ^(frontend workspace - AMD^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%frontend\bun.lock" (
    echo    [FILE] frontend\bun.lock  ^(bun dependency lockfile^)
    set "HAS_ITEMS=1"
)

if exist "%ROOT%tools\uv-amd" (
    echo    [DIR]  tools\uv-amd\  ^(uv binary - AMD^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\python\cpython-3.12*" (
    echo    [DIR]  tools\python\cpython-3.12*\  ^(portable Python 3.12.9 - AMD^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\bun-amd" (
    echo    [DIR]  tools\bun-amd\  ^(Bun - AMD^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\clip_src" (
    echo    [DIR]  tools\clip_src\  ^(CLIP source archive - AMD^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%\.tmp" (
    echo    [DIR]  .tmp\  ^(installer temp files^)
    set "HAS_ITEMS=1"
)

set "PYCACHE_COUNT=0"
for /d /r "%AMD_DIR%" %%d in (__pycache__) do (
    if exist "%%d" set /a PYCACHE_COUNT+=1
)
if !PYCACHE_COUNT! GTR 0 (
    echo    [DIR]  backend_amd_gpu\*\__pycache__\  ^(!PYCACHE_COUNT! directories^)
    set "HAS_ITEMS=1"
)

if exist "%MODELS_DIR%\pycache" (
    echo    [DIR]  model_weights\pycache\  ^(python bytecode cache^)
    set "HAS_ITEMS=1"
)

if "%HAS_ITEMS%"=="0" (
    echo   [INFO]  Nothing to clean.
    pause
    exit /b 0
)

echo(
echo   [WARN]  This cleanup cannot be undone.
echo(

call :section "[ 2 / 3 ]  Stopping processes"
echo   [INFO]  Stopping AMD backend...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*backend_amd_gpu.main:app*') { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 (
        taskkill /F /PID %%p >nul 2>&1 && echo  backend killed (PID %%p^)
    )
)
echo   [INFO]  Stopping frontend...
call :is_own_web "%ROOT%frontend\node_modules" "%AMD_DIR%\web\node_modules"
if not errorlevel 1 (
    for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%p >nul 2>&1 && echo  frontend killed (PID %%p^)
    )
)
timeout /t 2 /nobreak >nul
echo   [ OK ]  Processes stopped.

echo(

set /p "CONFIRM=Proceed with cleanup? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo   [INFO]  Cleanup cancelled.
    exit /b 0
)

echo(

call :section "[ 3 / 3 ]  Removing items"

if exist "%VENV_DIR%" (
    echo Removing .venv...
    rmdir /s /q "%VENV_DIR%" 2>nul && echo  Done. || echo   [WARN]  Could not remove .venv ^(in use?^)
)

if exist "%UV_CACHE%" (
    echo Removing .uv_cache_amd...
    rmdir /s /q "%UV_CACHE%" 2>nul && echo  Done.
)

if exist "%AMD_DIR%\pip_cache" (
    echo Removing pip_cache...
    rmdir /s /q "%AMD_DIR%\pip_cache" 2>nul && echo  Done.
)

if exist "%TOOLS_DIR%" (
    echo Removing tools...
    rmdir /s /q "%TOOLS_DIR%" 2>nul && echo  Done.
)

if exist "%ROOT%tools\uv-amd" (
    echo Removing tools\uv-amd ^(uv binary - AMD^)...
    rmdir /s /q "%ROOT%tools\uv-amd" 2>nul && echo  Done.
)
echo Removing tools\python\cpython-3.12*\ (portable Python 3.12 - AMD)...
for /d %%d in ("%ROOT%tools\python\cpython-3.12*") do (
    rmdir /s /q "%%~d" 2>nul && echo  Done.
)
set "PY_REMAIN=0"
for /d %%d in ("%ROOT%tools\python\cpython-*") do set "PY_REMAIN=1"
if "!PY_REMAIN!"=="0" if exist "%ROOT%tools\python" (
    echo Removing tools\python ^(no versions remain^)...
    rmdir /s /q "%ROOT%tools\python" 2>nul && echo  Done.
)
if exist "%ROOT%tools\bun-amd" (
    echo Removing tools\bun-amd ^(Bun - AMD^)...
    rmdir /s /q "%ROOT%tools\bun-amd" 2>nul && echo  Done.
)
if exist "%ROOT%tools\clip_src" (
    echo Removing tools\clip_src ^(CLIP source archive - AMD^)...
    rmdir /s /q "%ROOT%tools\clip_src" 2>nul && echo  Done.
)

if exist "%LOGS_DIR%" (
    echo Removing logs...
    rmdir /s /q "%LOGS_DIR%" 2>nul && echo  Done.
)

echo Removing __pycache__ directories...
for /d /r "%AMD_DIR%" %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
echo  Done.

if exist "%MODELS_DIR%\pycache" (
    echo Removing model_weights\pycache...
    rmdir /s /q "%MODELS_DIR%\pycache" 2>nul && echo  Done.
)

if exist "%AMD_DIR%\.gpu_arch.txt" del /q "%AMD_DIR%\.gpu_arch.txt" && echo  Removed .gpu_arch.txt
if exist "%AMD_DIR%\.gpu_id.txt" del /q "%AMD_DIR%\.gpu_id.txt" && echo  Removed .gpu_id.txt
if exist "%MODELS_DIR%\.gpu_arch.txt" del /q "%MODELS_DIR%\.gpu_arch.txt" && echo  Removed model_weights\.gpu_arch.txt
if exist "%MODELS_DIR%\.gpu_id.txt" del /q "%MODELS_DIR%\.gpu_id.txt" && echo  Removed model_weights\.gpu_id.txt
if exist "%AMD_DIR%\backend_launch.log" del /q "%AMD_DIR%\backend_launch.log" && echo  Removed backend_launch.log
if exist "%AMD_DIR%\spectra_launcher.cfg" del /q "%AMD_DIR%\spectra_launcher.cfg" && echo  Removed spectra_launcher.cfg
if exist "%AMD_DIR%\spectra_launcher.cfg.env" del /q "%AMD_DIR%\spectra_launcher.cfg.env" && echo  Removed spectra_launcher.cfg.env
if exist "%AMD_DIR%\web" (
    echo Removing backend_amd_gpu\web\ ^(frontend workspace^)...
    rmdir /s /q "%AMD_DIR%\web" 2>nul && echo  Done.
)
if exist "%ROOT%frontend\bun.lock" (
    del /q "%ROOT%frontend\bun.lock" && echo  Removed frontend\bun.lock
)
call :remove_own_web_link "%ROOT%frontend\node_modules" "%AMD_DIR%\web\node_modules"
call :remove_own_web_link "%ROOT%frontend\dist" "%AMD_DIR%\web\dist"

if exist "%ROOT%\.tmp" (
    rmdir /s /q "%ROOT%\.tmp" >nul 2>&1
)

if exist "%ROOT%tools" (
    dir /b "%ROOT%tools" 2>nul | findstr "." >nul 2>&1
    if !errorlevel! neq 0 (
        rmdir "%ROOT%tools" >nul 2>&1
    )
)

if exist "%MODELS_DIR%" (
    echo(
    echo   [WARN]  About to remove ALL downloaded model weights and caches.
    choice /c YN /n /m "[WARN] Delete model_weights ^(including hf_cache, torch_cache, caches^)? [Y/N] "
    if errorlevel 2 (
        echo   [INFO]  Model weights kept
    ) else (
        rmdir /s /q "%MODELS_DIR%" 2>nul
        if exist "%MODELS_DIR%" (
            echo   [FAIL]  Could not remove model_weights
        ) else (
            echo   [ OK ]  Removed model_weights
        )
    )
)

echo(
call :section "Verifying protected files"
set "PROTECTED_FAIL=0"
for %%f in (.gitignore README.md THIRD-PARTY-LICENSES.txt LICENSE) do (
    if exist "%ROOT%%%f" (
        echo   [ OK ]  %%f
    ) else (
        echo   [FAIL]  %%f is MISSING!
        set "PROTECTED_FAIL=1"
    )
)
if exist "%ROOT%.git" (
    echo   [ OK ]  .git\
) else (
    echo   [FAIL]  .git\ is MISSING!
    set "PROTECTED_FAIL=1"
)
if "!PROTECTED_FAIL!"=="1" (
    echo(
    echo   [WARN]  Some protected files were lost. Restore from git or backup.
)

echo(
echo  +------------------------------------------------------------+
echo  ^|  Cleanup complete.                                         ^|
echo  ^|  To reinstall: install_amd.bat                             ^|
echo  +------------------------------------------------------------+
echo(
pause

endlocal
exit /b 0

:remove_own_web_link
rem Remove a frontend junction only if it points at this backend's web dir
set "CUR="
if exist "%~1" (
    fsutil reparsepoint query "%~1" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%t in ('powershell -NoProfile -Command "(Get-Item -LiteralPath '%~1').Target" 2^>nul') do set "CUR=%%t"
        if /i "!CUR!"=="%~2" (
            echo Removing %~1 ^(junction^)...
            rd "%~1" >nul 2>&1
            if not exist "%~1" echo   Done.
        )
    )
)
exit /b 0

:is_own_web
rem Return errorlevel 0 only if %~1 is a junction pointing at %~2
set "CUR="
if exist "%~1" (
    fsutil reparsepoint query "%~1" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%t in ('powershell -NoProfile -Command "(Get-Item -LiteralPath '%~1').Target" 2^>nul') do set "CUR=%%t"
    )
)
if /i "!CUR!"=="%~2" exit /b 0
exit /b 1

:section
echo(
echo   ================================================================
echo    %~1
echo   ----------------------------------------------------------------
echo(
exit /b 0
