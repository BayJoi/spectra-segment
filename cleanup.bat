@echo off
setlocal enabledelayedexpansion

title Spectra Segment - Cleanup

set "ROOT=%~dp0"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                         Cleanup  (CPU)                     ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo( 
echo    Protected: docs\ folder and checks\ folder are never removed.

call :section "[ 1 / 3 ]  Reviewing items to remove"
echo   The following will be permanently removed:
echo(
echo    docs\ is documentation and is always kept.

set "HAS_ITEMS=0"

if exist "%ROOT%tools\uv-cpu" (
    echo    [DIR]  tools\uv-cpu\  ^(uv binary - CPU^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\uv" (
    echo    [DIR]  tools\uv\  ^(legacy shared uv from older installs^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\python\cpython-*" (
    echo    [DIR]  tools\python\cpython-*\  ^(portable Python - CPU^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\bun-cpu" (
    echo    [DIR]  tools\bun-cpu\  ^(Bun - CPU^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\bun" (
    echo    [DIR]  tools\bun\  ^(legacy shared Bun from older installs^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%tools\clip_src" (
    echo    [DIR]  tools\clip_src\  ^(CLIP source archive^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\.embedded_python" (
    echo    [DIR]  backend\.embedded_python\  ^(legacy embedded Python^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\.venv" (
    echo    [DIR]  backend\.venv\  ^(virtual environment^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\.uv" (
    echo    [DIR]  backend\.uv\  ^(package cache^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights" (
    echo    [DIR]  backend\model_weights\  ^(downloaded models^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\.local_token" (
    echo    [FILE] backend\.local_token
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\.backend_pid" (
    echo    [FILE] backend\.backend_pid
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\backend_launch.log" (
    echo    [FILE] backend\backend_launch.log  ^(launch output^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\spectra_launcher.cfg" (
    echo    [FILE] backend\spectra_launcher.cfg  ^(memory profile^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\spectra_launcher.cfg.env" (
    echo    [FILE] backend\spectra_launcher.cfg.env  ^(transient env from launch^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\web" (
    echo    [DIR]  backend\web\  ^(frontend workspace - CPU^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%frontend\bun.lock" (
    echo    [FILE] frontend\bun.lock  ^(bun dependency lockfile^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%\.tmp" (
    echo    [DIR]  .tmp\  ^(installer temp files^)
    set "HAS_ITEMS=1"
)

if exist "%ROOT%backend\model_weights\hf_cache" (
    echo    [DIR]  backend\model_weights\hf_cache\  ^(HuggingFace cache^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\torch_cache" (
    echo    [DIR]  backend\model_weights\torch_cache\  ^(PyTorch cache^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\matplotlib" (
    echo    [DIR]  backend\model_weights\matplotlib\  ^(matplotlib cache^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\cache" (
    echo    [DIR]  backend\model_weights\cache\  ^(XDG cache^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\config" (
    echo    [DIR]  backend\model_weights\config\  ^(XDG config^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\data" (
    echo    [DIR]  backend\model_weights\data\  ^(XDG data^)
    set "HAS_ITEMS=1"
)
if exist "%ROOT%backend\model_weights\tmp" (
    echo    [DIR]  backend\model_weights\tmp\  ^(temp files^)
    set "HAS_ITEMS=1"
)

if exist "%ROOT%backend\logs" (
    echo    [DIR]  backend\logs\  ^(log files^)
    set "HAS_ITEMS=1"
)

set "PYCACHE_COUNT=0"
for /d /r "%ROOT%backend" %%d in (__pycache__) do (
    if exist "%%d" set /a PYCACHE_COUNT+=1
)
if !PYCACHE_COUNT! GTR 0 (
    echo    [DIR]  backend\*\__pycache__\  ^(!PYCACHE_COUNT! directories^)
    set "HAS_ITEMS=1"
)

if exist "%ROOT%backend\model_weights\pycache" (
    echo    [DIR]  backend\model_weights\pycache\  ^(python bytecode cache^)
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

call :section "[ 2 / 3 ]  Stopping running processes"
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*main:app*') { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 (
        taskkill /F /PID %%p >nul 2>&1
    )
)
call :is_own_web "%ROOT%frontend\node_modules" "%ROOT%backend\web\node_modules"
if not errorlevel 1 (
    for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
        powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*vite*' -or $c.CommandLine -like '*run dev*') { exit 0 } } catch {}; exit 1" >nul 2>&1
        if not errorlevel 1 (
            taskkill /F /PID %%p >nul 2>&1
        )
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

if exist "%ROOT%tools\uv-cpu" (
    echo Removing tools\uv-cpu\ ^(uv binary - CPU^)...
    rmdir /s /q "%ROOT%tools\uv-cpu"
    if exist "%ROOT%tools\uv-cpu" (
        echo   [WARN]  Failed to remove ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%tools\uv" (
    echo Removing tools\uv\ ^(legacy shared uv from older installs^)...
    rmdir /s /q "%ROOT%tools\uv"
    if exist "%ROOT%tools\uv" (
        echo   [WARN]  Failed to remove ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

echo Removing tools\python\cpython-*\ (portable Python - CPU)...
for /d %%d in ("%ROOT%tools\python\cpython-*") do (
    rmdir /s /q "%%~d"
    if exist "%%~d" (
        echo   [WARN]  Failed to remove %%~nxd\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)
set "PY_REMAIN=0"
for /d %%d in ("%ROOT%tools\python\cpython-*") do set "PY_REMAIN=1"
if "!PY_REMAIN!"=="0" if exist "%ROOT%tools\python" (
    echo Removing tools\python\ ^(no versions remain^)...
    rmdir /s /q "%ROOT%tools\python" >nul 2>&1
)

if exist "%ROOT%tools\bun-cpu" (
    echo Removing tools\bun-cpu\ ^(Bun - CPU^)...
    rmdir /s /q "%ROOT%tools\bun-cpu"
    if exist "%ROOT%tools\bun-cpu" (
        echo   [WARN]  Failed to remove tools\bun-cpu\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%tools\bun" (
    echo Removing tools\bun\ ^(legacy shared Bun from older installs^)...
    rmdir /s /q "%ROOT%tools\bun"
    if exist "%ROOT%tools\bun" (
        echo   [WARN]  Failed to remove tools\bun\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%tools\clip_src" (
    echo Removing tools\clip_src\ ^(CLIP source archive^)...
    rmdir /s /q "%ROOT%tools\clip_src"
    if exist "%ROOT%tools\clip_src" (
        echo   [WARN]  Failed to remove tools\clip_src\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%backend\.venv" (
    echo Removing backend\.venv\...
    rmdir /s /q "%ROOT%backend\.venv"
    if exist "%ROOT%backend\.venv" (
        echo   [WARN]  Failed to remove backend\.venv\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%backend\.embedded_python" (
    echo Removing backend\.embedded_python\...
    rmdir /s /q "%ROOT%backend\.embedded_python"
    if exist "%ROOT%backend\.embedded_python" (
        echo   [WARN]  Failed to remove backend\.embedded_python\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%backend\.uv" (
    echo Removing backend\.uv\ ^(cache^)...
    rmdir /s /q "%ROOT%backend\.uv"
    if exist "%ROOT%backend\.uv" (
        echo   [WARN]  Failed to remove backend\.uv\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)

if exist "%ROOT%backend\.local_token" (
    echo Removing backend\.local_token...
    del /q "%ROOT%backend\.local_token"
    echo   Done.
)
if exist "%ROOT%backend\.backend_pid" (
    echo Removing backend\.backend_pid...
    del /q "%ROOT%backend\.backend_pid"
    echo   Done.
)
if exist "%ROOT%backend\backend_launch.log" (
    echo Removing backend\backend_launch.log...
    del /q "%ROOT%backend\backend_launch.log"
    echo   Done.
)
if exist "%ROOT%backend\spectra_launcher.cfg" (
    echo Removing backend\spectra_launcher.cfg...
    del /q "%ROOT%backend\spectra_launcher.cfg"
    echo   Done.
)
if exist "%ROOT%backend\spectra_launcher.cfg.env" (
    echo Removing backend\spectra_launcher.cfg.env...
    del /q "%ROOT%backend\spectra_launcher.cfg.env"
    echo   Done.
)

if exist "%ROOT%backend\web" (
    echo Removing backend\web\ ^(frontend workspace^)...
    rmdir /s /q "%ROOT%backend\web"
    if exist "%ROOT%backend\web" (
        echo   [WARN]  Failed to remove backend\web\ ^(files may be in use^)
    ) else (
        echo   Done.
    )
)
if exist "%ROOT%frontend\bun.lock" (
    echo Removing frontend\bun.lock...
    del /q "%ROOT%frontend\bun.lock"
    echo   Done.
)
call :remove_own_web_link "%ROOT%frontend\node_modules" "%ROOT%backend\web\node_modules"
call :remove_own_web_link "%ROOT%frontend\dist" "%ROOT%backend\web\dist"

if exist "%ROOT%\.tmp" (
    echo Removing .tmp\...
    rmdir /s /q "%ROOT%\.tmp"
    echo   Done.
)

if exist "%ROOT%backend\logs" (
    echo Removing backend\logs\...
    rmdir /s /q "%ROOT%backend\logs"
    echo   Done.
)

echo Removing __pycache__\ directories in backend\...
for /d /r "%ROOT%backend" %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d" >nul 2>&1
    )

if exist "%ROOT%backend\model_weights\pycache" (
    rmdir /s /q "%ROOT%backend\model_weights\pycache" >nul 2>&1
)

)
echo   Done.

if exist "%ROOT%tools" (
    dir /b "%ROOT%tools" 2>nul | findstr "." >nul 2>&1
    if !errorlevel! neq 0 (
        rmdir "%ROOT%tools" >nul 2>&1
    )
)

if exist "%ROOT%backend\model_weights" (
    echo(
    echo   [WARN]  About to remove ALL downloaded model weights and caches.
    choice /c YN /n /m "[WARN] Delete model_weights ^(including hf_cache, torch_cache, caches^)? [Y/N] "
    if errorlevel 2 (
        echo   [INFO]  Model weights kept
    ) else (
        for %%d in (matplotlib cache config data tmp) do (
            if exist "%ROOT%backend\model_weights\%%d" (
                echo Removing backend\model_weights\%%d\...
                rmdir /s /q "%ROOT%backend\model_weights\%%d"
                if exist "%ROOT%backend\model_weights\%%d" (
                    echo   [WARN]  Failed to remove %%d\ ^(files may be in use^)
                ) else (
                    echo   Done.
                )
            )
        )
        rmdir /s /q "%ROOT%backend\model_weights"
        if exist "%ROOT%backend\model_weights" (
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
if exist "%ROOT%docs\*.md" (
    echo   [ OK ]  docs\ ^(documentation^)
) else (
    echo   [FAIL]  docs\ is MISSING or empty!
    set "PROTECTED_FAIL=1"
)
if exist "%ROOT%checks\check_gpu_health.py" (
    echo   [ OK ]  checks\check_gpu_health.py
) else (
    echo   [FAIL]  checks\check_gpu_health.py is MISSING!
    set "PROTECTED_FAIL=1"
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
echo  ^|  Cleanup complete                                          ^|
echo  ^|  Run install.bat for a fresh setup.                        ^|
echo  +------------------------------------------------------------+
echo(
pause

endlocal
exit /b 0

:remove_own_web_link
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
