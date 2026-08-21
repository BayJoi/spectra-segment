@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "VENV_PYTHON=%ROOT%backend\.venv\Scripts\python.exe"
set "PIDFILE=%ROOT%\backend\.backend_pid"

set "RC_ARG="
if /i "%~1"=="-reconfigure" set "RC_ARG=-Reconfigure"

title Spectra Segment - Launcher (NVIDIA / CPU)

cls
echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                  Launcher (NVIDIA / CPU)                   ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 5 ]  Preflight"
if not exist "%VENV_PYTHON%" (
    echo   [FAIL]  Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

call :section "[ 2 / 5 ]  Memory profile"
set "FORCE_CPU_ARG="
powershell -NoProfile -Command "if (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'NVIDIA' }) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo   [INFO]  No NVIDIA GPU detected - using CPU mode.
    set "FORCE_CPU_ARG=-ForceCpu"
) else (
    echo   [INFO]  NVIDIA GPU detected.
    choice /c GC /n /m "  [INFO]  Use (G)PU or (C)PU? "
    if errorlevel 2 set "FORCE_CPU_ARG=-ForceCpu"
)
if not exist "%ROOT%launcher" mkdir "%ROOT%launcher" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%launcher\setup_memory.ps1" -ConfigPath "%ROOT%backend\spectra_launcher.cfg" -Backend NVIDIA %RC_ARG% %FORCE_CPU_ARG%
if errorlevel 1 (
    echo   [FAIL]  Memory profile setup failed.
    pause
    exit /b 1
)
if exist "%ROOT%backend\spectra_launcher.cfg.env" (
    for /f "usebackq delims=" %%a in ("%ROOT%backend\spectra_launcher.cfg.env") do set "%%a"
    del "%ROOT%backend\spectra_launcher.cfg.env" 2>nul
)
if defined SPECTRA_VRAM_MODE (
    echo   [INFO]  Active memory profile: !SPECTRA_VRAM_MODE!
    if defined SPECTRA_CPU_THREADS echo   [INFO]  CPU threads: !SPECTRA_CPU_THREADS!
    if defined SPECTRA_CPU_RAM_MB echo   [INFO]  CPU RAM budget: !SPECTRA_CPU_RAM_MB! MB
    if "!SPECTRA_VRAM_MODE!"=="cpu" echo   [INFO]  CPU mode: forcing CPU inference
)

call :section "[ 3 / 5 ]  Checking Bun"
set "BUN_DIR_OVERRIDE=%ROOT%tools\bun-cpu"
call "%ROOT%frontend\bun_env.cmd"
if !errorlevel! neq 0 (
    echo   [FAIL]  Bun is not available.
    echo           Download it from https://bun.com/docs/install
    pause
    exit /b 1
)

call :section "[ 4 / 5 ]  Starting backend"
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and $p.CommandLine -like '*main:app*') { Stop-Process -Id $_ -Force } }; exit 0"
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and ($p.CommandLine -like '*run dev*' -or $p.CommandLine -like '*vite*')) { Stop-Process -Id $_ -Force } }; exit 0"
if exist "%PIDFILE%" del "%PIDFILE%" >nul 2>&1
if not exist "%ROOT%backend\logs" mkdir "%ROOT%backend\logs"
start /b cmd /c ""%ROOT%backend\run.cmd" > "%ROOT%backend\backend_launch.log" 2>&1"
echo   [ OK ]  backend started (port 8000)
echo   [WAIT]  Waiting for backend at http://127.0.0.1:8000/health ...

set "CURL_OK="
where curl >nul 2>&1
if not errorlevel 1 set "CURL_OK=1"

set "WAIT_SECONDS=0"
:wait_backend
if !WAIT_SECONDS! geq 30 (
    echo   [FAIL]  Backend failed to start within 30 seconds.
    if exist "%ROOT%backend\backend_launch.log" (
        echo   --- %ROOT%backend\backend_launch.log ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%ROOT%backend\backend_launch.log' -Tail 15"
    )
    if exist "%ROOT%backend\logs\backend.log" (
        echo   --- %ROOT%backend\logs\backend.log ^(tail^) ---
        powershell -NoProfile -Command "Get-Content -LiteralPath '%ROOT%backend\logs\backend.log' -Tail 15"
    )
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
set /a WAIT_SECONDS+=1
if defined CURL_OK (
    curl -s -o nul --max-time 2 http://127.0.0.1:8000/health >nul 2>&1
    if not errorlevel 1 goto backend_ready
) else (
    powershell -NoProfile -Command "try { (New-Object System.Net.WebClient).DownloadString('http://127.0.0.1:8000/health') | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if errorlevel 1 goto wait_backend
)
goto wait_backend

:backend_ready
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo %%p > "%ROOT%\backend\.backend_pid"
)

call :section "[ 5 / 5 ]  Starting frontend"
echo   [ OK ]  Backend ready at http://127.0.0.1:8000
set "WEB_DIR=%ROOT%backend\web"
call "%ROOT%frontend\use_web.bat" "%WEB_DIR%" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL]  Could not set up frontend workspace at %WEB_DIR%
    goto :frontend_done
)
if not exist "%WEB_DIR%\node_modules\vite\package.json" (
    echo   [INFO]  Installing frontend dependencies ^(bun install^)...
    pushd "%WEB_DIR%"
    call "%BUN_EXE%" install
    popd
)
cd /d "%ROOT%frontend"
call "%BUN_EXE%" run dev --host 127.0.0.1 --port 3000

:frontend_done

if exist "%ROOT%\backend\.backend_pid" (
    set /p PID=<"%ROOT%\backend\.backend_pid"
    taskkill /F /PID !PID! >nul 2>&1
    del "%ROOT%\backend\.backend_pid" >nul 2>&1
)
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and $p.CommandLine -like '*main:app*') { Stop-Process -Id $_ -Force } }; exit 0"

echo(
echo   [ OK ]  Frontend closed, backend stopped.
endlocal
exit /b 0

:section
echo(
echo   ================================================================
echo    %~1
echo   ----------------------------------------------------------------
echo(
exit /b 0
