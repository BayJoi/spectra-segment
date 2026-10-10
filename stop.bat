@echo off
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
set "PIDFILE=%ROOT%\backend\.backend_pid"

title Spectra Segment - Stopper

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                         Stopper  (CPU)                     ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 2 ]  Stopping backend"

if exist "%PIDFILE%" (
    set /p PID=<"%PIDFILE%"
    del "%PIDFILE%" >nul 2>&1
    if defined PID (
        powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %PID%); if ($c.CommandLine -like '*main:app*') { exit 0 } } catch {}; exit 1" >nul 2>&1
        if not errorlevel 1 (
            taskkill /F /PID !PID! >nul 2>&1 && echo   [ OK ]  backend killed ^(PID !PID!^)
        ) else (
            echo   [INFO]  Stale PID file ignored - process no longer matches.
        )
    )
)

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and $p.CommandLine -like '*main:app*') { Stop-Process -Id $_ -Force; Write-Host '  [ OK ]  backend killed' } }; exit 0"

call :section "[ 2 / 2 ]  Stopping frontend"

call :is_own_web "%ROOT%frontend\node_modules" "%ROOT%backend\web\node_modules"
if not errorlevel 1 (
    powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_); if ($p -and ($p.CommandLine -like '*run dev*' -or $p.CommandLine -like '*vite*')) { Stop-Process -Id $_ -Force; Write-Host '  [ OK ]  frontend killed' } }; exit 0"
)

echo   [ OK ]  All Spectra Segment processes stopped.
echo(
pause >nul
endlocal
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
