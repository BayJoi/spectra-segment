@echo off
setlocal enabledelayedexpansion

title Spectra Segment - Stopper (AMD)

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                        Stopper  (AMD)                      ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(

call :section "[ 1 / 2 ]  Stopping backend"

for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    powershell -NoProfile -Command "try { $c = Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%p); if ($c.CommandLine -like '*backend_amd_gpu.main:app*') { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 (
        taskkill /F /PID %%p >nul 2>&1 && echo   [ OK ]  backend killed (PID %%p^)
    )
)

call :section "[ 2 / 2 ]  Stopping frontend"

call :is_own_web "%ROOT%frontend\node_modules" "%AMD_DIR%\web\node_modules"
if not errorlevel 1 (
    for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%p >nul 2>&1 && echo   [ OK ]  frontend killed (PID %%p^)
    )
)

echo   [ OK ]  All Spectra Segment processes stopped.
echo(
endlocal
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
