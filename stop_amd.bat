@echo off
setlocal enabledelayedexpansion

title Spectra Segment - Stopper (AMD)

set "ROOT=%~dp0"
set "AMD_DIR=%ROOT%backend_amd_gpu"
set "STOP_FAILED=0"
set "WAS_RUNNING=0"

echo(
echo  +------------------------------------------------------------+
echo  ^|                      SPECTRA SEGMENT                       ^|
echo  ^|                        Stopper  (AMD)                      ^|
echo  ^|  Portable, self-contained. Everything stays in this folder ^|
echo  +------------------------------------------------------------+
echo(
netstat -aon | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 set "WAS_RUNNING=1"
netstat -aon | findstr ":3000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 set "WAS_RUNNING=1"
if "%WAS_RUNNING%"=="0" echo   [INFO]  Nothing was listening on ports 3000 or 8000.
echo(

call :section "[ 1 / 3 ]  Stopping backend"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "$root='%ROOT%';" ^
  "$self=$PID; $targets=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -and (($_.CommandLine -like '*backend_amd_gpu.main:app*') -or (($_.CommandLine -like ($root + '*')) -and ($_.CommandLine -like '*backend_amd_gpu*'))) };" ^
  "foreach ($p in @($targets)) { taskkill /F /T /PID $p.ProcessId | Out-Null; Write-Host ('  [ OK ]  backend killed (PID ' + $p.ProcessId + ')') }; exit 0"

call :section "[ 2 / 3 ]  Stopping frontend"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "$root='%ROOT%';" ^
  "$self=$PID; $targets=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -and ($_.CommandLine -like '*vite*') -and ($_.CommandLine -like ($root + '*')) };" ^
  "foreach ($p in @($targets)) { taskkill /F /T /PID $p.ProcessId | Out-Null; Write-Host ('  [ OK ]  frontend killed (PID ' + $p.ProcessId + ')') }; exit 0"

call :section "[ 3 / 3 ]  Verifying shutdown"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "$open=Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 3000,8000 };" ^
  "if ($open) { foreach ($c in $open) { Write-Host ('  [WARN]  port ' + $c.LocalPort + ' is still held by PID ' + $c.OwningProcess) }; exit 1 }" ^
  "Write-Host '  [ OK ]  ports 3000 and 8000 are closed'; exit 0"
if errorlevel 1 (
    set "STOP_FAILED=1"
    echo   [FAIL]  One or more Spectra ports are still open. Close the owning process manually.
) else (
    echo   [ OK ]  Ports 3000 and 8000 are free.
)
echo(
echo  +------------------------------------------------------------+
if "%STOP_FAILED%"=="0" (
    echo  ^|  RESULT: Spectra Segment is STOPPED.                      ^|
    echo  ^|  Ports 3000 and 8000 are free.                            ^|
) else (
    echo  ^|  RESULT: STOP INCOMPLETE.                                 ^|
    echo  ^|  A port is still held; see the PID above.                 ^|
)
echo  +------------------------------------------------------------+
echo(
if not defined SPECTRA_NO_PAUSE (
    echo   Press any key to close this window...
    pause >nul
)
endlocal & exit /b %STOP_FAILED%

:section
echo(
echo   ================================================================
echo    %~1
echo   ----------------------------------------------------------------
echo(
exit /b 0
