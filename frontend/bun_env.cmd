@echo off

if defined BUN_DIR_OVERRIDE (
    set "BUN_DIR=%BUN_DIR_OVERRIDE%"
) else (
    set "BUN_DIR=%~dp0..\tools\bun"
)
set "BUN_EXE="

if not exist "%BUN_DIR%" mkdir "%BUN_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bun_fetch.ps1" -Dir "%BUN_DIR%"
if not errorlevel 1 goto :have_project_bun

if exist "%BUN_DIR%\bun.exe" goto :have_project_bun

echo(
echo  [ERROR] The project Bun could not be fetched and no local copy
echo          exists at %BUN_DIR%.
echo          Connect to the internet and run this again, or download
echo          Bun manually to %BUN_DIR% from https://bun.com/docs/install
echo(
exit /b 1

:have_project_bun
set "BUN_EXE=%BUN_DIR%\bun.exe"

:bun_ready
set "DO_NOT_TRACK=1"
set "BUN_INSTALL_CACHE_DIR=%BUN_DIR%\install\cache"
set "BUN_RUNTIME_TRANSPILER_CACHE_PATH=%BUN_DIR%\transpile"
set "TMPDIR=%BUN_DIR%\tmp"
if not exist "%BUN_INSTALL_CACHE_DIR%" mkdir "%BUN_INSTALL_CACHE_DIR%"
if not exist "%BUN_RUNTIME_TRANSPILER_CACHE_PATH%" mkdir "%BUN_RUNTIME_TRANSPILER_CACHE_PATH%"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"
set "PATH=%BUN_DIR%;%PATH%"

"%BUN_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Bun could not run: %BUN_EXE%
    exit /b 1
)
exit /b 0
