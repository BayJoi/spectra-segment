@echo off
setlocal enabledelayedexpansion

rem ================================================================
rem  use_web.bat <WebDir>
rem  Ensures the per-backend frontend artifact vault at <WebDir>
rem  exists and points the shared frontend's node_modules/ and
rem  dist/ at it via directory junctions.
rem
rem  <WebDir> owns the real node_modules/, dist/ and build cache.
rem  The shared source in this folder stays untouched, so each
rem  backend's frontend can be cleaned up independently.
rem
rem  Idempotent. Re-points the junctions to <WebDir>, replacing
rem  any legacy real directories or stale junctions.
rem ================================================================

set "FRONT=%~dp0"
set "WEB=%~1"

if "%WEB%"=="" (
    echo  [FAIL]  Usage: use_web.bat ^<WebDir^>
    echo          e.g. use_web.bat ..\backend\web
    exit /b 1
)

if not exist "%FRONT%package.json" (
    echo  [FAIL]  Shared frontend source not found at %FRONT%
    exit /b 1
)

for %%D in (node_modules dist) do (
    if not exist "%WEB%\%%D" mkdir "%WEB%\%%D" >nul 2>&1
)

for %%D in (node_modules dist) do call :link_dir "%FRONT%%%D" "%WEB%\%%D"

for %%D in (node_modules dist) do (
    if not exist "%FRONT%%%D" (
        echo  [FAIL]  Could not create junction %FRONT%%%D
        exit /b 1
    )
)

call :link_file "%WEB%\package.json" "%FRONT%package.json"
if not exist "%WEB%\package.json" (
    echo  [FAIL]  Could not link %WEB%\package.json
    exit /b 1
)

rem One-time migration: seed this backend's lockfile from the legacy
rem shared one, then drop the shared copy so it is never written again.
if not exist "%WEB%\bun.lock" if exist "%FRONT%bun.lock" (
    copy /y "%FRONT%bun.lock" "%WEB%\bun.lock" >nul 2>&1
    del /q "%FRONT%bun.lock" >nul 2>&1
)

exit /b 0

:link_dir
set "CUR="
rem %1 = link path (in shared frontend), %2 = target path (in web dir)
if exist "%~1" (
    fsutil reparsepoint query "%~1" >nul 2>&1
    if errorlevel 1 (
        rem real directory (legacy layout) - replace it
        rmdir /s /q "%~1" >nul 2>&1
    ) else (
        for /f "delims=" %%t in ('powershell -NoProfile -Command "(Get-Item -LiteralPath '%~1').Target" 2^>nul') do set "CUR=%%t"
        if /i not "!CUR!"=="%~2" rd "%~1" >nul 2>&1
    )
)
if not exist "%~1" mklink /J "%~1" "%~2" >nul 2>&1
exit /b 0

:link_file
rem %1 = link path (in web dir), %2 = source path (in shared frontend)
rem package.json is only read by bun install, so a hard link stays in sync.
if exist "%~1" del /q "%~1" >nul 2>&1
mklink /H "%~1" "%~2" >nul 2>&1
if not exist "%~1" exit /b 1
exit /b 0
