@echo off
setlocal enabledelayedexpansion


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

if not exist "%WEB%\bun.lock" if exist "%FRONT%bun.lock" (
    copy /y "%FRONT%bun.lock" "%WEB%\bun.lock" >nul 2>&1
    del /q "%FRONT%bun.lock" >nul 2>&1
)

exit /b 0

:link_dir
set "CUR="
if exist "%~1" (
    fsutil reparsepoint query "%~1" >nul 2>&1
    if errorlevel 1 (
        rmdir /s /q "%~1" >nul 2>&1
    ) else (
        for /f "delims=" %%t in ('powershell -NoProfile -Command "(Get-Item -LiteralPath '%~1').Target" 2^>nul') do set "CUR=%%t"
        if /i not "!CUR!"=="%~2" rd "%~1" >nul 2>&1
    )
)
if not exist "%~1" mklink /J "%~1" "%~2" >nul 2>&1
exit /b 0

:link_file
if exist "%~1" del /q "%~1" >nul 2>&1
mklink /H "%~1" "%~2" >nul 2>&1
if not exist "%~1" exit /b 1
exit /b 0
