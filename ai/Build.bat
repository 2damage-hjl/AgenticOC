@echo off
chcp 65001 >nul
title AgenticOC - Build Package

echo ============================================
echo   AgenticOC - Build Standalone Package
echo ============================================
echo.

::: Detect Python command (prefer py launcher, fallback to python)
set "PYCMD="
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYCMD=py -3"
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYCMD=python"
    ) else (
        echo [ERROR] Python not found! Please install Python 3.10+ first.
        pause
        exit /b 1
    )
)
echo [INFO] Using Python: %PYCMD%

::: Check PyInstaller
%PYCMD% -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing PyInstaller...
    %PYCMD% -m pip install pyinstaller
)

::: Check core dependencies (skip install if already satisfied)
echo [INFO] Checking project dependencies...
%PYCMD% -c "import fastapi, FlagEmbedding, dotenv" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies from requirements.txt...
    %PYCMD% -m pip install -r requirements.txt
) else (
    echo [INFO] All core dependencies already installed, skipping.
)

echo.
echo [INFO] Starting PyInstaller build (this may take 5-10 minutes)...
echo.

cd /d "%~dp0"
%PYCMD% -m PyInstaller AgenticOC.spec --noconfirm --clean

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo.
echo   Output: dist\AgenticOC-Server\
echo.
echo   Next steps:
echo   1. Copy config.json to dist\AgenticOC-Server\
echo   2. Copy the entire dist\AgenticOC-Server\ folder
echo      into your Stardew Valley Mods\AgenticOC\ai\ folder
echo   3. Users just double-click AgenticOC-Server.exe
echo.

pause
