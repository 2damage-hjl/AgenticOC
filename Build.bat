@echo off
chcp 65001 >nul
title AgenticOC - Build Package

echo ============================================
echo   AgenticOC - Build Standalone Package
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    pause
    exit /b 1
)

:: Check PyInstaller
python -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
)

:: Install dependencies
echo [INFO] Installing project dependencies...
pip install -r requirements.txt
pip install langchain langchain-openai langchain-chroma langchain-google-genai fastapi uvicorn

echo.
echo [INFO] Starting PyInstaller build (this may take 5-10 minutes)...
echo.

cd /d "%~dp0"
pyinstaller AgenticOC.spec --noconfirm --clean

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
