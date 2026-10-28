@echo off
chcp 65001 >nul
title AgenticOC - AI NPC Server

echo ============================================
echo   AgenticOC - AI-Powered NPC Conversations
echo ============================================
echo.
echo   OC Builder:  http://localhost:8000/oc
echo   AI Server:   http://localhost:8000
echo.

::: Check for packaged exe (in ai/ subdirectory)
set "EXE_PATH=%~dp0ai\AgenticOC-Server.exe"

if exist "%EXE_PATH%" (
    :: --- Packaged mode: run the standalone exe ---
    echo [INFO] Starting AgenticOC Server (standalone)...
    echo [INFO] Keep this window open while playing Stardew Valley.
    echo.

    :: Check config
    if not exist "%~dp0ai\config.json" (
        echo [ERROR] ai\config.json not found!
        echo.
        echo   Please create ai\config.json with:
        echo   {
        echo       "Provider": "deepseek",
        echo       "ModelName": "deepseek-chat",
        echo       "ServerAddress": "https://api.deepseek.com",
        echo       "ApiKey": "sk-your-api-key-here",
        echo       "Temperature": 0.75
        echo   }
        echo.
        pause
        exit /b 1
    )

    cd /d "%~dp0ai"
    AgenticOC-Server.exe
) else (
    :: --- Dev mode: run via Python ---
    echo [INFO] Starting AgenticOC Server (Python dev mode)...
    echo [INFO] Keep this window open while playing Stardew Valley.
    echo.

    ::: Detect a Python that already has the project dependencies
    set "PYCMD="
    python -c "import fastapi, FlagEmbedding, dotenv" >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
    if not defined PYCMD (
        py -3 -c "import fastapi, FlagEmbedding, dotenv" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -3"
    )
    if not defined PYCMD (
        echo [ERROR] No Python with AgenticOC dependencies found.
        echo.
        echo   Install Python 3.11, then run:
        echo     cd ai
        echo     python -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo [INFO] Using Python: %PYCMD%

    :: Check config
    %PYCMD% -c "import json; c=json.load(open('ai/config.json','r',encoding='utf-8')); assert c.get('ApiKey','').strip()!='', 'ApiKey is empty'" 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Please fill in your API key in ai\config.json!
        echo.
        echo   Open ai\config.json and set "ApiKey": "sk-your-api-key-here"
        echo.
        pause
        exit /b 1
    )

    cd /d "%~dp0"
    %PYCMD% ai/server.py
)

pause
