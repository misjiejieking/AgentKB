@echo off
echo ========================================
echo   AgentKB - Personal Knowledge Agent
echo ========================================
echo.

cd /d "%~dp0\.."

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11+.
    pause
    exit /b 1
)

:: Create data directories
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\vectors" mkdir "data\vectors"
if not exist "data\logs" mkdir "data\logs"

:: Install dependencies
echo [1/2] Installing dependencies...
pip install -q -r requirements.txt 2>nul
if %errorlevel% neq 0 (
    echo [WARN] pip install failed, trying to continue...
)

:: Run
echo [2/2] Starting AgentKB...
set PYTHONPATH=src;%PYTHONPATH%
python -m agentkb.main

pause
