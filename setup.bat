@echo off
setlocal enabledelayedexpansion
title INTERVION AI - Environment Setup & Launcher
echo ===============================================================================
echo                      INTERVION AI - Automated Setup
echo ===============================================================================
echo.

:: 0. Verify Python Version (Strictly 3.11 or 3.12)
echo [0/5] Verifying Python Compatibility...
python -c "import sys; ver = sys.version_info; sys.exit(0 if (ver.major == 3 and ver.minor in (11, 12)) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Incompatible Python version detected!
    echo INTERVION strictly requires Python 3.11 or Python 3.12.
    echo Please install Python 3.11 or 3.12 and ensure it is in your PATH.
    pause
    exit /b 1
)
echo [INFO] Compatible Python version detected (3.11 / 3.12).
echo.

:: 1. Check Ollama
echo [1/5] Checking Ollama Installation...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
        echo [INFO] Found Ollama in LocalAppData.
    ) else (
        echo [WARNING] Ollama executable not found in PATH!
        echo Please install Ollama from https://ollama.com
        echo Opening download page in default browser...
        start https://ollama.com/download/windows
        echo Once installed and running, press any key to continue setup...
        pause >nul
    )
)

:: Ensure Ollama service is running in background
echo Checking Ollama server connection...
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama server...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: 2. Pull Qwen 2.5 Model
echo.
echo [2/5] Pulling Qwen 2.5 LLM Model (qwen2.5:latest)...
echo This may take a few minutes if not already downloaded...
ollama pull qwen2.5:latest
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull Qwen model. Please check your internet connection or Ollama installation.
    pause
    exit /b 1
)

:: 3. Create Python Virtual Environment
echo.
echo [3/5] Setting up Python Virtual Environment (interview_ai)...
if not exist "interview_ai\Scripts\python.exe" (
    echo [INFO] Creating new virtual environment...
    python -m venv interview_ai
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.11 or 3.12 is installed.
        pause
        exit /b 1
    )
)

:: Activate environment
call "interview_ai\Scripts\activate.bat"
echo [INFO] Virtual environment activated: %VIRTUAL_ENV%

:: 4. Install Dependencies
echo.
echo [4/5] Installing Dependencies (PyTorch >= 2.4 with CUDA, Whisper, Gradio, etc.)...
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Installing PyTorch >= 2.4.0 with CUDA 12.4 support...
pip install torch>=2.4.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

echo [INFO] Installing PyAudio for microphone capture...
pip install PyAudio
if %errorlevel% neq 0 (
    echo [WARNING] Default PyAudio wheel failed. Attempting pipwin installation...
    pip install pipwin
    pipwin install pyaudio
)

echo [INFO] Installing required project dependencies from requirements.txt...
pip install -r requirements.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    echo Please check your internet connection and try running setup.bat again.
    pause
    exit /b 1
)

:: 5. Launch UI
echo.
echo ===============================================================================
echo [5/5] Setup Complete! Launching INTERVION AI UI...
echo ===============================================================================
echo.
python ui\app.py

pause
