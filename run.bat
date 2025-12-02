@echo off
echo ========================================
echo   ProximaHand - Hand Proximity Demo
echo ========================================
echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo Starting ProximaHand...
echo.
python main.py
pause
