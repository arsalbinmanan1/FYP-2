@echo off
REM ============================================================
REM VisionAI - Webcam Detection Script (Windows)
REM ============================================================

echo.
echo ============================================================
echo   VisionAI - Webcam Detection
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Change to project root
cd /d "%~dp0.."

REM Find best model
set MODEL_PATH=computer-vision\models\best.pt

if not exist "%MODEL_PATH%" (
    echo Model not found at %MODEL_PATH%
    echo Looking for alternative models...
    
    if exist "computer-vision\models\my_model.pt" (
        set MODEL_PATH=computer-vision\models\my_model.pt
    ) else (
        echo No trained model found. Using default yolo11n.pt
        set MODEL_PATH=yolo11n.pt
    )
)

echo Using model: %MODEL_PATH%
echo.

REM Run webcam detection
python computer-vision\src\inference.py ^
    --source webcam ^
    --model "%MODEL_PATH%" ^
    --conf 0.5

echo.
pause

