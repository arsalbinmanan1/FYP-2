@echo off
REM ============================================================
REM VisionAI - Quick Training Script (Windows)
REM ============================================================

echo.
echo ============================================================
echo   VisionAI - Training Script
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Change to project root
cd /d "%~dp0.."

REM Run training
echo Starting training...
echo.

python computer-vision\src\train.py ^
    --data computer-vision\data\processed\data.yaml ^
    --model yolo11s.pt ^
    --epochs 50 ^
    --batch 16 ^
    --dagshub-owner arsal6010 ^
    --dagshub-repo FYP_visionAI

echo.
echo ============================================================
echo   Training Complete!
echo ============================================================
echo.

pause

