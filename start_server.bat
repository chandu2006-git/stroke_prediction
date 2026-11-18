@echo off
cd /d "C:\Users\akula\OneDrive\Desktop\stroke_prediction"
echo Starting StrokeGuard backend on http://127.0.0.1:8000
echo.
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
echo.
echo Backend stopped (exit code %ERRORLEVEL%).
