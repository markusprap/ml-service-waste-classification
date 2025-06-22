@echo off
echo ================================
echo ML Service Quick Start
echo ================================
echo.
echo Activating virtual environment...
call "tf_216_env\Scripts\activate.bat"

echo.
echo Starting ML Classification Server...
echo Server will be available at: http://localhost:5000
echo.
echo Health check: http://localhost:5000/health  
echo Classification endpoint: http://localhost:5000/api/classify
echo.
echo Press Ctrl+C to stop the server
echo ================================

python app.py

pause
