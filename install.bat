@echo off

echo Checking if Python is installed...
python --version
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.x from https://www.python.org/downloads/
    exit /b
)

echo Checking if pip is installed...
pip --version
if %errorlevel% neq 0 (
    echo pip is not installed. Please install pip
)

echo Installing Python dependencies...
pip install -r requirements.txt

echo Installation complete.