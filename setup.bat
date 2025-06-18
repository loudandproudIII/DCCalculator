@echo off
set VENV_NAME=venv_%USERNAME%

echo Setting up environment for %USERNAME%...

if not exist "%VENV_NAME%" (
    echo Creating virtual environment: %VENV_NAME%
    python -m venv %VENV_NAME%
    echo Virtual environment created for %USERNAME%.
)

call %VENV_NAME%\Scripts\activate
pip install --upgrade pip --quiet
pip install pip-tools --quiet

REM Create requirements.in if it doesn't exist
if not exist "requirements.in" (
    echo # Add your high-level dependencies here > requirements.in
    echo requests >> requirements.in
)

REM Compile and install requirements
if exist "requirements.in" (
    pip-compile requirements.in --quiet
    pip-sync requirements.txt --quiet
) else if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
)

echo.
echo ✓ Setup complete for %USERNAME%!
echo ✓ Your environment: %VENV_NAME%
echo.
pause