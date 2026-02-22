@echo off
echo Setting up Python environment for RAG Workplace Assistant...

REM Remove old venv if it exists
IF EXIST venv (
    echo Existing venv found. Removing it...
    rmdir /S /Q venv
)

REM Create fresh virtual environment
python3 -m venv venv

REM Install dependencies
call .\venv\Scripts\activate && pip install -r requirements.txt

echo.
echo Environment ready.
echo Activate it with:
echo .\venv\Scripts\activate
echo.
echo Then run two commands:
echo python ingest.py
echo python app.py