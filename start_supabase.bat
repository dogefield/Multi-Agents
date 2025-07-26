@echo off
echo 🚀 Starting AI Agent System (Supabase Edition)
echo.

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found!
    echo Creating from template...
    copy .env.example .env
    echo.
    echo ⚠️  Please edit .env file with your:
    echo    - ANTHROPIC_API_KEY
    echo    - SUPABASE_URL
    echo    - SUPABASE_SERVICE_KEY
    echo.
    pause
    exit /b
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
pip install supabase python-dotenv

echo.
echo 🌐 Starting portal...
echo.
echo Visit http://localhost:8000
echo If Supabase isn't configured, go to http://localhost:8000/setup
echo.

python portal_supabase.py
