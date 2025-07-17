@echo off
echo 🤖 Starting Multi-Agent AI System...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo 📋 Creating .env file from template...
    copy .env.example .env
    echo ✅ Please edit .env file with your API keys, then run this script again.
    echo    Most importantly, set your ANTHROPIC_API_KEY
    pause
    exit /b 1
)

REM Check if API key is set
findstr "sk-ant-api03-" .env >nul
if %errorlevel% neq 0 (
    echo ❌ Please set your ANTHROPIC_API_KEY in the .env file
    echo    Get your key from: https://console.anthropic.com/
    pause
    exit /b 1
)

echo 🐳 Starting Docker services...

REM Start core services (database, message queue, cache)
docker-compose up -d postgres redis nats

echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Start the application
echo 🚀 Starting portal and agents...
docker-compose up -d portal agents

echo 📊 Starting monitoring (optional)...
docker-compose up -d prometheus grafana

echo.
echo ✅ System is starting up!
echo.
echo 🌐 Web Portal: http://localhost:8000
echo 📊 Grafana Dashboard: http://localhost:3000 (admin/admin)
echo 📈 Prometheus Metrics: http://localhost:9090
echo 💬 NATS Monitoring: http://localhost:8222
echo.
echo 📋 To check status:
echo    docker-compose ps
echo.
echo 📋 To view logs:
echo    docker-compose logs -f portal
echo    docker-compose logs -f agents
echo.
echo 📋 To stop everything:
echo    docker-compose down
echo.
echo 🎉 Visit http://localhost:8000 to submit your first request!
pause
