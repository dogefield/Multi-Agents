#!/bin/bash

# Multi-Agent System Quick Start Script

echo "🤖 Starting Multi-Agent AI System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please edit .env file with your API keys, then run this script again."
    echo "   Most importantly, set your ANTHROPIC_API_KEY"
    exit 1
fi

# Check if API key is set
if ! grep -q "sk-ant-api03-" .env; then
    echo "❌ Please set your ANTHROPIC_API_KEY in the .env file"
    echo "   Get your key from: https://console.anthropic.com/"
    exit 1
fi

echo "🐳 Starting Docker services..."

# Start core services (database, message queue, cache)
docker-compose up -d postgres redis nats

echo "⏳ Waiting for services to be ready..."
sleep 10

# Start the application
echo "🚀 Starting portal and agents..."
docker-compose up -d portal agents

echo "📊 Starting monitoring (optional)..."
docker-compose up -d prometheus grafana

echo ""
echo "✅ System is starting up!"
echo ""
echo "🌐 Web Portal: http://localhost:8000"
echo "📊 Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "📈 Prometheus Metrics: http://localhost:9090"
echo "💬 NATS Monitoring: http://localhost:8222"
echo ""
echo "📋 To check status:"
echo "   docker-compose ps"
echo ""
echo "📋 To view logs:"
echo "   docker-compose logs -f portal"
echo "   docker-compose logs -f agents"
echo ""
echo "📋 To stop everything:"
echo "   docker-compose down"
echo ""
echo "🎉 Visit http://localhost:8000 to submit your first request!"
