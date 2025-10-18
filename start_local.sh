#!/bin/bash
# Start RAG system locally
# Run ./setup_local.sh first if this is your first time

set -e

echo "🚀 Starting RAG Knowledge Search System..."
echo ""

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment from .env..."
    export $(cat .env | grep -v '^#' | xargs)
elif [ -f "env.local.template" ]; then
    echo "Loading environment from env.local.template..."
    source env.local.template
fi

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run: ./setup_local.sh"
    exit 1
fi

# Check Elasticsearch
echo ""
echo "Checking services..."
if ! curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    echo "⚠️  Elasticsearch not running. Starting..."
    docker-compose up -d elasticsearch
    sleep 20
fi

if curl -s http://localhost:9200/_cluster/health | grep -q green; then
    echo "✅ Elasticsearch is healthy"
else
    echo "⚠️  Elasticsearch starting up..."
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running"
    echo "   Start Ollama.app or run: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

# Clear Python cache for clean start
echo ""
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Start API
echo ""
echo "🔌 Starting API on port 8000..."
python rag_api.py > /tmp/rag_api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
for i in {1..15}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ API is ready (PID: $API_PID)"
        break
    fi
    sleep 1
done

# Start UI
echo ""
echo "📊 Starting UI on port 8501..."
streamlit run rag_ui.py --server.port 8501 > /tmp/rag_ui.log 2>&1 &
UI_PID=$!

sleep 3

echo ""
echo "======================================"
echo "🎉 RAG system is running!"
echo "======================================"
echo ""
echo "📊 UI:  http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "🔍 Elasticsearch: http://localhost:9200"
echo ""
echo "Logs:"
echo "  API: tail -f /tmp/rag_api.log"
echo "  UI:  tail -f /tmp/rag_ui.log"
echo ""
echo "To stop:"
echo "  kill $API_PID $UI_PID"
echo "  docker-compose down"
echo ""
echo "Press Ctrl+C to stop (will clean up background processes)"

# Cleanup on exit
trap "echo ''; echo 'Stopping services...'; kill $API_PID $UI_PID 2>/dev/null; echo 'Services stopped.'; echo 'Elasticsearch still running. To stop: docker-compose down'" EXIT

# Wait for UI process
wait $UI_PID