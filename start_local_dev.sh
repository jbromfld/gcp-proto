#!/bin/bash
# Start local development environment

set -e

echo "🚀 Starting local RAG development environment..."

# Load environment variables
if [ -f env.local.template ]; then
    echo "Loading environment from env.local.template..."
    source env.local.template
else
    echo "⚠️  env.local.template not found, using defaults"
fi

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if Elasticsearch is running
echo "Checking Elasticsearch..."
if docker ps | grep -q rag-elasticsearch; then
    echo "✅ Elasticsearch already running"
else
    echo "Starting Elasticsearch..."
    docker-compose up -d elasticsearch
    sleep 20  # Wait for Elasticsearch to be ready
fi

# Check if Ollama is running
echo "Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama not detected. Make sure it's running for local LLM."
    echo "   Download from: https://ollama.com"
fi

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Start API
echo ""
echo "Starting API on port 8000..."
python rag_api.py > /tmp/rag_api.log 2>&1 &
API_PID=$!
echo "API started (PID: $API_PID)"

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Start UI
echo ""
echo "Starting UI on port 8501..."
streamlit run rag_ui.py --server.port 8501 > /tmp/rag_ui.log 2>&1 &
UI_PID=$!
echo "UI started (PID: $UI_PID)"

echo ""
echo "======================================"
echo "🎉 Local development environment ready!"
echo "======================================"
echo ""
echo "📊 UI:  http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo ""
echo "Logs:"
echo "  API: tail -f /tmp/rag_api.log"
echo "  UI:  tail -f /tmp/rag_ui.log"
echo ""
echo "To stop:"
echo "  kill $API_PID $UI_PID"
echo "  docker-compose down"
echo ""

