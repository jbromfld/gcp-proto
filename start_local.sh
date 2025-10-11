# start_local.sh - Start all services locally

#!/bin/bash

echo "🚀 Starting RAG Knowledge Search System..."

# Activate virtual environment
source venv/bin/activate

# Check if services are running
if ! curl -s http://localhost:9200 > /dev/null; then
    echo "❌ Elasticsearch not running. Run setup.sh first."
    exit 1
fi

if ! curl -s http://localhost:11434 > /dev/null; then
    echo "❌ Ollama not running. Run setup.sh first."
    exit 1
fi

# Start API in background
echo "🔧 Starting API server..."
python rag_api.py > logs/api.log 2>&1 &
API_PID=$!
echo "API server started (PID: $API_PID)"

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5
until curl -s http://localhost:8000 > /dev/null; do
    sleep 2
done
echo "✓ API is ready"

# Start UI
echo "🎨 Starting Streamlit UI..."
streamlit run rag_ui.py

# Cleanup on exit
trap "echo 'Stopping services...'; kill $API_PID" EXIT