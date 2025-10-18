#!/bin/bash
# One-time setup for local RAG development
# Run this once, then use ./start_local.sh to start services

set -e

echo "🚀 Setting up RAG Knowledge Search System (Local)"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop"
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ Docker found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create env file if needed
if [ ! -f ".env" ] && [ -f "env.local.template" ]; then
    echo ""
    echo "📝 Creating .env from template..."
    cp env.local.template .env
    echo "✓ .env file created"
fi

# Start Elasticsearch
echo ""
echo "🔍 Starting Elasticsearch (Docker)..."
if docker ps | grep -q rag-elasticsearch; then
    echo "✓ Elasticsearch already running"
else
    docker-compose up -d elasticsearch
    echo "⏳ Waiting for Elasticsearch to be ready (30s)..."
    sleep 30
    
    # Test Elasticsearch
    if curl -s http://localhost:9200/_cluster/health | grep -q green; then
        echo "✓ Elasticsearch is healthy"
    else
        echo "⚠️  Elasticsearch may still be starting up"
    fi
fi

# Check Ollama
echo ""
echo "🤖 Checking Ollama (Local LLM)..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    models=$(curl -s http://localhost:11434/api/tags | grep -o 'llama3.2' || echo "")
    if [ -n "$models" ]; then
        echo "✓ Ollama is running with llama3.2 model"
    else
        echo "✓ Ollama is running"
        echo "⚠️  llama3.2 model not found. Installing..."
        ollama pull llama3.2
        echo "✓ llama3.2 model installed"
    fi
else
    echo "❌ Ollama not running"
    echo ""
    echo "Please install and start Ollama:"
    echo "  1. Download from: https://ollama.com"
    echo "  2. Install and start Ollama.app"
    echo "  3. Run: ollama pull llama3.2"
    echo "  4. Re-run this script"
    exit 1
fi

# Test Python imports
echo ""
echo "🧪 Testing Python imports..."
python3 << 'PYTHON'
try:
    from elasticsearch import Elasticsearch
    from sentence_transformers import SentenceTransformer
    import streamlit
    import fastapi
    print("✓ All Python packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
PYTHON

echo ""
echo "======================================"
echo "✅ Local setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Start services: ./start_local.sh"
echo "  2. Open browser: http://localhost:8501"
echo "  3. Add knowledge sources in Admin tab"
echo ""
echo "For more info, see: docs/LOCAL_DEVELOPMENT.md"
echo ""

