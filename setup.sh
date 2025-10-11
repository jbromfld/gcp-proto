# setup.sh - Quick setup script for local development

#!/bin/bash

echo "🚀 Setting up RAG Knowledge Search System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "✓ Docker found"

# Start Elasticsearch
echo "🔍 Starting Elasticsearch..."
docker run -d --name rag-elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Wait for Elasticsearch
echo "⏳ Waiting for Elasticsearch to be ready..."
until curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"\|"status":"yellow"'; do
    sleep 2
done
echo "✓ Elasticsearch is ready"

# Start Ollama for local LLM
echo "🤖 Starting Ollama (local LLM)..."
docker run -d --name rag-ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama

# Wait for Ollama
sleep 5

# Pull LLM model
echo "📥 Pulling Llama 3.2 model (this may take a few minutes)..."
docker exec rag-ollama ollama pull llama3.2

echo "✓ Llama 3.2 model downloaded"

# Run initial ETL to populate data
echo "📚 Running initial data ingestion..."
python3 << 'PYTHON'
from elasticsearch import Elasticsearch
from rag_embeddings import EmbeddingFactory, EMBEDDING_CONFIGS
from rag_etl_pipeline import ETLPipeline, ElasticsearchIndexer, DocumentChunker

print("Initializing ETL pipeline...")
es_client = Elasticsearch(['http://localhost:9200'])

embedding_config = EMBEDDING_CONFIGS['local_minilm']
embedder = EmbeddingFactory.create(embedding_config)

indexer = ElasticsearchIndexer(es_client, index_name="knowledge_base")
indexer.create_index(embedding_dim=embedder.dimensions)

chunker = DocumentChunker(chunk_size=500, overlap=50)
pipeline = ETLPipeline(embedder, indexer, chunker)

scrape_urls = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://fastapi.tiangolo.com/"
]

print(f"Scraping {len(scrape_urls)} URLs...")
docs, chunks = pipeline.run_scrape_and_index(scrape_urls, max_pages_per_url=10)
print(f"✓ Indexed {docs} documents, {chunks} chunks")
PYTHON

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎉 Next steps:"
echo "   1. Start the API server:"
echo "      python rag_api.py"
echo ""
echo "   2. In another terminal, start the UI:"
echo "      streamlit run rag_ui.py"
echo ""
echo "   3. Open your browser:"
echo "      http://localhost:8501"
echo ""
echo "📖 For more information, see README.md"
