# Local Development Guide

Complete guide for running the RAG system on your local machine.

## Quick Start (First Time)

```bash
# 1. One-time setup
./setup_local.sh

# 2. Start services
./start_local.sh

# 3. Open browser
open http://localhost:8501
```

That's it! 🎉

---

## Detailed Setup Instructions

### Prerequisites

**Required:**
- Python 3.9+ ([python.org](https://python.org))
- Docker Desktop ([docker.com](https://docker.com/products/docker-desktop))
- Ollama ([ollama.com](https://ollama.com)) - for local LLM

**Optional:**
- `jq` for JSON parsing: `brew install jq`

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| Disk | 10GB free | 20GB free |
| CPU | 2 cores | 4+ cores |
| GPU | None | Helps with Ollama performance |

---

## Setup Process

### Step 1: Install Prerequisites

**Python:**
```bash
# Check version
python3 --version  # Should be 3.9+

# If not installed:
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
# Windows: Download from python.org
```

**Docker:**
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify
docker --version
docker-compose --version
```

**Ollama (Local LLM):**
```bash
# macOS: Download from https://ollama.com
# Or: brew install ollama

# Start Ollama
ollama serve  # Or start Ollama.app

# Pull model
ollama pull llama3.2

# Verify
curl http://localhost:11434/api/tags
```

### Step 2: Run Setup Script

```bash
chmod +x setup_local.sh
./setup_local.sh
```

This will:
1. ✅ Create Python virtual environment
2. ✅ Install all dependencies
3. ✅ Start Elasticsearch (Docker)
4. ✅ Verify Ollama is running
5. ✅ Test Python imports

**Troubleshooting setup:**

```bash
# If Elasticsearch fails to start:
docker-compose down
docker volume rm gcp-proto_es-data
docker-compose up -d elasticsearch

# If Python packages fail:
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If Ollama not found:
# Download and install from https://ollama.com
ollama pull llama3.2
```

### Step 3: Start Services

```bash
./start_local.sh
```

This will:
1. ✅ Load environment variables
2. ✅ Start/verify Elasticsearch
3. ✅ Start API (port 8000)
4. ✅ Start UI (port 8501)
5. ✅ Display access URLs

**Services:**
- **UI**: http://localhost:8501 - Main interface
- **API**: http://localhost:8000 - REST API
- **Elasticsearch**: http://localhost:9200 - Vector store

---

## Using the System

### Add Knowledge Sources

1. Open http://localhost:8501
2. Go to **Admin** → **Add Source** tab
3. Enter a URL pattern:
   - Single page: `https://fastapi.tiangolo.com/`
   - Recursive: `https://docs.python.org/3/*`
4. Configure:
   - Crawl Depth: 2 (how many link levels deep)
   - Max Pages: 50 (page limit)
5. Click **Add Source**
6. Go to **Sources** tab, click **▶️ Ingest**
7. Wait for status: ⏳ Pending → 🔄 In Progress → ✅ Completed

### Query Knowledge Base

1. Go to **Search** tab
2. Enter a question: "How do I install FastAPI?"
3. View:
   - Answer from LLM
   - Retrieved documents (if relevant)
   - Relevance scores
4. Give feedback with 👍/👎

### View Metrics

1. Go to **Metrics** tab
2. See:
   - Total queries
   - Thumbs up rate
   - Response times
   - Popular queries
   - Low-rated queries (for improvement)

---

## Environment Variables

Edit `.env` or `env.local.template`:

```bash
# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# LLM & Embeddings (local)
EMBEDDING_PROVIDER=local
LLM_PROVIDER=local
OLLAMA_URL=http://localhost:11434

# API
API_URL=http://localhost:8000

# Optional: Use Vertex AI instead
# EMBEDDING_PROVIDER=vertex
# LLM_PROVIDER=vertex
# GOOGLE_PROJECT_ID=your-project-id
# GOOGLE_REGION=us-central1
```

---

## Development Workflow

### Making Code Changes

```bash
# 1. Make changes to Python files
vim rag_api.py

# 2. Stop services (Ctrl+C in terminal running start_local.sh)

# 3. Clear cache
find . -name "*.pyc" -delete
rm -rf __pycache__

# 4. Restart
./start_local.sh
```

### Testing Changes

```bash
# Test API endpoint
curl http://localhost:8000/

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Test source management
curl http://localhost:8000/api/admin/sources
```

### Viewing Logs

```bash
# API logs
tail -f /tmp/rag_api.log

# UI logs  
tail -f /tmp/rag_ui.log

# Elasticsearch logs
docker logs rag-elasticsearch -f
```

### Database Operations

```bash
# Check indexed documents
curl http://localhost:9200/knowledge_base/_count

# View sample documents
curl http://localhost:9200/knowledge_base/_search?size=1 | jq .

# Check sources
curl http://localhost:9200/knowledge_sources/_search | jq .

# Reset knowledge base (delete all)
curl -X DELETE http://localhost:9200/knowledge_base

# Reset sources
curl -X DELETE http://localhost:9200/knowledge_sources

# Completely reset (including Docker volumes)
docker-compose down -v
./setup_local.sh
```

---

## Performance Tuning

### For Faster Local Development

**Use smaller models:**
```bash
# Edit rag_llm_abstraction.py
model_name='llama3.2:1b'  # Instead of llama3.2 (3B)

# Or use even smaller embedding model
# Edit rag_embeddings.py - already using all-MiniLM-L6-v2 (good balance)
```

**Reduce chunk size for faster indexing:**
```bash
# Edit rag_api.py startup_event()
chunker = DocumentChunker(chunk_size=300, overlap=30)  # Instead of 500/50
```

### For Better Quality

**Use better embedding model:**
```bash
# Install
pip install sentence-transformers

# Edit rag_embeddings.py, add config:
'local_mpnet': EmbeddingConfig(
    provider='local',
    model_name='all-mpnet-base-v2',  # Better quality than MiniLM
    dimensions=768
)

# Update rag_api.py to use it
embedding_config = EMBEDDING_CONFIGS['local_mpnet']
```

**Adjust relevance threshold:**
```bash
# Edit rag_service.py
RELEVANCE_THRESHOLD = 3.0  # Lower = more lenient, higher = stricter
```

---

## Troubleshooting

### Elasticsearch Issues

**Port already in use:**
```bash
docker ps | grep 9200
docker stop $(docker ps -q --filter "publish=9200")
```

**Elasticsearch won't start:**
```bash
# Check logs
docker logs rag-elasticsearch

# Common fix: Clear data
docker-compose down -v
docker-compose up -d elasticsearch
```

**Out of memory:**
```bash
# Reduce Elasticsearch heap (in docker-compose.yml)
ES_JAVA_OPTS=-Xms512m -Xmx512m  # Instead of 1g
```

### Ollama Issues

**Connection refused:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# If not, start:
ollama serve

# Or start Ollama.app (macOS)
```

**Model not found:**
```bash
# List models
ollama list

# Pull llama3.2
ollama pull llama3.2

# For faster responses (smaller model):
ollama pull llama3.2:1b
```

### API Issues

**Port 8000 already in use:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

**Import errors:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Slow responses:**
- Check Ollama performance (GPU accelerated?)
- Reduce chunk count in search (edit `top_k` parameter)
- Use smaller LLM model

### UI Issues

**Port 8501 already in use:**
```bash
# Kill streamlit
pkill -f streamlit
```

**UI not loading:**
```bash
# Check UI logs
tail -f /tmp/rag_ui.log

# Verify API is running
curl http://localhost:8000/
```

---

## Data Persistence

### Where Data is Stored

```bash
# Elasticsearch data (persistent)
docker volume inspect gcp-proto_es-data

# Location on disk (macOS):
~/Library/Containers/com.docker.docker/Data/vms/0/

# All your knowledge sources and embeddings are here!
```

### Backup & Restore

**Backup:**
```bash
# Export Elasticsearch data
docker exec rag-elasticsearch \
  curl -X POST "localhost:9200/_snapshot/my_backup/snapshot_1?wait_for_completion=true"

# Or simpler: Export to JSON
curl http://localhost:9200/knowledge_base/_search?size=10000 > backup.json
```

**Restore:**
```bash
# Re-index from sources (preferred - always fresh)
# Use Admin UI → Sources → Re-ingest

# Or restore from JSON backup
# (requires custom import script)
```

### Reset Everything

```bash
# Stop services
docker-compose down

# Delete volumes (loses all data!)
docker-compose down -v

# Start fresh
./setup_local.sh
./start_local.sh
```

---

## Advanced Usage

### Using Vertex AI Locally

Test Vertex AI models before deploying to GCP:

```bash
# 1. Authenticate
gcloud auth application-default login

# 2. Set project
export GOOGLE_PROJECT_ID=your-gcp-project-id
export GOOGLE_REGION=us-central1

# 3. Start with Vertex AI
export EMBEDDING_PROVIDER=vertex
export LLM_PROVIDER=vertex

./start_local.sh
```

### Custom Scrapers

Add your own scraper for specific sites:

```python
# custom_scraper.py
from rag_etl_pipeline import Document

class CustomScraper:
    def scrape(self, url):
        # Your custom logic
        return Document(...)

# Use in Admin UI or API
```

### Testing & Evaluation

See `docs/SIMILARITY_SEARCH_GUIDE.md` for comprehensive testing strategies.

**Quick test:**
```bash
# Run test queries
python << 'EOF'
import requests

queries = [
    "How to install FastAPI?",
    "What is dependency injection?",
    "How to add CORS middleware?"
]

for q in queries:
    resp = requests.post("http://localhost:8000/api/query",
                        json={"query": q}).json()
    score = resp['retrieved_docs'][0]['score'] if resp.get('retrieved_docs') else 0
    print(f"{q[:30]:30} Score: {score:.2f}")
EOF
```

---

## File Structure

```
gcp-proto/
├── setup_local.sh          # One-time setup
├── start_local.sh          # Start services
├── env.local.template      # Environment variables
│
├── rag_api.py             # FastAPI backend
├── rag_ui.py              # Streamlit UI
├── rag_service.py         # RAG logic
├── rag_sources.py         # Source management
├── rag_etl_pipeline.py    # Document processing
│
├── docker-compose.yml     # Elasticsearch config
├── requirements.txt       # Python dependencies
│
└── docs/
    ├── LOCAL_DEVELOPMENT.md         # This file
    ├── SIMILARITY_SEARCH_GUIDE.md   # Testing & optimization
    └── GCP_DEPLOYMENT.md            # Deploy to cloud
```

---

## Tips & Best Practices

### 1. Keep Elasticsearch Running

Don't stop/start Elasticsearch frequently - it retains your data:

```bash
# This is fine - ES keeps running:
kill $API_PID $UI_PID

# Only stop ES when needed:
docker-compose down
```

### 2. Monitor Resources

```bash
# Check Docker resource usage
docker stats rag-elasticsearch

# If Elasticsearch uses too much RAM, reduce heap in docker-compose.yml
```

### 3. Use Meaningful Source Titles

When adding sources, use descriptive titles:
- ✅ "FastAPI Official Docs - Tutorial Section"
- ❌ "fastapi.tiangolo.com"

Makes management easier!

### 4. Start Small

Don't ingest huge sites immediately:
- Start with 10-20 pages
- Test search quality
- Then increase if needed

### 5. Clear Cache After Code Changes

```bash
# Always clear cache after editing Python files
find . -name "*.pyc" -delete
rm -rf __pycache__
```

Or just use `./start_local.sh` which does it automatically.

---

## Next Steps

- **Learn about tuning**: See `docs/SIMILARITY_SEARCH_GUIDE.md`
- **Deploy to GCP**: See `docs/GCP_DEPLOYMENT.md`
- **Add custom scrapers**: Extend `rag_etl_pipeline.py`
- **Integrate with your tools**: Use the REST API

---

## Quick Reference

### Common Commands

```bash
# Start everything
./start_local.sh

# Stop services
# (Press Ctrl+C in terminal running start_local.sh)

# View API logs
tail -f /tmp/rag_api.log

# Test API
curl http://localhost:8000/

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 3}'

# List sources
curl http://localhost:8000/api/admin/sources

# Check Elasticsearch
curl http://localhost:9200/_cluster/health
curl http://localhost:9200/knowledge_base/_count
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `EMBEDDING_PROVIDER` | `local` | local, vertex, azure |
| `LLM_PROVIDER` | `local` | local, vertex, azure |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `API_URL` | `http://localhost:8000` | API endpoint for UI |

---

**Need help?** Check the troubleshooting section or see `docs/SIMILARITY_SEARCH_GUIDE.md` for testing strategies.

