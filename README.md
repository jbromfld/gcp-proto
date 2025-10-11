# RAG Knowledge Search System

A production-ready Retrieval-Augmented Generation (RAG) system with multi-cloud support, evaluation framework, and automated data ingestion.

## 🌟 Features

### Core Capabilities
- **Hybrid Search**: Combines vector similarity (semantic) with keyword matching (BM25)
- **Multi-Cloud Support**: Switch between Google Vertex AI, Azure OpenAI, or local models
- **Automated ETL**: Scheduled document scraping and indexing (24-hour cadence)
- **Evaluation Framework**: Built-in feedback collection and metrics tracking
- **Cost-Effective**: Use free local models for development, scale to cloud for production

### Architecture Highlights
- **Abstraction Layers**: Easily switch between embedding and LLM providers
- **Reranking**: Improves retrieval quality by reordering results
- **Context Management**: Prevents token overflow with smart chunking
- **Feedback Loop**: Thumbs up/down + ratings for continuous improvement

## 📋 Prerequisites

- Docker & Docker Compose (for containerized setup)
- Python 3.9+ (for local development)
- 8GB+ RAM (for running Elasticsearch + local models)
- Optional: NVIDIA GPU (for faster local LLM inference)

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo>
cd rag-knowledge-search

# Start all services
docker-compose up -d

# Wait for services to be healthy (2-3 minutes)
docker-compose logs -f

# Pull local LLM model (first time only)
docker exec rag-ollama ollama pull llama3.2

# Access the UI
open http://localhost:8501
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Elasticsearch
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Start Ollama (for local LLM)
docker run -d -p 11434:11434 ollama/ollama
docker exec $(docker ps -q -f ancestor=ollama/ollama) ollama pull llama3.2

# Run API server
python rag_api.py

# In another terminal, run UI
streamlit run rag_ui.py

# In another terminal, run ETL scheduler
python -c "from rag_etl_pipeline import *; ..."
```

## 📦 Project Structure

```
src/
├── api/               # FastAPI backend
│   ├── __init__.py
│   └── rag_api.py
├── core/              # Core business logic
│   ├── __init__.py
│   ├── rag_embeddings.py
│   ├── rag_llm_abstraction.py
│   └── rag_service.py
├── etl/               # ETL and evaluation
│   ├── __init__.py
│   ├── rag_etl_pipeline.py
│   └── rag_evaluation.py
└── ui/                # Streamlit UI
    ├── __init__.py
    └── rag_ui.py             # This file
```

## 🔧 Configuration

### Embedding Providers

```python
# Local (Free, runs on CPU)
EMBEDDING_CONFIGS['local_minilm']  # 384 dims, fast
EMBEDDING_CONFIGS['local_mpnet']   # 768 dims, better quality

# Google Vertex AI (Paid, requires GCP setup)
EMBEDDING_CONFIGS['vertex_gecko']  # 768 dims

# Azure OpenAI (Paid, requires Azure setup)
EMBEDDING_CONFIGS['azure_ada']     # 1536 dims
```

### LLM Providers

```python
# Local (Free, uses Ollama)
LLM_CONFIGS['local_llama']  # llama3.2, mistral, phi, etc.

# Google Vertex AI (Paid)
LLM_CONFIGS['vertex_gemini_flash']  # Fast and cheap
LLM_CONFIGS['vertex_gemini_pro']    # More capable

# Azure OpenAI (Paid)
LLM_CONFIGS['azure_gpt4']  # Most capable
```

### Environment Variables

```bash
# For local development
export EMBEDDING_PROVIDER=local
export LLM_PROVIDER=local
export ELASTICSEARCH_URL=http://localhost:9200

# For Google Cloud deployment
export EMBEDDING_PROVIDER=vertex
export LLM_PROVIDER=vertex
export GOOGLE_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# For Azure deployment
export EMBEDDING_PROVIDER=azure
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-api-key
```

## 📊 Evaluation & Ranking

### Feedback Collection

The system automatically logs every query with:
- Retrieved documents and scores
- LLM response
- Latency metrics
- Cost estimates

Users can provide feedback via:
- 👍 Thumbs up / 👎 Thumbs down
- 1-5 star ratings
- Text comments

### Metrics Dashboard

Access at `http://localhost:8501` → Metrics tab:
- **Satisfaction rate**: % thumbs up
- **Response time**: Avg, P95, P99 latency
- **Popular queries**: Most common searches
- **Low-rated queries**: Flagged for improvement
- **Cost tracking**: Total API costs

### Ranking Strategy

1. **Hybrid search**: Combines vector (70%) + keyword (30%)
2. **Reranking**: Adjusts scores based on query term overlap
3. **Context limit**: Selects top results within 2K token budget
4. **Deduplication**: Prevents similar chunks from same document

### Continuous Improvement

```python
# Export low-rated queries to test suite
from rag_evaluation import TestSuite, FeedbackStore

test_suite = TestSuite(feedback_store)
test_suite.export_failing_queries_to_tests(days=7)

# Add manual test cases
test_suite.add_test_case(
    query="How do I install FastAPI?",
    expected_keywords=["pip", "install", "fastapi"],
    min_score=0.7
)

# Run regression tests
results = test_suite.run_tests(rag_service)
print(f"Passed: {results['passed']}/{results['total']}")
```

## 🔄 ETL & Data Ingestion

### Scheduled Scraping

ETL runs automatically every 24 hours:
- Scrapes configured URLs
- Detects changed documents (via content hash)
- Re-indexes only modified content
- Logs scraping results

### Manual Ingestion

Via UI (Admin tab):
1. Enter URLs to scrape
2. Set max pages per URL
3. Click "Start Ingestion"

Via API:
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://docs.example.com"],
    "max_pages_per_url": 50
  }'
```

### Supported Sources

- Public documentation sites (HTML)
- Confluence (with authentication)
- Static site generators (Jekyll, Hugo, etc.)
- Any site with structured content

## 🧪 Testing

### Local Testing with Free Models

```bash
# Use sentence-transformers for embeddings (no API key needed)
export EMBEDDING_PROVIDER=local

# Use Ollama for LLM (no API key needed)
export LLM_PROVIDER=local
docker run -d -p 11434:11434 ollama/ollama
docker exec $(docker ps -q -f ancestor=ollama/ollama) ollama pull llama3.2

# Run tests
python -m pytest tests/
```

### Migration to Cloud

1. **Test locally first** with free models
2. **Benchmark quality** with your data
3. **Switch to cloud** when ready:
   ```python
   # Update config in rag_api.py startup
   embedding_config = EMBEDDING_CONFIGS['vertex_gecko']
   llm_config = LLM_CONFIGS['vertex_gemini_flash']
   ```
4. **Monitor costs** via metrics dashboard

## 🔐 Security Considerations

- **API authentication**: Add API key middleware (not included in POC)
- **Rate limiting**: Implement per-user limits
- **PII detection**: Scan queries/responses for sensitive data
- **Access control**: Restrict admin endpoints
- **HTTPS**: Use reverse proxy (nginx) in production

## 📈 Scaling to Production

### Performance Optimization

```python
# Enable caching for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(query: str):
    return embedder.embed_query(query)

# Use semantic cache
from sentence_transformers import util

def semantic_cache_lookup(query_embedding, cache, threshold=0.95):
    for cached_query, cached_response in cache.items():
        similarity = util.cos_sim(query_embedding, cached_query)
        if similarity > threshold:
            return cached_response
    return None
```

### Infrastructure

- **Elasticsearch cluster**: 3+ nodes for HA
- **Load balancer**: Distribute API requests
- **Redis cache**: Cache embeddings and responses
- **CDN**: Serve UI assets
- **Monitoring**: Prometheus + Grafana

## 💰 Cost Estimates

### Local (Development)
- **Cost**: $0/month
- **Performance**: Good for <100 queries/day
- **Latency**: 1-3 seconds per query

### Google Vertex AI (Production)
- **Embeddings**: ~$0.025 per 1K queries (one-time per document)
- **Gemini Flash**: ~$0.0005 per query
- **Expected**: $15-50/month for 10K queries

### Azure OpenAI (Production)
- **Embeddings**: ~$0.0001 per 1K tokens
- **GPT-4 Turbo**: ~$0.001-0.003 per query
- **Expected**: $30-100/month for 10K queries

## 🐛 Troubleshooting

### Elasticsearch won't start
```bash
# Increase vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### Ollama model not found
```bash
# Pull the model manually
docker exec rag-ollama ollama pull llama3.2
```

### API returns 503
```bash
# Check if all services are healthy
docker-compose ps
docker-compose logs api
```

### Slow queries
- Reduce `top_k` (fewer documents to retrieve)
- Use smaller LLM model (llama3.2 → phi)
- Enable caching
- Check Elasticsearch heap size

## 📝 License

MIT License - feel free to use for commercial or personal projects.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Advanced reranking (cross-encoder models)
- Multi-turn conversations with memory
- Fine-tuning embedding models
- Support for more document formats (PDF, DOCX)
- A/B testing framework

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Open an issue on GitHub
4. Contact: your-email@example.com