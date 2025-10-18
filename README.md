# RAG Knowledge Search System

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search, multi-cloud support, and intelligent source management.

## 🚀 Quick Start

### Local Development

```bash
# One-time setup
./setup_local.sh

# Start services
./start_local.sh

# Open browser → http://localhost:8501
```

### GCP Deployment

```bash
# 1. Configure Terraform
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
vim terraform/terraform.tfvars  # Set project_id

# 2. Build and push Docker images
gcloud builds submit --config cloudbuild.yaml

# 3. Deploy infrastructure
cd terraform
terraform init
terraform apply

# 4. Services auto-deploy via Terraform
```

**📖 Complete Guides:**
- **Local**: [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md)
- **GCP**: [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md)

**Full shutdown** ($0/month):
```bash
cd terraform && terraform destroy  # Complete removal
terraform apply                     # Rebuild from scratch (~15 mins)
```

---

## 📖 About the Project

### What Inspired This Project

As a DevOps engineer, I found myself constantly searching through internal documentation, troubleshooting guides, and knowledge bases to solve infrastructure challenges. Traditional search tools often returned irrelevant results or outdated information, making it difficult to find the right solutions quickly.

I was inspired to build this RAG system after seeing how AI-powered search could transform how we access and utilize organizational knowledge. The idea was to create a system that could understand the *intent* behind queries and provide contextual, accurate answers based on our actual documentation and runbooks.

### What I Learned

Building this RAG system taught me several key concepts that are crucial for DevOps engineers working with AI:

#### **RAG Architecture Fundamentals**
- **Embeddings**: How text gets converted to numerical vectors for semantic search
- **Vector Databases**: Why Elasticsearch with HNSW indexing is powerful for similarity search
- **Retrieval Strategies**: Balancing recall vs. precision in document retrieval
- **Token Management**: Critical importance of staying within LLM token limits (20k for Vertex AI)

#### **Production Considerations**
- **Scalability**: Auto-scaling Cloud Run services with proper resource allocation
- **Cost Management**: Hybrid local/GCP deployment to minimize operational costs
- **Monitoring**: Comprehensive logging and metrics for system health
- **Security**: VPC networking and IAM roles for secure cloud deployments

#### **DevOps Integration Patterns**
- **Infrastructure as Code**: Terraform for reproducible GCP deployments
- **CI/CD**: Cloud Build for automated Docker image building and deployment
- **Container Orchestration**: Docker Compose for local development, Cloud Run for production
- **Configuration Management**: Environment-specific configs with proper secrets handling

### How I Built This Project

#### **Phase 1: Core RAG Implementation**
```python
# Key architectural decisions:
# 1. Multi-provider abstraction for LLMs and embeddings
# 2. Elasticsearch as vector store with HNSW indexing
# 3. Document chunking with overlap for better retrieval
# 4. RESTful API design with FastAPI
```

#### **Phase 2: Production Deployment**
- **Local Development**: Docker Compose with Ollama for cost-effective testing
- **GCP Production**: Cloud Run services with Vertex AI for enterprise-grade performance
- **Infrastructure**: Terraform for reproducible, version-controlled deployments

#### **Phase 3: DevOps Integration**
- **Monitoring**: Comprehensive metrics and logging
- **Cost Optimization**: Pause/resume scripts for development environments
- **Documentation**: Complete guides for local and production deployment

### Challenges I Faced

#### **Token Limit Management**
**Challenge**: Vertex AI embeddings have a 20,000 token limit, but some documents exceeded this during ingestion.

**Solution**: Implemented intelligent chunking with token validation:
```python
def count_tokens(text: str) -> int:
    """Accurate token counting with tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Automatic truncation for oversized chunks
if token_count > 15000:  # Conservative buffer
    text = truncate_text(text, target_tokens=15000)
```

#### **Hybrid Local/Cloud Deployment**
**Challenge**: Balancing development efficiency with production capabilities.

**Solution**: Created dual deployment modes:
- **Local**: Ollama + sentence-transformers for rapid iteration
- **Production**: Vertex AI for enterprise-grade performance
- **Unified Interface**: Same API regardless of backend

#### **Infrastructure Complexity**
**Challenge**: Managing VPC networking, IAM roles, and service dependencies in GCP.

**Solution**: Infrastructure as Code with Terraform:
```hcl
# VPC connector for Cloud Run → GCE communication
resource "google_vpc_access_connector" "connector" {
  name = "rag-vpc-connector"
  ip_cidr_range = "10.8.0.0/28"
  network = "default"
  region = "us-central1"
}
```

#### **Cost Optimization**
**Challenge**: GCP costs can escalate quickly with always-on services.

**Solution**: Implemented cost management strategies:
- **Pause/Resume Scripts**: Stop Elasticsearch VM when not needed
- **Auto-scaling**: Cloud Run scales to zero when idle
- **Local Development**: Complete local stack for testing

### DevOps Applications

This RAG system demonstrates several DevOps principles:

#### **Observability**
- **Metrics**: Query latency, satisfaction rates, cost tracking
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Service health monitoring and alerting

#### **Reliability**
- **Error Handling**: Graceful degradation and retry logic
- **Circuit Breakers**: Protection against cascading failures
- **Data Consistency**: Proper indexing and change detection

#### **Scalability**
- **Horizontal Scaling**: Auto-scaling Cloud Run services
- **Performance**: Optimized vector search with HNSW indexing
- **Efficiency**: Batch processing and connection pooling

### Future Enhancements

#### **Hybrid RAG + LLM Fallback**
```markdown
# Planned feature for enhanced flexibility
Query → Similarity Check → Response Strategy
                        ├─ High Similarity → RAG + Context Enhancement  
                        └─ Low Similarity → Direct LLM Query
```

#### **Advanced DevOps Integration**
- **CI/CD Pipeline Integration**: Automatic documentation updates
- **Monitoring Integration**: Real-time error log context injection
- **Infrastructure Context**: Environment-specific knowledge base

This project represents a practical application of AI/ML technologies to solve real DevOps challenges, demonstrating how modern tools can enhance productivity and knowledge accessibility in infrastructure operations.

---

## 🛠️ Tech Stack

**Languages:** Python 3.11+, HTML/CSS/JavaScript, YAML, HCL (Terraform)

**Frameworks:** FastAPI, Streamlit, Uvicorn, Pydantic

**AI/ML:** Google Vertex AI (Gemini 2.5 Pro), OpenAI API, Ollama, sentence-transformers, tiktoken

**Databases:** Elasticsearch 8.11 (Vector Store)

**Cloud Services:** Google Cloud Platform (Cloud Run, Compute Engine, Cloud Build, Artifact Registry, VPC, IAM)

**Infrastructure:** Docker, Docker Compose, Terraform, Cloud Build

**Data Processing:** NumPy, Pandas, BeautifulSoup4, Requests, lxml

**Monitoring:** Structured Logging, Custom Metrics, Health Checks

**Testing:** pytest, pytest-asyncio, Type Hints

**APIs:** RESTful API, OpenAI API, Google Vertex AI API

**Other:** python-dotenv, Plotly, Shell Scripts

---

## 🌟 Features

### Knowledge Management
- ✅ **Dynamic Source Management**: Add/remove knowledge sources via Admin UI
- ✅ **Recursive Scraping**: Wildcard support (`docs.python.org/*`)  
- ✅ **Source Tracking**: Monitor ingestion status, stats, and history
- ✅ **Flexible Configuration**: Crawl depth, page limits, URL patterns

### Search & Retrieval
- ✅ **Hybrid Search**: Vector similarity + keyword matching (BM25)
- ✅ **Smart Relevance**: Automatic fallback to general knowledge when docs aren't relevant
- ✅ **Multi-Provider**: Local models, Vertex AI, or Azure OpenAI
- ✅ **Configurable Threshold**: Tune precision vs recall

### User Experience
- ✅ **Streamlit UI**: Clean interface for search, feedback, and admin
- ✅ **Real-time Feedback**: Thumbs up/down on answers
- ✅ **Metrics Dashboard**: Track performance and user satisfaction
- ✅ **Search History**: View past queries and responses

### Production Ready
- ✅ **Auto-scaling**: Cloud Run scales 0 → N instances
- ✅ **Cost Optimized**: Self-hosted Elasticsearch ($25/mo vs $95/mo)
- ✅ **Persistent Storage**: Data survives restarts
- ✅ **Scheduled Updates**: Daily ETL via Cloud Scheduler
- ✅ **Infrastructure as Code**: Full Terraform configuration

---

## 📋 System Requirements

### Local Development
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| Disk | 10GB free | 20GB free |
| CPU | 2 cores | 4+ cores |
| Software | Python 3.9+, Docker, Ollama | |

### GCP Production
| Component | Config | Monthly Cost |
|-----------|--------|--------------|
| Elasticsearch | e2-medium (4GB) | ~$25 |
| Cloud Run | Auto-scaling | ~$10-30 |
| Vertex AI | 1K-5K queries/day | ~$10-30 |
| **Total** | | **~$45-85/mo** |

---

## 🏗️ Architecture

```
Local Development:
  UI (Streamlit) → API (FastAPI) → Elasticsearch (Docker)
                        ↓
                   Ollama (Local LLM)

GCP Production:
  Cloud Run (UI) → Cloud Run (API) → GCE (Elasticsearch)
                        ↓                    ↑
                   Vertex AI           VPC Connector
                   (Gemini + Embeddings)
```

---

## 📦 Project Structure

```
gcp-proto/
├── Core Application
│   ├── rag_api.py                    # FastAPI REST API
│   ├── rag_ui.py                     # Streamlit UI
│   ├── rag_service.py                # RAG orchestration
│   ├── rag_sources.py                # Source management
│   ├── rag_etl_pipeline.py           # Document processing & scraping
│   ├── rag_embeddings.py             # Multi-provider embeddings
│   ├── rag_llm_abstraction.py        # Multi-provider LLMs
│   └── rag_evaluation.py             # Metrics & feedback
│
├── Deployment
│   ├── setup_local.sh                # Local setup (one-time)
│   ├── start_local.sh                # Start local services
│   ├── setup-gcp.sh                  # GCP setup (one-time)
│   ├── docker-compose.yml            # Local Docker config
│   ├── Dockerfile.{api,ui,etl}       # Container images
│   ├── cloudbuild.yaml               # GCP CI/CD
│   └── terraform/                    # Infrastructure as Code
│
└── Documentation
    ├── docs/LOCAL_DEVELOPMENT.md     # Local dev guide
    ├── docs/GCP_DEPLOYMENT.md        # GCP deployment guide
    ├── docs/SIMILARITY_SEARCH_GUIDE.md # Testing & optimization
    └── docs/DEPLOYMENT_SUMMARY.md    # What we built
```

---

## 📖 Documentation

| Guide | Purpose | When to Read |
|-------|---------|--------------|
| [LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md) | Run locally, develop, test | First time setup |
| [GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md) | Deploy to Google Cloud | Going to production |
| [SIMILARITY_SEARCH_GUIDE.md](docs/SIMILARITY_SEARCH_GUIDE.md) | Tuning, testing, optimization | Improving search quality |
| [DEPLOYMENT_SUMMARY.md](docs/DEPLOYMENT_SUMMARY.md) | What was built, issues resolved | Quick reference |

---

## 🔧 Configuration

See `env.local.template` for all environment variables.

**Key settings:**

| Variable | Options | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | local, vertex, azure | Where to get embeddings |
| `LLM_PROVIDER` | local, vertex, azure | Which LLM to use |
| `ELASTICSEARCH_URL` | http://... | Vector store endpoint |
| `OLLAMA_URL` | http://localhost:11434 | Local LLM (if using local) |

**Multi-Cloud Support:**

```python
# Local (Free)
EMBEDDING_PROVIDER=local  # sentence-transformers
LLM_PROVIDER=local        # Ollama

# Google Cloud
EMBEDDING_PROVIDER=vertex  # text-embedding-004
LLM_PROVIDER=vertex        # gemini-pro

# Azure
EMBEDDING_PROVIDER=azure   # text-embedding-ada-002
LLM_PROVIDER=azure         # gpt-4
```

---

## 📊 Features Deep Dive

### Intelligent Source Management

**Add sources via UI:**
- Single page: `https://fastapi.tiangolo.com/`
- Recursive crawl: `https://docs.python.org/3/` or `https://docs.langchain.com`
- Configure depth (1-5 levels), page limits, URL patterns
- Track ingestion status and statistics

**API endpoints:**
```bash
POST   /api/admin/sources              # Add source
GET    /api/admin/sources              # List all
POST   /api/admin/sources/{id}/ingest  # Trigger ingestion
DELETE /api/admin/sources/{id}         # Remove source
```

### Smart Relevance Thresholding

**The system automatically decides when to use retrieved documents:**

```python
RELEVANCE_THRESHOLD = 3.0  # Configurable in rag_service.py

if max_score < 3.0:
    # Documents not relevant enough
    # Use LLM general knowledge
else:
    # Documents are relevant
    # Use RAG with context
```

**Example:**
- Query: "How to install FastAPI?" → Score: 5.5 → Uses FastAPI docs ✅
- Query: "What is Java?" → Score: 2.1 → Uses general knowledge ✅

### Hybrid Search

Combines two search strategies:

```
Vector Search (Semantic):
  - Understands meaning: "install" ≈ "set up" ≈ "configure"
  - Good for: Conceptual questions

Keyword Search (BM25):
  - Exact term matching: "@app.get" decorator
  - Good for: Specific API calls, code snippets

Combined Score = RRF(vector_results, keyword_results)
```

---

Built-in metrics tracking (view in UI → Metrics tab):
- Query volume and satisfaction rate
- Response times (avg, P95, P99)
- Popular and low-rated queries
- Cost tracking

See [SIMILARITY_SEARCH_GUIDE.md](docs/SIMILARITY_SEARCH_GUIDE.md) for testing and optimization strategies.

---

## 🛠️ Development

**Make changes:**
```bash
# Edit code
vim rag_api.py

# Test locally
./start_local.sh

# Deploy to GCP
gcloud builds submit --config cloudbuild.yaml
cd terraform && terraform apply
```

**Testing:**
```bash
# Test API
curl http://localhost:8000/

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 3}'

# View metrics
curl http://localhost:8000/api/metrics?days=7
```

---

## 💰 Costs

### Local Development
- **$0/month** - Everything runs locally

### GCP Production
- **Development**: ~$10-20/month (free tier + Vertex AI)
- **Production**: ~$45-85/month (self-hosted Elasticsearch)
- **vs Elastic Cloud**: Save 50-70%

See [COST_COMPARISON.md](COST_COMPARISON.md) for detailed breakdown.

---

## 🔐 Security

**For production:**
- Disable public access (set `allow_unauthenticated = false`)
- Enable Cloud IAP or custom authentication
- Rotate credentials regularly
- Use VPC Service Controls
- Enable audit logging

See [GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md) for security hardening.

---

**Add knowledge sources:**
- Go to Admin tab in UI
- Add URL patterns (e.g., `https://docs.python.org/3/*`)
- Configure crawl depth and limits
- Trigger ingestion and monitor progress

**Query your knowledge:**
- Go to Search tab
- Ask natural language questions
- Get answers augmented with your internal docs
- Provide feedback with 👍/👎

---

## 📝 License

MIT License

---

**Need help?** See [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md) or [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md)