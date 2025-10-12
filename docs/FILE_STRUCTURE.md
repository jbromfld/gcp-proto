# File Structure & Configuration Guide

Quick reference for which files to use when.

## 📁 Environment Configuration Files

### Local Development

**Template:**
```
env.local.template          # Template with all local env vars
```

**Your file (gitignored):**
```
.env                        # Copy from env.local.template
                           # Used by docker-compose automatically
                           # Or source it for native Python
```

**Usage:**
```bash
# One-time:
cp env.local.template .env

# Then either:
docker-compose up -d        # Reads .env automatically
# OR
source .env.local.template  # For native Python
python rag_api.py
```

### GCP Deployment

**Template:**
```
gcp-configs/env.template    # Template for GCP settings
```

**Your file (gitignored):**
```
.env.gcp                    # Copy from gcp-configs/env.template
                           # Optional reference file (Terraform uses terraform.tfvars)
```

**Usage:**
```bash
# Configure Terraform instead:
cd terraform
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars        # Set project_id, region, etc.
```

---

## 🚀 Startup Scripts

### Local

```bash
setup_local.sh              # One-time setup (install deps, start Elasticsearch)
start_local.sh              # Start API + UI (run every time)
```

### GCP

```bash
pause-gcp.sh                # Stop GCP resources for the week (~$16/mo)
resume-gcp.sh               # Restart GCP resources in 60 seconds
# See docs/GCP_DEPLOYMENT.md for full deployment guide
```

---

## 📚 Documentation

```
docs/
├── LOCAL_DEVELOPMENT.md         # Complete local guide
├── GCP_DEPLOYMENT.md            # Complete GCP guide
├── SIMILARITY_SEARCH_GUIDE.md   # Testing & optimization
├── DEPLOYMENT_SUMMARY.md        # What we built
└── FILE_STRUCTURE.md            # This file
```

**When to read what:**
- **First time local setup**: `LOCAL_DEVELOPMENT.md`
- **Deploy to GCP**: `GCP_DEPLOYMENT.md`
- **Improve search quality**: `SIMILARITY_SEARCH_GUIDE.md`
- **Quick reference**: `DEPLOYMENT_SUMMARY.md`

---

## 🔧 Key Files

### Application Code

```
rag_api.py                  # FastAPI REST API
rag_ui.py                   # Streamlit UI
rag_service.py              # RAG orchestration
rag_sources.py              # Source management
rag_etl_pipeline.py         # Document processing
rag_embeddings.py           # Embedding providers
rag_llm_abstraction.py      # LLM providers
rag_evaluation.py           # Metrics & feedback
elasticsearch_client.py     # ES connection helper
```

### Deployment

```
docker-compose.yml          # Local Docker setup
Dockerfile.{api,ui,etl}     # Container definitions
cloudbuild.yaml             # GCP CI/CD
terraform/                  # Infrastructure as Code
```

---

## 🎯 Quick Decision Tree

**Want to run locally?**
```
→ cp env.local.template .env
→ ./setup_local.sh (one time)
→ ./start_local.sh (every time)
```

**Want to deploy to GCP?**
```
→ cd terraform
→ cp terraform.tfvars.example terraform.tfvars
→ Edit terraform.tfvars (set project_id)
→ terraform init && terraform apply
→ cd .. && gcloud builds submit --config cloudbuild.yaml
→ cd terraform && terraform apply  # Update services with new images
```

**Want to use docker-compose locally?**
```
→ cp env.local.template .env
→ docker-compose up -d
→ open http://localhost:8501
```

---

## ⚠️ Common Mistakes

**❌ Wrong env file for context:**
```bash
# Don't use gcp-configs/env.template for local dev
source gcp-configs/env.template  # ❌ Wrong!

# Use env.local.template instead
source env.local.template        # ✅ Correct
```

**❌ Forgetting to copy template:**
```bash
docker-compose up -d  # ❌ No .env file, uses defaults

cp env.local.template .env  # ✅ Create .env first
docker-compose up -d
```

**❌ Mixing local and GCP configs:**
```bash
# Don't put GCP_PROJECT_ID in .env (local)
# Don't put OLLAMA_URL in .env.gcp (GCP)
```

---

## 📝 Summary

| File | Purpose | Create From | Used By |
|------|---------|-------------|---------|
| `.env` | Local env vars | `env.local.template` | docker-compose, native Python |
| `.env.gcp` | GCP settings (optional) | `gcp-configs/env.template` | Reference only |
| `env.local.template` | Local template | Don't edit | Reference |
| `gcp-configs/env.template` | GCP template | Don't edit | Reference |
| `terraform/terraform.tfvars` | GCP deployment config | `terraform.tfvars.example` | Terraform (actual config) |

**Both .env and .env.gcp are gitignored** - safe to add your secrets!

