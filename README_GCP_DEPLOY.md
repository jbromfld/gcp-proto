# 🚀 GCP Deployment Files

This directory contains everything needed to deploy the RAG system to Google Cloud Platform.

## 📁 Files Created

### Terraform (Infrastructure as Code)
```
terraform/
├── main.tf              # Main configuration, APIs, service accounts
├── cloud_run.tf         # Cloud Run services (API, UI, ETL)
├── secrets.tf           # Secret Manager setup
├── variables.tf         # Input variables
├── outputs.tf           # Deployment outputs
└── terraform.tfvars.example  # Configuration template
```

### Deployment Scripts
```
setup-gcp.sh            # Initial GCP setup (run once)
deploy.sh               # Build & deploy services
cloudbuild.yaml         # Cloud Build CI/CD configuration
```

### Configuration
```
gcp-configs/
├── env.template        # Environment variables template
├── api.yaml           # Cloud Run API service config  
└── ui.yaml            # Cloud Run UI service config

.env.gcp               # Your secrets (gitignored)
```

### Documentation
```
DEPLOY_GCP.md          # Comprehensive deployment guide
GCP_QUICKSTART.md      # 5-minute quick start
docs/ELASTICSEARCH_SETUP.md  # Elasticsearch options
```

### Code Updates
```
elasticsearch_client.py  # Elasticsearch auth support
rag_api.py              # Updated to use auth client
```

## 🎯 Quick Start

```bash
# 1. Configure
cp gcp-configs/env.template .env.gcp
# Edit: GCP_PROJECT_ID, ELASTICSEARCH_URL, ELASTICSEARCH_PASSWORD

# 2. Setup
./setup-gcp.sh

# 3. Deploy
./deploy.sh all

# 4. Access
gcloud run services list
```

## 🔐 Where to Add Credentials

### 1. GCP Project & Authentication
```bash
# Local development
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR-PROJECT-ID

# Production: Automatic via service account
```

### 2. Elasticsearch (Stored in Secret Manager)
```bash
echo 'https://your-elastic-url:9243' | \
    gcloud secrets versions add elasticsearch-url --data-file=-

echo 'your-password' | \
    gcloud secrets versions add elasticsearch-password --data-file=-
```

### 3. Vertex AI
**No credentials needed!** Uses Workload Identity.

## ✅ Architecture Meets Requirements

| Requirement | Solution |
|-------------|----------|
| Elastic hybrid search | ✅ Vector (cosine) + BM25 keyword search |
| Google Cloud generative AI | ✅ Vertex AI Gemini + Embeddings |
| Conversational solution | ✅ RAG with multi-turn capability |
| Agent-based | ✅ Extendable to Gemini function calling |
| Transform data interaction | ✅ Natural language → precise answers |

## 💰 Cost Estimate

- **Development**: ~$100-120/mo (free tier coverage)
- **Production (1K queries/day)**: ~$110-135/mo  
- **Production (10K queries/day)**: ~$150-200/mo

## 📊 Deployment Options

| Method | Use Case | Time |
|--------|----------|------|
| `deploy.sh` | Quick deploy, testing | 10 min |
| Terraform | Production, IaC, teams | 15 min |
| Cloud Build | CI/CD, automation | 20 min setup |

## 🔧 What Was Automated

1. ✅ GCP API enablement
2. ✅ Service account creation + IAM
3. ✅ Artifact Registry setup
4. ✅ Secret Manager configuration
5. ✅ Cloud Run service deployment
6. ✅ Cloud Scheduler for ETL
7. ✅ Logging and monitoring
8. ✅ Auto-scaling configuration

## 📖 Documentation

- **Quick Start**: `GCP_QUICKSTART.md` (5 minutes)
- **Full Guide**: `DEPLOY_GCP.md` (complete reference)
- **Elasticsearch**: `docs/ELASTICSEARCH_SETUP.md`

## 🎓 Next Steps After Deployment

1. **Index your data**: Update `SCRAPE_URLS` and trigger ETL
2. **Custom domain**: Map your domain to Cloud Run UI
3. **Enable auth**: Set up Cloud IAP or OAuth
4. **Monitor**: Set up Cloud Monitoring alerts
5. **Scale**: Adjust min/max instances based on traffic

Need help? Check `DEPLOY_GCP.md` for troubleshooting.

