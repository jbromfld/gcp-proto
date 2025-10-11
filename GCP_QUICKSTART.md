# GCP Deployment - Quick Start Guide

## ⚡ 5-Minute Deploy

### Prerequisites
- GCP account with billing
- `gcloud` CLI installed
- Elastic Cloud account (or existing Elasticsearch)

### Steps

**1. Configure**
```bash
# Copy template
cp gcp-configs/env.template .env.gcp

# Edit these variables:
nano .env.gcp
  GCP_PROJECT_ID=your-project-id
  GCP_REGION=us-central1
  ELASTICSEARCH_URL=https://your-elastic-url:9243
  ELASTICSEARCH_PASSWORD=your-password
```

**2. Setup GCP**
```bash
./setup-gcp.sh
```

**3. Deploy**
```bash
./deploy.sh all
```

**4. Access**
```bash
# Get URLs
gcloud run services list --region=us-central1

# Open UI
UI_URL=$(gcloud run services describe rag-ui --region=us-central1 --format='value(status.url)')
open $UI_URL
```

## 📍 Where to Add Credentials

### 1. GCP Project Credentials

**Authentication:**
```bash
# For local development/deployment
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR-PROJECT-ID
```

No API keys needed! Uses:
- **Local**: Your user credentials via `gcloud auth`
- **Cloud Run**: Automatic via service account

### 2. Elasticsearch Credentials

**Stored in Secret Manager (secure):**

```bash
# Set URL
echo 'https://your-deployment.es.us-central1.gcp.cloud.es.io:9243' | \
    gcloud secrets versions add elasticsearch-url --data-file=-

# Set password
echo 'your-elastic-password' | \
    gcloud secrets versions add elasticsearch-password --data-file=-
```

**Where to get Elastic Cloud credentials:**
1. Go to https://cloud.elastic.co/
2. Create deployment → Choose GCP
3. Copy the "Cloud ID" or "Elasticsearch endpoint"
4. Copy the "elastic" user password

### 3. Vertex AI (No Credentials Needed!)

Vertex AI uses **Workload Identity** - no API keys required!

The service account `rag-service@PROJECT-ID.iam.gserviceaccount.com` automatically has access via IAM roles.

## Cost Breakdown

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Cloud Run API | ~100K requests | $5-15 |
| Cloud Run UI | ~50K requests | $3-8 |
| Elastic Cloud | Basic tier | $95 |
| Vertex AI Gemini Flash | 10K queries | $5-10 |
| Vertex AI Embeddings | 100K docs | $2-5 |
| **TOTAL** | | **~$110-133** |

**Free tier:**
- Cloud Run: 2M requests/month free
- Secret Manager: 6 secrets free
- Artifact Registry: 0.5GB free

## Terraform Option (IaC)

```bash
# Edit with your values
cd terraform
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars

# Deploy infrastructure
terraform init
terraform plan
terraform apply

# Get outputs
terraform output
```

## Monitoring

**View logs:**
```bash
gcloud run services logs read rag-api --region=us-central1 --limit=50
```

**Metrics dashboard:**
- Visit your UI → Metrics tab
- Or Cloud Console → Cloud Run → rag-api → Metrics

## Troubleshooting

**"Permission denied"**
```bash
# Check service account
gcloud iam service-accounts describe rag-service@PROJECT-ID.iam.gserviceaccount.com
```

**"Cannot connect to Elasticsearch"**
```bash
# Test from Cloud Shell
curl -u elastic:PASSWORD https://your-elastic-url:9243/_cluster/health
```

**"Vertex AI quota exceeded"**
- Visit: https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas
- Request quota increase

## Next Steps

1. **Add more documents**: Update `SCRAPE_URLS` in .env.gcp
2. **Custom domain**: Map your domain to Cloud Run
3. **Enable auth**: Set `ALLOW_UNAUTHENTICATED=false`
4. **CI/CD**: Connect Cloud Build to GitHub
5. **Scale**: Increase `API_MAX_INSTANCES` for more traffic

## Full Documentation

See `DEPLOY_GCP.md` for comprehensive deployment guide.

