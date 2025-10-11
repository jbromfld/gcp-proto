# Deploying RAG System to Google Cloud Platform

This guide walks you through deploying the RAG Knowledge Search System to GCP using Cloud Run and Vertex AI.

## Architecture Overview

```
User → Cloud Load Balancer → Cloud Run (UI:8501)
                                    ↓
                            Cloud Run (API:8000)
                                    ↓
                        ┌───────────┴───────────┐
                        ↓                       ↓
                  Elastic Cloud          Vertex AI
                 (Hybrid Search)    (Gemini + Embeddings)
```

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Terraform** (optional, for IaC) - `brew install terraform`
4. **Docker** for building images
5. **Elastic Cloud account** (or GKE for self-hosted)

## Step 1: Initial GCP Setup

### 1.1 Create/Select GCP Project

```bash
# Create new project (or use existing)
gcloud projects create your-project-id --name="RAG System"

# Set as current project
gcloud config set project your-project-id

# Enable billing (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing
```

### 1.2 Configure Environment

```bash
# Copy environment template
cp gcp-configs/env.template .env.gcp

# Edit with your settings
vim .env.gcp
```

**Required settings:**
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: GCP region (e.g., `us-central1`)
- `ELASTICSEARCH_URL`: Your Elasticsearch endpoint
- `ELASTICSEARCH_PASSWORD`: Elasticsearch password

### 1.3 Run Setup Script

```bash
# Make scripts executable
chmod +x setup-gcp.sh deploy.sh

# Run initial setup
./setup-gcp.sh
```

This will:
- Enable required GCP APIs
- Create Artifact Registry repository
- Create service account with permissions
- Store Elasticsearch credentials in Secret Manager

## Step 2: Set Up Elasticsearch

### Option A: Elastic Cloud (Recommended)

1. **Sign up**: https://cloud.elastic.co/
2. **Create deployment**:
   - Choose "GCP" as cloud provider
   - Select same region as your Cloud Run services
   - Choose "Optimized for GCP" template
   - Select tier (starts at ~$95/mo)

3. **Get credentials**:
   ```bash
   # After deployment, copy the Cloud ID and password
   # Update secrets:
   echo 'https://your-deployment.es.REGION.gcp.cloud.es.io:9243' | \
       gcloud secrets versions add elasticsearch-url --data-file=-
   
   echo 'your-elastic-password' | \
       gcloud secrets versions add elasticsearch-password --data-file=-
   ```

### Option B: Self-Hosted on GKE

See `docs/ELASTICSEARCH_GKE.md` for detailed GKE setup.

## Step 3: Deploy Services

### Method 1: Using Deploy Script (Easiest)

```bash
# Build and deploy everything
./deploy.sh all

# Or step by step:
./deploy.sh build    # Build and push images
./deploy.sh deploy   # Deploy to Cloud Run
```

### Method 2: Using Terraform (Infrastructure as Code)

```bash
cd terraform

# Initialize Terraform
terraform init -backend-config="bucket=${GCP_PROJECT_ID}-terraform-state"

# Review plan
terraform plan \
    -var="project_id=${GCP_PROJECT_ID}" \
    -var="region=${GCP_REGION}" \
    -var="elasticsearch_url=${ELASTICSEARCH_URL}" \
    -var="elasticsearch_password=${ELASTICSEARCH_PASSWORD}"

# Apply infrastructure
terraform apply

# Get outputs
terraform output
```

### Method 3: Using Cloud Build (CI/CD)

```bash
# Submit build manually
gcloud builds submit --config cloudbuild.yaml

# Or connect to GitHub for automatic deployments
gcloud builds triggers create github \
    --repo-name=your-repo \
    --repo-owner=your-github \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

## Step 4: Verify Deployment

```bash
# Get service URLs
API_URL=$(gcloud run services describe rag-api --region=$GCP_REGION --format='value(status.url)')
UI_URL=$(gcloud run services describe rag-ui --region=$GCP_REGION --format='value(status.url)')

echo "API: $API_URL"
echo "UI:  $UI_URL"

# Test API
curl $API_URL/

# Test query
curl -X POST $API_URL/api/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is Python?", "max_results": 3}'

# Open UI in browser
open $UI_URL
```

## Step 5: Initial Data Load

```bash
# Trigger ETL manually
ETL_URL=$(gcloud run services describe rag-etl --region=$GCP_REGION --format='value(status.url)')

curl -X POST $ETL_URL/trigger \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)"
```

## Configuration Details

### Where to Add Credentials

#### 1. **GCP Project ID & Credentials**

**Local Development:**
```bash
# Authenticate with your user account
gcloud auth application-default login

# Set project
gcloud config set project YOUR-PROJECT-ID
```

**Production (Cloud Run):**
- Automatically uses service account: `rag-service@PROJECT-ID.iam.gserviceaccount.com`
- No manual credentials needed - handled by Workload Identity

#### 2. **Elasticsearch Credentials**

Stored in **Secret Manager** (secure, encrypted):

```bash
# Set Elasticsearch URL
echo 'https://your-elastic-url:9243' | \
    gcloud secrets versions add elasticsearch-url --data-file=-

# Set password
echo 'your-password' | \
    gcloud secrets versions add elasticsearch-password --data-file=-

# Verify
gcloud secrets versions access latest --secret=elasticsearch-url
```

#### 3. **Vertex AI Authentication**

No API keys needed! Uses:
- **Local**: Application Default Credentials (ADC)
- **Cloud Run**: Service Account with `roles/aiplatform.user`

## Environment Variables

Cloud Run services automatically get:

| Variable | Source | Purpose |
|----------|--------|---------|
| `GOOGLE_PROJECT_ID` | Environment | Your GCP project |
| `GOOGLE_REGION` | Environment | GCP region |
| `ELASTICSEARCH_URL` | Secret Manager | Elasticsearch endpoint |
| `ELASTICSEARCH_PASSWORD` | Secret Manager | Elasticsearch auth |
| `EMBEDDING_PROVIDER` | Environment | Set to `vertex` |
| `LLM_PROVIDER` | Environment | Set to `vertex` |

## Monitoring & Logs

### View Logs

```bash
# API logs
gcloud run services logs read rag-api --region=$GCP_REGION --limit=50

# UI logs
gcloud run services logs read rag-ui --region=$GCP_REGION --limit=50

# Or use Cloud Console
open "https://console.cloud.google.com/run?project=$GCP_PROJECT_ID"
```

### Metrics

Access metrics at: `https://YOUR-UI-URL/` → Metrics tab

Or view in Cloud Monitoring:
```bash
open "https://console.cloud.google.com/monitoring?project=$GCP_PROJECT_ID"
```

## Cost Optimization

### Cloud Run Costs

- **Minimum instances = 0**: Pay only when processing requests
- **Scale to zero**: No cost when idle
- **CPU throttling**: Enabled (reduces cost between requests)

### Vertex AI Costs

| Service | Model | Cost (estimate) |
|---------|-------|-----------------|
| LLM | Gemini Flash | $0.075/1M input tokens, $0.30/1M output |
| Embeddings | text-embedding-004 | $0.025/1M tokens |

Example: 1,000 queries/day ≈ $5-10/month

### Total Estimated Costs

| Component | Monthly Cost |
|-----------|--------------|
| Cloud Run (API+UI+ETL) | $10-30 |
| Elastic Cloud (Basic) | $95+ |
| Vertex AI (1K queries/day) | $5-10 |
| **Total** | **$110-135/mo** |

## Scaling

Cloud Run auto-scales based on:
- **CPU utilization** (default: 60%)
- **Request concurrency** (80 concurrent/instance)
- **Custom metrics** (optional)

Configure in terraform or via:
```bash
gcloud run services update rag-api \
    --region=$GCP_REGION \
    --min-instances=1 \
    --max-instances=20
```

## Security

### Production Checklist

- [ ] Set `ALLOW_UNAUTHENTICATED=false` in .env.gcp
- [ ] Configure Cloud IAP or Cloud Armor
- [ ] Enable VPC Service Controls
- [ ] Rotate Elasticsearch credentials regularly
- [ ] Enable audit logging
- [ ] Use custom domain with Cloud Load Balancer

### Enable Authentication

```bash
# Require authentication
gcloud run services update rag-ui \
    --region=$GCP_REGION \
    --no-allow-unauthenticated

# Grant access to specific users
gcloud run services add-iam-policy-binding rag-ui \
    --region=$GCP_REGION \
    --member='user:email@example.com' \
    --role='roles/run.invoker'
```

## Troubleshooting

### Common Issues

**1. "Permission denied" errors**
```bash
# Verify service account permissions
gcloud projects get-iam-policy $GCP_PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:rag-service@"
```

**2. "Cannot connect to Elasticsearch"**
```bash
# Test from Cloud Shell
curl -u elastic:PASSWORD https://your-elastic-url:9243/_cluster/health

# Check VPC connector (if using GKE)
gcloud compute networks vpc-access connectors describe rag-vpc-connector \
    --region=$GCP_REGION
```

**3. "Vertex AI quota exceeded"**
```bash
# Check quotas
gcloud alpha services quota list \
    --service=aiplatform.googleapis.com \
    --consumer=projects/$GCP_PROJECT_ID

# Request increase at:
# https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas
```

**4. Container fails to start**
```bash
# Check logs
gcloud run services logs read rag-api --region=$GCP_REGION --limit=100

# Check service details
gcloud run services describe rag-api --region=$GCP_REGION
```

## Updating the Application

### Update Code and Redeploy

```bash
# Make your changes
git commit -am "Update feature X"

# Rebuild and deploy
./deploy.sh all

# Or use Cloud Build (if connected to GitHub)
git push origin main  # Triggers automatic deployment
```

### Rolling Back

```bash
# List revisions
gcloud run revisions list --service=rag-api --region=$GCP_REGION

# Rollback to previous
gcloud run services update-traffic rag-api \
    --region=$GCP_REGION \
    --to-revisions=rag-api-00002-abc=100
```

## Cleanup

### Delete Everything

```bash
# Using Terraform
cd terraform && terraform destroy

# Or manually
gcloud run services delete rag-api --region=$GCP_REGION
gcloud run services delete rag-ui --region=$GCP_REGION
gcloud run services delete rag-etl --region=$GCP_REGION
gcloud artifacts repositories delete rag-system --location=$GCP_REGION
gcloud iam service-accounts delete rag-service@$GCP_PROJECT_ID.iam.gserviceaccount.com
```

## Next Steps

1. **Custom Domain**: Map your domain to Cloud Run
2. **CI/CD**: Connect Cloud Build to GitHub
3. **Monitoring**: Set up Cloud Monitoring alerts
4. **Authentication**: Enable Cloud IAP
5. **CDN**: Add Cloud CDN for UI static assets
6. **Multi-region**: Deploy to multiple regions for HA

## Support

- GCP Documentation: https://cloud.google.com/run/docs
- Elastic Cloud: https://www.elastic.co/guide/en/cloud/current
- Vertex AI: https://cloud.google.com/vertex-ai/docs

For issues, check logs and monitoring dashboards first.

