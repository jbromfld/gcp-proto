# GCP Deployment Guide

Complete guide for deploying the RAG system to Google Cloud Platform.

## Quick Start (5 Minutes)

```bash
# 1. Configure
cp gcp-configs/env.template .env.gcp
vim .env.gcp  # Set GCP_PROJECT_ID

# 2. Setup GCP
./setup-gcp.sh

# 3. Deploy with Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars  # Set project_id
terraform init
terraform apply

# 4. Access
terraform output ui_url
```

Done! Your system is live in the cloud. 🚀

---

## Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
┌──────▼────────────────────────────────────────┐
│  Cloud Run (UI)                               │
│  Streamlit on port 8501                       │
└──────┬────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────┐
│  Cloud Run (API)                              │
│  FastAPI on port 8000                         │
└──┬────────────┬────────────────────┬──────────┘
   │            │                    │
   ▼            ▼                    ▼
┌─────────┐ ┌──────────┐    ┌────────────────┐
│  GCE    │ │ Vertex   │    │  Cloud Run     │
│  Elast  │ │ AI       │    │  (ETL)         │
│  icsear │ │ Gemini+  │    │  Scheduled     │
│  ch     │ │ Embed    │    │  via Scheduler │
└─────────┘ └──────────┘    └────────────────┘
```

**Components:**
- **Cloud Run (API, UI, ETL)**: Serverless containers, auto-scaling
- **GCE Elasticsearch**: Self-hosted vector store (~$25-30/mo)
- **Vertex AI**: Gemini Pro LLM + text-embedding-004
- **VPC Connector**: Connects Cloud Run → GCE
- **Cloud Scheduler**: Triggers daily ETL job
- **Artifact Registry**: Docker image storage
- **Secret Manager**: Credentials (optional)

---

## Prerequisites

### Required Accounts & Tools

1. **GCP Account** with billing enabled
   - Free tier covers development!
   - Go to: https://console.cloud.google.com

2. **gcloud CLI** installed
   ```bash
   # macOS
   brew install --cask google-cloud-sdk
   
   # Or download from:
   # https://cloud.google.com/sdk/docs/install
   ```

3. **Terraform** (recommended)
   ```bash
   brew install terraform
   ```

4. **Git** (for version control)
   ```bash
   git --version
   ```

### Authenticate

```bash
# 1. Login to GCP
gcloud auth login

# 2. Set application default credentials (for Terraform/API)
gcloud auth application-default login

# 3. Create or select project
gcloud projects create my-rag-project --name="RAG System"
# Or: gcloud config set project existing-project-id

# 4. Enable billing
# Visit: https://console.cloud.google.com/billing
# Link your project to a billing account
```

---

## Step-by-Step Deployment

### Step 1: Configure Environment

```bash
# Copy template
cp gcp-configs/env.template .env.gcp

# Edit with your settings
vim .env.gcp
```

**Minimal required settings:**
```bash
GCP_PROJECT_ID=your-project-id    # Your GCP project
GCP_REGION=us-central1             # Or your preferred region

# Elasticsearch option (choose one):
USE_GCE_ELASTICSEARCH=true         # Self-hosted (FREE or $25/mo) ✅ Recommended
# OR
# USE_GCE_ELASTICSEARCH=false
# ELASTICSEARCH_URL=https://...    # Elastic Cloud ($95/mo)
# ELASTICSEARCH_PASSWORD=...
```

**Full configuration:**
```bash
# GCP Settings
GCP_PROJECT_ID=my-rag-project-12345
GCP_REGION=us-central1

# Elasticsearch (Self-hosted on GCE - Recommended)
USE_GCE_ELASTICSEARCH=true
ELASTICSEARCH_MACHINE_TYPE=e2-medium  # 4GB RAM, ~$25/mo (or e2-micro FREE but limited)
ELASTICSEARCH_DISK_SIZE_GB=50
ELASTICSEARCH_USE_PREEMPTIBLE=false   # Set true for 60% discount (dev only)

# Deployment
USE_TERRAFORM=true
ALLOW_UNAUTHENTICATED=true  # Set false for production

# Knowledge sources (optional - can manage via Admin UI)
SCRAPE_URLS=https://docs.python.org/3/tutorial/,https://fastapi.tiangolo.com/
```

### Step 2: Run Setup Script

```bash
chmod +x setup-gcp.sh
./setup-gcp.sh
```

This will:
1. ✅ Enable required GCP APIs (Cloud Run, Vertex AI, etc.)
2. ✅ Create Artifact Registry for Docker images
3. ✅ Create service account with IAM permissions
4. ✅ Setup Secret Manager (if using Elastic Cloud)
5. ✅ Create Terraform state bucket

**Output:**
```
✓ APIs enabled
✓ Artifact Registry ready  
✓ Service account created: rag-service@PROJECT-ID.iam.gserviceaccount.com
✓ IAM permissions granted
✓ Terraform state bucket ready
```

### Step 3: Deploy Infrastructure

**Using Terraform (Recommended):**

```bash
cd terraform

# Create configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your project ID
vim terraform.tfvars
```

**terraform.tfvars:**
```hcl
project_id = "your-project-id"
region     = "us-central1"

# Use self-hosted Elasticsearch (cost-effective)
use_gce_elasticsearch      = true
use_vpc_connector          = true  # Required for Cloud Run → GCE
elasticsearch_machine_type = "e2-medium"  # 4GB RAM
elasticsearch_disk_size_gb = 50

# Resource allocation
api_cpu          = "2"
api_memory       = "4Gi"
api_min_instances = 0
api_max_instances = 10

# Security
allow_unauthenticated = true  # Set false for production
```

**Deploy:**
```bash
# Initialize (first time only)
terraform init

# Review what will be created
terraform plan

# Deploy everything!
terraform apply

# Get service URLs
terraform output
```

**What gets created:**
- Elasticsearch VM (e2-medium with persistent disk)
- VPC Connector (for Cloud Run → GCE communication)
- Cloud Run services (API, UI, ETL)
- Cloud Scheduler (daily ETL job)
- IAM bindings
- Firewall rules

### Step 4: Build & Deploy Code

```bash
# Build Docker images on GCP (handles AMD64 architecture)
gcloud builds submit --config cloudbuild.yaml

# This takes ~15-20 minutes and builds:
# - rag-api:latest
# - rag-ui:latest
# - rag-etl:latest
```

**If images already exist, force Cloud Run to update:**
```bash
cd terraform
terraform apply  # Pulls latest images
```

### Step 5: Verify Deployment

```bash
# Get URLs
terraform output

# Test API
API_URL=$(terraform output -raw api_url)
curl $API_URL/

# Test query
curl -X POST $API_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "top_k": 3}'

# Open UI
UI_URL=$(terraform output -raw ui_url)
open $UI_URL
```

**Expected output:**
```json
{
  "answer": "Python is a high-level programming language...",
  "retrieved_docs": [...],
  "query_id": "...",
  "latency_ms": 1234
}
```

---

## Cost Breakdown

### Development (Minimal Cost)

| Service | Config | Monthly Cost |
|---------|--------|--------------|
| **Elasticsearch** | e2-micro (1GB RAM) | **$0** (free tier) |
| Cloud Run (all 3) | Min=0, scales to zero | **$0-5** (free tier) |
| Persistent Disk | 30GB | **$0** (free tier) |
| VPC Connector | e2-micro | **$8** |
| Vertex AI | <1000 queries/month | **$2-5** |
| **TOTAL** | | **~$10-18/month** |

### Production (Low Cost)

| Service | Config | Monthly Cost |
|---------|--------|--------------|
| **Elasticsearch** | e2-medium (4GB RAM) | **$25** |
| Cloud Run API | 2CPU, 4GB, baseline traffic | **$10-15** |
| Cloud Run UI | 1CPU, 2GB | **$5-10** |
| Cloud Run ETL | Scheduled (1x/day) | **$1** |
| Persistent Disk | 50GB SSD | **$10** |
| VPC Connector | e2-micro | **$8** |
| Vertex AI | 1K-5K queries/day | **$10-30** |
| **TOTAL** | | **~$70-100/month** |

**vs Elastic Cloud Alternative:**
- Elastic Cloud Minimum: **$95/month** (just Elasticsearch!)
- **Savings with GCE**: **50-70%**

### Production (High Scale)

| Service | Config | Monthly Cost |
|---------|--------|--------------|
| Elasticsearch | e2-standard-4 (16GB RAM) | **$120** |
| Cloud Run | Higher limits, min instances=1 | **$50-100** |
| Vertex AI | 10K+ queries/day | **$100-200** |
| **TOTAL** | | **~$270-420/month** |

### Cost Optimization Tips

```bash
# 1. Use preemptible VMs for development (60% discount)
elasticsearch_use_preemptible = true

# 2. Set Cloud Run min instances to 0 (scales to zero)
api_min_instances = 0

# 3. Use smaller Elasticsearch VM for light traffic
elasticsearch_machine_type = "e2-small"  # $15/mo

# 4. Reduce log retention
gcloud logging sinks update ...

# 5. Use committed use discounts for stable workloads
```

---

## Monitoring & Maintenance

### View Logs

```bash
# Cloud Run logs (last hour)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api" \
  --limit=50 --format=json

# Elasticsearch VM logs
gcloud compute ssh elasticsearch --zone=us-central1-a --command="sudo docker logs elasticsearch"

# Real-time logs
gcloud alpha logging tail \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api"
```

### Metrics Dashboard

**Via UI:**
- Open your UI → Metrics tab
- Shows: queries, feedback, latency, popular queries

**Via Cloud Console:**
```bash
open "https://console.cloud.google.com/run?project=PROJECT-ID"
```

### Alerting

Set up alerts for:
- High error rates
- Slow responses (p95 > 3s)
- Low feedback scores (<70% thumbs up)
- Elasticsearch disk usage (>80%)

```bash
# Example: Alert on high error rate
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL-ID \
  --display-name="RAG API Errors" \
  --condition-threshold-value=10 \
  --condition-threshold-duration=300s
```

### Health Checks

```bash
# API health
curl https://your-api-url.run.app/

# Elasticsearch health
gcloud compute ssh elasticsearch --zone=us-central1-a \
  --command="curl -s localhost:9200/_cluster/health"

# Full system test
# Use Admin UI → System Health tab
```

---

## Updating & Scaling

### Deploy Updates

**Method 1: Terraform (Recommended)**
```bash
# Make code changes
git commit -am "Add feature X"

# Rebuild images
gcloud builds submit --config cloudbuild.yaml

# Deploy
cd terraform && terraform apply
```

**Method 2: Direct Update**
```bash
# Build new image
gcloud builds submit --config cloudbuild.yaml

# Update service
gcloud run deploy rag-api \
  --region=us-central1 \
  --image=us-central1-docker.pkg.dev/PROJECT-ID/rag-system/rag-api:latest
```

### Scale Services

**Increase capacity:**
```bash
# Via Terraform (edit terraform.tfvars)
api_min_instances = 1      # Keep 1 instance always warm
api_max_instances = 20     # Allow up to 20 instances
api_cpu = "4"              # More CPU per instance
api_memory = "8Gi"         # More RAM

terraform apply
```

**Or directly:**
```bash
gcloud run services update rag-api \
  --region=us-central1 \
  --min-instances=1 \
  --max-instances=20 \
  --cpu=4 \
  --memory=8Gi
```

### Upgrade Elasticsearch

```bash
# Via Terraform (edit terraform.tfvars)
elasticsearch_machine_type = "e2-standard-2"  # 8GB RAM, $50/mo

terraform apply

# Elasticsearch VM will be upgraded (brief downtime)
```

---

## Security Hardening

### Production Checklist

- [ ] Disable public access: `allow_unauthenticated = false`
- [ ] Enable Cloud IAP or custom authentication
- [ ] Use custom domain with SSL
- [ ] Enable VPC Service Controls
- [ ] Rotate credentials regularly
- [ ] Enable audit logging
- [ ] Set up Cloud Armor (WAF)
- [ ] Use least-privilege IAM roles
- [ ] Enable Secret Manager for all secrets
- [ ] Set up backup/disaster recovery

### Enable Authentication

```bash
# Require authentication for UI
gcloud run services update rag-ui \
  --region=us-central1 \
  --no-allow-unauthenticated

# Grant access to specific users
gcloud run services add-iam-policy-binding rag-ui \
  --region=us-central1 \
  --member='user:jane@example.com' \
  --role='roles/run.invoker'

# Or grant to a group
gcloud run services add-iam-policy-binding rag-ui \
  --region=us-central1 \
  --member='group:team@example.com' \
  --role='roles/run.invoker'
```

### Custom Domain

```bash
# 1. Verify domain ownership in Cloud Console
# 2. Map domain to service
gcloud beta run domain-mappings create \
  --service=rag-ui \
  --domain=rag.yourdomain.com \
  --region=us-central1

# 3. Update DNS records (shown in output)
```

---

## Troubleshooting

### Common Issues & Solutions

**1. "Container failed to start"**

```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=20

# Common causes:
# - Wrong project ID in Vertex AI config
# - Elasticsearch connection timeout
# - Missing environment variables

# Verify environment variables:
gcloud run services describe rag-api --region=us-central1 --format=yaml | grep -A 20 env
```

**2. "Cannot connect to Elasticsearch"**

```bash
# Check if VPC connector exists
gcloud compute networks vpc-access connectors list --region=us-central1

# If missing, enable in terraform.tfvars:
use_vpc_connector = true
terraform apply

# Test from Cloud Shell
gcloud compute ssh elasticsearch --zone=us-central1-a \
  --command="curl localhost:9200/_cluster/health"
```

**3. "Vertex AI permission denied"**

```bash
# Check if API is enabled
gcloud services list --enabled | grep aiplatform

# Enable if needed
gcloud services enable aiplatform.googleapis.com

# Check service account permissions
gcloud projects get-iam-policy PROJECT-ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:rag-service"

# Should have: roles/aiplatform.user
```

**4. "Out of memory" (Elasticsearch)**

```bash
# Check current size
gcloud compute instances describe elasticsearch \
  --zone=us-central1-a \
  --format='value(machineType)'

# Upgrade to larger instance
gcloud compute instances set-machine-type elasticsearch \
  --zone=us-central1-a \
  --machine-type=e2-standard-2

# Or update Terraform
elasticsearch_machine_type = "e2-standard-2"
terraform apply
```

**5. "Build timeout" or "exec format error"**

```bash
# Always build on GCP (not locally) to ensure correct architecture
gcloud builds submit --config cloudbuild.yaml

# Don't use:
# docker build ... && docker push  # ❌ Wrong architecture (ARM vs AMD64)
```

---

## Advanced Configuration

### Multi-Region Deployment

Deploy to multiple regions for high availability:

```bash
# Deploy to us-central1 (primary)
cd terraform && terraform apply

# Deploy to europe-west1 (secondary)
terraform workspace new europe
terraform apply -var="region=europe-west1"

# Use Cloud Load Balancer to route traffic
```

### CI/CD Pipeline

**Connect Cloud Build to GitHub:**

```bash
# 1. Connect repository
gcloud beta builds triggers create github \
  --repo-name=gcp-proto \
  --repo-owner=your-github-username \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml

# 2. Now every push to main triggers:
#    - Build Docker images
#    - Push to Artifact Registry
#    - (Optional) Auto-deploy via terraform

# 3. Add auto-deploy to cloudbuild.yaml:
#    - Uncomment deployment steps
#    - Add terraform apply step
```

### Custom LLM Models

Use different Vertex AI models:

```bash
# Edit terraform/cloud_run.tf, add env var:
env {
  name  = "VERTEX_LLM_MODEL"
  value = "gemini-1.5-pro"  # More capable, slower
}

# Or use different embedding model:
env {
  name  = "VERTEX_EMBEDDING_MODEL" 
  value = "textembedding-gecko@003"
}
```

---

## Maintenance

### Regular Tasks

**Weekly:**
- Review metrics in UI
- Check low-rated queries
- Add new knowledge sources

**Monthly:**
- Review Cloud Run logs for errors
- Check Elasticsearch disk usage
- Update dependencies (rebuild images)
- Review costs in Cloud Console

**Quarterly:**
- Evaluate search quality (see SIMILARITY_SEARCH_GUIDE.md)
- Consider model upgrades
- Review and optimize costs

### Backup Strategy

**Elasticsearch backups:**
```bash
# Option 1: Snapshot to GCS bucket
# Configure on Elasticsearch VM

# Option 2: Re-ingest from sources (preferred)
# Your sources are tracked, so just re-run ingestion

# Option 3: Export to JSON
gcloud compute ssh elasticsearch --zone=us-central1-a \
  --command="curl -X GET localhost:9200/knowledge_base/_search?size=10000" > backup.json
```

**Infrastructure as Code:**
```bash
# Your infrastructure is version controlled!
git add terraform/
git commit -m "Infrastructure snapshot"

# To restore: just run terraform apply
```

### Disaster Recovery

**Scenario: Elasticsearch VM crashes**
```bash
# 1. Terraform will recreate it
cd terraform && terraform apply

# 2. Re-ingest sources (tracked in knowledge_sources index)
# Use Admin UI → Sources → Re-ingest all
```

**Scenario: Accidentally deleted Cloud Run service**
```bash
cd terraform && terraform apply  # Recreates everything
```

---

## Performance Optimization

### Reduce Cold Start Time

Cloud Run cold starts can be slow. Optimize:

```bash
# 1. Keep min instances warm (costs more but faster)
api_min_instances = 1

# 2. Reduce image size (already optimized)
# Our images use slim Python base: 200-300MB

# 3. Use startup probes (already configured)
# Allows longer initialization time
```

### Improve Query Speed

```bash
# 1. Use faster embedding model
# Already using text-embedding-004 (fast)

# 2. Reduce chunk count
# Edit relevance threshold or top_k parameter

# 3. Add caching (future enhancement)
# Cache common queries in Redis/Memorystore
```

### Handle High Traffic

```bash
# 1. Increase max instances
api_max_instances = 50

# 2. Add more CPU/RAM per instance
api_cpu = "4"
api_memory = "8Gi"

# 3. Use Cloud CDN for UI static assets

# 4. Consider Elasticsearch cluster (multiple nodes)
```

---

## Where Credentials Are Used

### GCP Project Authentication

**Local Development:**
```bash
# Uses your user account
gcloud auth application-default login
```

**Cloud Run (Production):**
```bash
# Uses service account automatically
# No credentials needed in code!
# Service account: rag-service@PROJECT-ID.iam.gserviceaccount.com
```

### Elasticsearch Authentication

**With GCE (Self-Hosted):**
```bash
# No authentication needed!
# Internal communication within VPC
# Cloud Run → VPC Connector → GCE (private IP)
```

**With Elastic Cloud:**
```bash
# Stored in Secret Manager
# Cloud Run reads from secrets automatically
# See terraform/secrets.tf for configuration
```

### Vertex AI Authentication

```bash
# No API keys needed!
# Uses Workload Identity:
# - Service account has roles/aiplatform.user
# - Automatic authentication via metadata server
```

**In code (automatic):**
```python
import vertexai

# Automatically uses:
# - Local: Application Default Credentials
# - Cloud Run: Service account via Workload Identity
vertexai.init(project=PROJECT_ID, location=REGION)
```

---

## Rollback & Recovery

### Rollback to Previous Version

```bash
# List revisions
gcloud run revisions list --service=rag-api --region=us-central1

# Route traffic to older revision
gcloud run services update-traffic rag-api \
  --region=us-central1 \
  --to-revisions=rag-api-00003-xyz=100

# Or use Terraform (revert code, apply)
git checkout previous-commit
gcloud builds submit --config cloudbuild.yaml
cd terraform && terraform apply
```

### Complete Teardown

```bash
# Delete everything
cd terraform && terraform destroy

# Confirm when prompted
# This deletes:
# - All Cloud Run services
# - Elasticsearch VM and disk
# - VPC connector
# - (Keeps: Artifact Registry, Service Account)

# To delete those too:
gcloud artifacts repositories delete rag-system --location=us-central1
gcloud iam service-accounts delete rag-service@PROJECT-ID.iam.gserviceaccount.com
```

---

## Production Best Practices

### 1. Use Terraform for Everything

```yaml
Benefits:
  - Version controlled infrastructure
  - Reproducible deployments
  - Easy rollback
  - Team collaboration
  - Disaster recovery
```

### 2. Enable Monitoring & Alerting

```bash
# Create uptime check
gcloud monitoring uptime create https://your-api-url.run.app/ \
  --display-name="RAG API Uptime"

# Alert on downtime
gcloud alpha monitoring policies create ...
```

### 3. Set Up CI/CD

```yaml
Workflow:
  1. Push to GitHub
  2. Cloud Build triggers
  3. Builds Docker images
  4. Runs tests
  5. Deploys to staging
  6. Manual approval
  7. Deploys to production
```

### 4. Use Separate Environments

```bash
# Development
terraform workspace new dev
terraform apply -var="environment=dev"

# Staging
terraform workspace new staging
terraform apply -var="environment=staging"

# Production
terraform workspace new prod
terraform apply -var="environment=prod"
```

---

## Support & Resources

### Documentation
- **Local Development**: `docs/LOCAL_DEVELOPMENT.md`
- **Similarity Search Guide**: `docs/SIMILARITY_SEARCH_GUIDE.md`
- **Cost Comparison**: `COST_COMPARISON.md`

### GCP Resources
- Cloud Run Docs: https://cloud.google.com/run/docs
- Vertex AI Docs: https://cloud.google.com/vertex-ai/docs
- Terraform Google Provider: https://registry.terraform.io/providers/hashicorp/google

### Getting Help

1. Check logs (see Monitoring section above)
2. Review error messages in Cloud Console
3. Verify configuration in terraform.tfvars
4. Test locally first (see LOCAL_DEVELOPMENT.md)

---

**Ready to deploy?** Run: `./setup-gcp.sh`

