# Setup Instructions - Before You Begin

## Quick Start Checklist

Before running `./setup-gcp.sh`, you need:

### ✅ 1. GCP Account & Project

**Create a GCP project:**
```bash
# Option A: Via web console
# Go to: https://console.cloud.google.com/projectcreate
# Name: "RAG Knowledge System"
# Note the Project ID (e.g., rag-system-12345)

# Option B: Via CLI
gcloud projects create my-rag-project --name="RAG Knowledge System"
```

**Enable billing:**
- Go to: https://console.cloud.google.com/billing
- Link your project to a billing account
- **Note**: Free tier covers development costs!

### ✅ 2. Authenticate gcloud CLI

```bash
# Login to GCP
gcloud auth login

# Set application default credentials (for Terraform/API access)
gcloud auth application-default login

# Set your project
gcloud config set project YOUR-PROJECT-ID
```

### ✅ 3. Configure Environment

```bash
# Copy the template
cp gcp-configs/env.template .env.gcp

# Edit with your settings
nano .env.gcp  # or vim, code, etc.
```

**Required settings to update:**

| Setting | What to Change | Example |
|---------|----------------|---------|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-rag-project-12345` |
| `GCP_REGION` | Your preferred region | `us-central1` (default) |
| `USE_GCE_ELASTICSEARCH` | Use free GCE option | `true` (recommended) |
| `ELASTICSEARCH_MACHINE_TYPE` | VM size | `e2-micro` (FREE) or `e2-medium` ($25/mo) |

**Example `.env.gcp` file:**
```bash
GCP_PROJECT_ID=my-rag-project-12345
GCP_REGION=us-central1
USE_GCE_ELASTICSEARCH=true
ELASTICSEARCH_MACHINE_TYPE=e2-micro  # FREE!
ELASTICSEARCH_DISK_SIZE_GB=30
USE_TERRAFORM=true
SCRAPE_URLS=https://docs.python.org/3/tutorial/,https://fastapi.tiangolo.com/
ALLOW_UNAUTHENTICATED=true
```

### ✅ 4. Run Setup

```bash
# Make scripts executable
chmod +x setup-gcp.sh deploy.sh

# Run initial setup
./setup-gcp.sh
```

This will:
- ✅ Enable required GCP APIs (Cloud Run, Vertex AI, etc.)
- ✅ Create Artifact Registry for Docker images
- ✅ Create service account with permissions
- ✅ Set up Secret Manager (if using Elastic Cloud)

### ✅ 5. Deploy Services

**Option A: Using Terraform (Recommended)**
```bash
cd terraform

# Create config from template
cp terraform.tfvars.example terraform.tfvars

# Edit with your project ID
nano terraform.tfvars

# Deploy everything
terraform init
terraform apply

# Get service URLs
terraform output
```

**Option B: Using deploy script**
```bash
./deploy.sh all
```

## Cost Expectations

### Development (FREE Tier)
- ✅ Elasticsearch: e2-micro VM (FREE)
- ✅ Cloud Run: FREE tier (2M requests/mo)
- ✅ Disk: 30GB (FREE tier)
- ⚠️ Vertex AI: ~$0.50-1/mo (no free tier)
- **Total: ~$1/mo**

### Production (Low Cost)
- Elasticsearch: e2-medium VM ($25/mo)
- Cloud Run: $10-20/mo
- Vertex AI: $5-10/mo (1K queries/day)
- **Total: ~$40-55/mo**

## Common Issues

### "Project not found"
```bash
# List your projects
gcloud projects list

# Set the correct project
gcloud config set project YOUR-PROJECT-ID
```

### "Billing not enabled"
- Visit: https://console.cloud.google.com/billing
- Link your project to a billing account

### "API not enabled"
The setup script will enable required APIs, but you can manually enable:
```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com
```

### "Permission denied"
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

## Next Steps After Setup

1. **Verify Elasticsearch**: Check that it's running
   ```bash
   # If using Terraform
   terraform output elasticsearch_internal_ip
   
   # Test connection
   curl http://ELASTICSEARCH_IP:9200
   ```

2. **Deploy services**: Run `./deploy.sh all`

3. **Test the system**: Access the UI URL from deployment output

4. **Index your data**: Trigger the ETL pipeline to scrape and index documents

5. **Query**: Try asking questions through the UI!

## Full Documentation

- **Quick Start**: `GCP_QUICKSTART.md` (5-minute guide)
- **Complete Guide**: `DEPLOY_GCP.md` (all options)
- **Elasticsearch Setup**: `docs/ELASTICSEARCH_GCE.md` (detailed)
- **Cost Analysis**: `COST_COMPARISON.md` (pricing options)

## Support

If you encounter issues:
1. Check logs: `gcloud logging read --limit 50`
2. Review the troubleshooting section in `DEPLOY_GCP.md`
3. Verify your `.env.gcp` settings
4. Ensure all prerequisite steps are complete

Ready to proceed? Run: `./setup-gcp.sh`

