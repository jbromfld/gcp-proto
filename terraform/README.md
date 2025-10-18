# Terraform Deployment Guide

## Quick Start (Recommended Order)

### Step 1: Run setup-gcp.sh First
```bash
# From project root
./setup-gcp.sh
```

This creates:
- ✅ Required GCP APIs
- ✅ Service account
- ✅ Artifact Registry
- ✅ **Terraform state bucket** (`PROJECT-ID-terraform-state`)

### Step 2: Configure Terraform
```bash
cd terraform

# Copy template
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
nano terraform.tfvars
```

**Required settings:**
```hcl
project_id = "gcp-poc-474818"  # Your project ID
region     = "us-central1"

# GCE Elasticsearch (FREE or ~$30/mo)
use_gce_elasticsearch      = true
elasticsearch_machine_type = "e2-micro"  # FREE tier
elasticsearch_disk_size_gb = 30
```

### Step 3: Deploy Infrastructure
```bash
# Initialize Terraform (uses local state)
terraform init

# Review what will be created
terraform plan

# Deploy everything
terraform apply

# Get service URLs
terraform output
```

## Remote State (Optional)

The initial setup uses **local state** (stored in `terraform.tfstate`). This is fine for:
- ✅ Single developer
- ✅ POC/Demo projects
- ✅ Getting started quickly

### When to Use Remote State

Use remote state (GCS bucket) if:
- Team collaboration (multiple people running Terraform)
- Production environments
- Need state locking
- Want backup/versioning

### Migrate to Remote State

After your first successful `terraform apply`:

**1. Edit `main.tf`:**
```hcl
# Uncomment these lines in main.tf:
backend "gcs" {
  bucket = "gcp-poc-474818-terraform-state"  # Your project ID + suffix
  prefix = "rag-system/state"
}
```

**2. Migrate state:**
```bash
# Reinitialize with backend
terraform init -migrate-state

# Confirm migration when prompted
# Type: yes
```

**3. Verify:**
```bash
# Check state is in GCS
gsutil ls gs://gcp-poc-474818-terraform-state/rag-system/

# Your local terraform.tfstate will become a backup
```

## Troubleshooting

### "Bucket not found" error

**Problem:** Terraform backend looking for non-existent bucket

**Solution:**
```bash
# Option 1: Use local state (recommended for start)
# Comment out the backend "gcs" block in main.tf (already done)

# Option 2: Create bucket manually
gsutil mb -p gcp-poc-474818 -l us-central1 gs://gcp-poc-474818-terraform-state
gsutil versioning set on gs://gcp-poc-474818-terraform-state
```

### "Enter a value for Cloud Storage Bucket"

**Problem:** Terraform prompting for backend bucket during init

**Solution:** The backend is now commented out in `main.tf`. Just run:
```bash
terraform init
```

### Already have terraform.tfstate file

**Problem:** State file exists from previous run

**Solution:** You can continue using it or start fresh:
```bash
# Continue with existing state
terraform plan

# Or start fresh (CAREFUL - will destroy existing resources)
rm terraform.tfstate*
terraform init
terraform apply
```

## What Gets Created

Running `terraform apply` creates:

| Resource | Purpose | Cost |
|----------|---------|------|
| **GCE VM (e2-micro)** | Elasticsearch | FREE |
| **Persistent Disk (30GB)** | Elasticsearch data | FREE |
| **Artifact Registry** | Docker images | ~$0.10/GB |
| **Cloud Run (3 services)** | API, UI, ETL | FREE tier |
| **Service Account** | IAM permissions | FREE |
| **Firewall Rule** | Elasticsearch access | FREE |
| **Cloud Scheduler** | ETL trigger | FREE tier |

**Total first month: ~$0-5** (mostly Vertex AI usage)

## Useful Commands

```bash
# View current state
terraform show

# List all resources
terraform state list

# Get specific output
terraform output api_url

# Destroy everything (careful!)
terraform destroy

# Update specific resource
terraform apply -target=google_compute_instance.elasticsearch

# Refresh outputs without changes
terraform refresh
```

## Common Variables

Edit `terraform.tfvars`:

```hcl
# Scale up Elasticsearch
elasticsearch_machine_type = "e2-medium"  # $25/mo
elasticsearch_disk_size_gb = 50

# Use preemptible VM (dev only - saves 60%)
elasticsearch_use_preemptible = true

# Scale Cloud Run
api_max_instances = 20
api_min_instances = 1  # Keep warm for faster response
```

## Next Steps

After `terraform apply` succeeds:

1. **Get URLs:**
   ```bash
   terraform output
   ```

2. **Wait for services:**
   ```bash
   # Check Cloud Run
   gcloud run services list
   
   # Check Elasticsearch VM
   gcloud compute instances list
   ```

3. **Test Elasticsearch:**
   ```bash
   ES_IP=$(terraform output -raw elasticsearch_internal_ip)
   curl http://$ES_IP:9200/_cluster/health
   ```

4. **Access UI:**
   ```bash
   UI_URL=$(terraform output -raw ui_url)
   open $UI_URL
   ```

5. **Trigger ETL:**
   ```bash
   ETL_URL=$(terraform output -raw etl_url)
   curl -X POST $ETL_URL/trigger \
       -H "Authorization: Bearer $(gcloud auth print-identity-token)"
   ```

## Support

- Main guide: `../DEPLOY_GCP.md`
- Setup checklist: `../SETUP_INSTRUCTIONS.md`
- Elasticsearch details: `../docs/ELASTICSEARCH_GCE.md`

