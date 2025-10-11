#!/bin/bash
# Initial setup script for GCP deployment
# Run this once before deploying

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ ${NC}$1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# Check if .env.gcp exists
if [ ! -f ".env.gcp" ]; then
    log_error ".env.gcp not found. Creating from template..."
    cp .env.gcp.template .env.gcp
    log_warning "Please edit .env.gcp with your GCP project details"
    exit 1
fi

# Load configuration
source .env.gcp

: "${GCP_PROJECT_ID:?Please set GCP_PROJECT_ID in .env.gcp}"
: "${GCP_REGION:?Please set GCP_REGION in .env.gcp}"

log_info "Setting up GCP project: ${GCP_PROJECT_ID}"

# Set current project
gcloud config set project ${GCP_PROJECT_ID}

# Enable required APIs
log_info "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    compute.googleapis.com

log_success "APIs enabled"

# Create Artifact Registry repository
log_info "Creating Artifact Registry repository..."
gcloud artifacts repositories create rag-system \
    --repository-format=docker \
    --location=${GCP_REGION} \
    --description="Docker repository for RAG system" \
    || log_warning "Repository may already exist"

log_success "Artifact Registry ready"

# Create service account
log_info "Creating service account..."
gcloud iam service-accounts create rag-service \
    --display-name="RAG System Service Account" \
    --description="Service account for RAG API, UI, and ETL" \
    || log_warning "Service account may already exist"

SERVICE_ACCOUNT="rag-service@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Grant permissions
log_info "Granting IAM permissions..."
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user" \
    --condition=None

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

log_success "Permissions granted"

# Create secrets
log_info "Creating Secret Manager secrets..."

# Elasticsearch URL
if [ ! -z "${ELASTICSEARCH_URL}" ]; then
    echo -n "${ELASTICSEARCH_URL}" | gcloud secrets create elasticsearch-url \
        --data-file=- \
        --replication-policy="automatic" \
        || gcloud secrets versions add elasticsearch-url --data-file=- <<< "${ELASTICSEARCH_URL}"
    log_success "Elasticsearch URL secret created"
else
    gcloud secrets create elasticsearch-url \
        --replication-policy="automatic" \
        || log_warning "elasticsearch-url secret exists"
    log_warning "Please set Elasticsearch URL manually:"
    echo "  echo 'https://your-elastic-url' | gcloud secrets versions add elasticsearch-url --data-file=-"
fi

# Elasticsearch password
if [ ! -z "${ELASTICSEARCH_PASSWORD}" ]; then
    echo -n "${ELASTICSEARCH_PASSWORD}" | gcloud secrets create elasticsearch-password \
        --data-file=- \
        --replication-policy="automatic" \
        || gcloud secrets versions add elasticsearch-password --data-file=- <<< "${ELASTICSEARCH_PASSWORD}"
    log_success "Elasticsearch password secret created"
else
    gcloud secrets create elasticsearch-password \
        --replication-policy="automatic" \
        || log_warning "elasticsearch-password secret exists"
    log_warning "Please set Elasticsearch password manually:"
    echo "  echo 'your-password' | gcloud secrets versions add elasticsearch-password --data-file=-"
fi

# Create terraform state bucket (if using terraform)
if [ "${USE_TERRAFORM:-false}" = "true" ]; then
    log_info "Creating Terraform state bucket..."
    gsutil mb -p ${GCP_PROJECT_ID} -l ${GCP_REGION} gs://${GCP_PROJECT_ID}-terraform-state || log_warning "Bucket may already exist"
    gsutil versioning set on gs://${GCP_PROJECT_ID}-terraform-state
    log_success "Terraform state bucket ready"
fi

echo ""
log_success "GCP setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set up Elasticsearch (Elastic Cloud or GKE)"
echo "  2. Update secrets if needed:"
echo "     gcloud secrets versions add elasticsearch-url --data-file=-"
echo "  3. Deploy services:"
echo "     ./deploy.sh all"
echo ""

