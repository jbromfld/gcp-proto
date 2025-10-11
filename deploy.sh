#!/bin/bash
# Deployment script for GCP Cloud Run
# Usage: ./deploy.sh [build|deploy|all]

set -e

# Load configuration
if [ -f ".env.gcp" ]; then
    source .env.gcp
else
    echo "Error: .env.gcp not found. Copy .env.gcp.template and fill in values."
    exit 1
fi

# Required variables
: "${GCP_PROJECT_ID:?Please set GCP_PROJECT_ID in .env.gcp}"
: "${GCP_REGION:?Please set GCP_REGION in .env.gcp}"

REPO_NAME="rag-system"
REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Authenticate with GCP
authenticate() {
    log_info "Authenticating with GCP..."
    gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
    log_success "Authenticated"
}

# Build and push images
build_images() {
    log_info "Building and pushing Docker images..."
    
    # Build API
    log_info "Building API image..."
    docker build -t ${REGISTRY}/rag-api:latest -f Dockerfile.api .
    docker push ${REGISTRY}/rag-api:latest
    log_success "API image pushed"
    
    # Build UI
    log_info "Building UI image..."
    docker build -t ${REGISTRY}/rag-ui:latest -f Dockerfile.ui .
    docker push ${REGISTRY}/rag-ui:latest
    log_success "UI image pushed"
    
    # Build ETL
    log_info "Building ETL image..."
    docker build -t ${REGISTRY}/rag-etl:latest -f Dockerfile.etl .
    docker push ${REGISTRY}/rag-etl:latest
    log_success "ETL image pushed"
}

# Deploy to Cloud Run
deploy_services() {
    log_info "Deploying to Cloud Run..."
    
    # Get Elasticsearch credentials from Secret Manager
    ES_URL=$(gcloud secrets versions access latest --secret="elasticsearch-url" 2>/dev/null || echo "")
    
    if [ -z "$ES_URL" ]; then
        log_warning "Elasticsearch URL not in Secret Manager. Using environment variable."
        ES_URL="${ELASTICSEARCH_URL}"
    fi
    
    # Deploy API
    log_info "Deploying API service..."
    gcloud run deploy rag-api \
        --image=${REGISTRY}/rag-api:latest \
        --platform=managed \
        --region=${GCP_REGION} \
        --allow-unauthenticated \
        --service-account=rag-service@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --set-env-vars="EMBEDDING_PROVIDER=vertex,LLM_PROVIDER=vertex,GOOGLE_PROJECT_ID=${GCP_PROJECT_ID}" \
        --set-secrets="ELASTICSEARCH_URL=elasticsearch-url:latest,ELASTICSEARCH_PASSWORD=elasticsearch-password:latest" \
        --cpu=2 \
        --memory=4Gi \
        --timeout=300 \
        --max-instances=10 \
        --min-instances=0
    
    API_URL=$(gcloud run services describe rag-api --region=${GCP_REGION} --format='value(status.url)')
    log_success "API deployed: ${API_URL}"
    
    # Deploy UI
    log_info "Deploying UI service..."
    gcloud run deploy rag-ui \
        --image=${REGISTRY}/rag-ui:latest \
        --platform=managed \
        --region=${GCP_REGION} \
        --allow-unauthenticated \
        --service-account=rag-service@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --set-env-vars="API_URL=${API_URL}" \
        --cpu=1 \
        --memory=2Gi \
        --timeout=300 \
        --max-instances=5 \
        --min-instances=0
    
    UI_URL=$(gcloud run services describe rag-ui --region=${GCP_REGION} --format='value(status.url)')
    log_success "UI deployed: ${UI_URL}"
    
    # Deploy ETL
    log_info "Deploying ETL service..."
    gcloud run deploy rag-etl \
        --image=${REGISTRY}/rag-etl:latest \
        --platform=managed \
        --region=${GCP_REGION} \
        --no-allow-unauthenticated \
        --service-account=rag-service@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --set-env-vars="EMBEDDING_PROVIDER=vertex,GOOGLE_PROJECT_ID=${GCP_PROJECT_ID},SCRAPE_URLS=${SCRAPE_URLS:-https://docs.python.org/3/tutorial/}" \
        --set-secrets="ELASTICSEARCH_URL=elasticsearch-url:latest,ELASTICSEARCH_PASSWORD=elasticsearch-password:latest" \
        --cpu=2 \
        --memory=4Gi \
        --timeout=3600 \
        --max-instances=1 \
        --min-instances=0
    
    log_success "ETL deployed"
    
    echo ""
    log_success "Deployment complete!"
    echo ""
    echo "Access your services:"
    echo "  UI:  ${UI_URL}"
    echo "  API: ${API_URL}"
}

# Main script
case "${1:-all}" in
    build)
        authenticate
        build_images
        ;;
    deploy)
        deploy_services
        ;;
    all)
        authenticate
        build_images
        deploy_services
        ;;
    *)
        echo "Usage: $0 [build|deploy|all]"
        exit 1
        ;;
esac

