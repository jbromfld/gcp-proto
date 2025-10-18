# Troubleshooting Guide

Quick reference for debugging common issues.

---

## 🐛 Local Development Issues

### Container Logs

```bash
# View all container logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Specific service logs
docker logs rag-api
docker logs rag-ui
docker logs rag-elasticsearch

# Last 50 lines with timestamps
docker logs rag-api --tail 50 --timestamps

# Filter for errors
docker logs rag-api 2>&1 | grep -i error
```

### Service Health Checks

```bash
# Check which containers are running
docker-compose ps

# Check Elasticsearch health
curl http://localhost:9200/_cluster/health?pretty

# Check API health
curl http://localhost:8000/

# Check if API is responding
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' | jq .

# Check UI (should return HTML)
curl -s http://localhost:8501/ | head -20
```

### Common Local Errors

**"Connection refused" to Elasticsearch:**
```bash
# Check if ES container is running
docker ps | grep elasticsearch

# Check ES logs
docker logs rag-elasticsearch

# Restart ES
docker-compose restart elasticsearch

# Full rebuild
docker-compose down -v
docker-compose up -d
```

**"Ollama connection error":**
```bash
# Check if native Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve &

# Or use host.docker.internal
export OLLAMA_URL=http://host.docker.internal:11434
docker-compose restart api
```

**"500 Server Error" in UI:**
```bash
# Check API logs for the actual error
docker logs rag-api --tail 100

# Common causes:
# - LLM model not found → Check rag_llm_abstraction.py config
# - Embedding dimension mismatch → Delete and recreate index
# - Elasticsearch not ready → Wait 30 seconds and retry
```

### Reset Everything Local

```bash
# Nuclear option - start fresh
docker-compose down -v
docker system prune -a --volumes  # Careful: deletes ALL Docker data!
rm -rf __pycache__
./setup_local.sh
./start_local.sh
```

---

## ☁️ GCP Deployment Issues

### Check Cloud Run Logs

```bash
# API errors (last 20)
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api AND severity>=ERROR" \
  --limit=20 --format="value(timestamp,textPayload)" \
  --project=gcp-poc-474818

# UI errors
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-ui AND severity>=ERROR" \
  --limit=20 \
  --project=gcp-poc-474818

# ETL errors
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-etl AND severity>=ERROR" \
  --limit=20 \
  --project=gcp-poc-474818

# Real-time logs (follow)
gcloud alpha logging tail \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api" \
  --project=gcp-poc-474818

# All startup errors
gcloud logging read \
  "resource.type=cloud_run_revision AND textPayload=~'startup' AND severity>=ERROR" \
  --limit=50 \
  --project=gcp-poc-474818
```

### Check Service Status

```bash
# List all Cloud Run services
gcloud run services list --project=gcp-poc-474818

# Describe specific service
gcloud run services describe rag-api \
  --region=us-central1 \
  --project=gcp-poc-474818 \
  --format=yaml

# Check latest revision status
gcloud run revisions list \
  --service=rag-api \
  --region=us-central1 \
  --limit=1 \
  --format="table(metadata.name,status.conditions[0].status,status.conditions[0].message)"

# Test service endpoint
API_URL=$(gcloud run services describe rag-api --region=us-central1 --format='value(status.url)' --project=gcp-poc-474818)
curl $API_URL/
```

### Check Elasticsearch VM

```bash
# Check VM status
gcloud compute instances describe elasticsearch-vm \
  --zone=us-central1-a \
  --project=gcp-poc-474818 \
  --format="value(status)"

# SSH into VM
gcloud compute ssh elasticsearch-vm \
  --zone=us-central1-a \
  --project=gcp-poc-474818

# Once inside VM:
sudo docker ps
sudo docker logs elasticsearch
curl localhost:9200/_cluster/health?pretty
exit

# Check from outside (if VPC connector working)
gcloud compute instances describe elasticsearch-vm \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].networkIP)' \
  --project=gcp-poc-474818
# Use this IP to test connectivity
```

### Verify APIs Enabled

```bash
# Check all enabled APIs
gcloud services list --enabled --project=gcp-poc-474818

# Check specific required APIs
gcloud services list --enabled \
  --filter="name:(run.googleapis.com OR aiplatform.googleapis.com OR artifactregistry.googleapis.com)" \
  --project=gcp-poc-474818

# Enable missing APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  vpcaccess.googleapis.com \
  --project=gcp-poc-474818
```

### Check IAM Permissions

```bash
# Check service account exists
gcloud iam service-accounts list --project=gcp-poc-474818

# Check service account permissions
gcloud projects get-iam-policy gcp-poc-474818 \
  --flatten="bindings[].members" \
  --filter="bindings.members:rag-service"

# Should see:
# - roles/aiplatform.user
# - roles/secretmanager.secretAccessor (if using Elastic Cloud)

# Grant missing permissions
gcloud projects add-iam-policy-binding gcp-poc-474818 \
  --member="serviceAccount:rag-service@gcp-poc-474818.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Common GCP Errors

**"Model not found" (Vertex AI):**
```bash
# Error: Publisher Model `gemini-X` was not found

# Check current model name in code
grep "model_name=" rag_llm_abstraction.py

# Should be: gemini-1.5-pro (confirmed working)
# NOT: gemini-pro, gemini-1.5-pro-002, etc.

# Verify Vertex AI API is enabled
gcloud services list --enabled | grep aiplatform

# Test Vertex AI access
gcloud ai models list --region=us-central1 --project=gcp-poc-474818
```

**"Container failed to start" (Cloud Run):**
```bash
# Get detailed error
gcloud logging read \
  "resource.type=cloud_run_revision AND textPayload=~'failed to start'" \
  --limit=5 \
  --project=gcp-poc-474818

# Common causes:
# 1. Wrong architecture (ARM vs AMD64)
#    → Build with Cloud Build, not local docker
# 2. Missing environment variable
#    → Check terraform/cloud_run.tf env vars
# 3. Port misconfigured
#    → Verify PORT=8000 (API), PORT=8501 (UI), PORT=8080 (ETL)
# 4. Startup timeout
#    → Increase timeout in terraform/cloud_run.tf
```

**"Cannot connect to Elasticsearch" (from Cloud Run):**
```bash
# Check VPC connector exists
gcloud compute networks vpc-access connectors list \
  --region=us-central1 \
  --project=gcp-poc-474818

# Check VPC connector status
gcloud compute networks vpc-access connectors describe rag-vpc-connector \
  --region=us-central1 \
  --project=gcp-poc-474818

# Verify in terraform.tfvars:
# use_vpc_connector = true
# use_gce_elasticsearch = true

# Re-create if needed
cd terraform && terraform apply
```

**"Image not found" (Artifact Registry):**
```bash
# List images in registry
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/gcp-poc-474818/rag-system \
  --project=gcp-poc-474818

# If empty, build images:
gcloud builds submit --config cloudbuild.yaml

# Check build status
gcloud builds list --limit=5 --project=gcp-poc-474818
```

### Check Build Status

```bash
# List recent builds
gcloud builds list --limit=5 --project=gcp-poc-474818

# Get build details
BUILD_ID=$(gcloud builds list --limit=1 --format="value(id)" --project=gcp-poc-474818)
gcloud builds describe $BUILD_ID --project=gcp-poc-474818

# Watch build progress
gcloud builds log $(gcloud builds list --limit=1 --format="value(id)" --project=gcp-poc-474818) \
  --stream \
  --project=gcp-poc-474818

# Check build failures
gcloud builds list --filter="status=FAILURE" --limit=10 --project=gcp-poc-474818
```

---

## 🔍 Debugging Specific Components

### Embeddings Issues

```bash
# Local: Check if model downloaded
ls ~/.cache/huggingface/hub/ | grep mpnet

# Local: Test embedding directly
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
embedding = model.encode("test")
print(f"✓ Model works, dims: {len(embedding)}")
EOF

# GCP: Check Vertex AI access
gcloud ai models list --region=us-central1 --project=gcp-poc-474818

# Check API logs for embedding errors
docker logs rag-api 2>&1 | grep -i "embedding\|dimension"
```

### LLM Issues

```bash
# Local: Check Ollama
curl http://localhost:11434/api/tags
ollama list

# Local: Test Ollama model
ollama run llama3.2 "hello"

# GCP: Test Vertex AI LLM
gcloud ai endpoints list --region=us-central1 --project=gcp-poc-474818

# Check logs for LLM errors
gcloud logging read \
  "resource.labels.service_name=rag-api AND textPayload=~'gemini\|LLM\|generate'" \
  --limit=20 \
  --project=gcp-poc-474818
```

### Elasticsearch Issues

#### **Lock File Issues (GCP)**
**Error**: `failed to obtain node locks, tried [/usr/share/elasticsearch/data]`

**Root Cause**: Elasticsearch container can't create lock files due to permission issues

**Fix**:
```bash
# SSH into the Elasticsearch VM
gcloud compute ssh elasticsearch --zone=us-central1-a --project=YOUR_PROJECT

# Fix permissions and clean up lock files
sudo mkdir -p /var/lib/elasticsearch
sudo chown -R 1000:1000 /var/lib/elasticsearch
sudo chmod -R 755 /var/lib/elasticsearch
sudo find /var/lib/elasticsearch -name "*.lock" -type f -delete
sudo rm -rf /var/lib/elasticsearch/nodes
sudo docker restart elasticsearch
```

**Prevention**: The Terraform startup script now includes these fixes automatically.

#### **General Elasticsearch Checks**
```bash
# Check index exists
curl http://localhost:9200/_cat/indices?v

# Check index mapping
curl http://localhost:9200/knowledge_base/_mapping | jq .

# Check document count
curl http://localhost:9200/knowledge_base/_count

# Check cluster health
curl http://localhost:9200/_cluster/health?pretty

# Check node stats
curl http://localhost:9200/_nodes/stats?pretty

# Delete and recreate index (DESTRUCTIVE!)
curl -X DELETE http://localhost:9200/knowledge_base
# Restart API to recreate
docker-compose restart api
```

### Network Connectivity

```bash
# Local: Check Docker network
docker network ls
docker network inspect gcp-proto_rag-network

# GCP: Test API from UI
# SSH into running UI container (if possible) or use Cloud Shell:
curl http://rag-api-internal-url/

# GCP: Test VPC connectivity
gcloud compute networks vpc-access connectors describe rag-vpc-connector \
  --region=us-central1 \
  --project=gcp-poc-474818 \
  --format="value(state,network,ipCidrRange)"
```

---

## 🚑 Emergency Fixes

### Local: Complete Reset

```bash
# Stop everything
docker-compose down -v

# Clear Docker cache (careful!)
docker system prune -a --volumes

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Delete Elasticsearch data
docker volume rm gcp-proto_es-data 2>/dev/null

# Start fresh
./setup_local.sh
./start_local.sh
```

### GCP: Rollback to Previous Version

```bash
# List revisions
gcloud run revisions list \
  --service=rag-api \
  --region=us-central1 \
  --project=gcp-poc-474818

# Route to previous working revision
gcloud run services update-traffic rag-api \
  --region=us-central1 \
  --to-revisions=rag-api-00005-xyz=100 \
  --project=gcp-poc-474818
```

### GCP: Force Rebuild

```bash
# Delete failed revisions
gcloud run revisions delete rag-api-00006-abc \
  --region=us-central1 \
  --project=gcp-poc-474818 \
  --quiet

# Rebuild images
gcloud builds submit --config cloudbuild.yaml --project=gcp-poc-474818

# Force redeploy
cd terraform
terraform taint google_cloud_run_service.api
terraform apply
```

---

## 📊 Verification Checklist

### Before Deploying to GCP

- [ ] Tested locally with Vertex AI (`EMBEDDING_PROVIDER=vertex`)
- [ ] Confirmed model names work (`gemini-1.5-pro`, `text-embedding-004`)
- [ ] Verified ADC credentials: `gcloud auth application-default login`
- [ ] Checked dimensions match (768 for both local mpnet and Vertex AI)
- [ ] Reviewed terraform.tfvars settings
- [ ] APIs enabled in GCP project

### After GCP Deployment

- [ ] All services show "Ready: True": `gcloud run services list`
- [ ] Elasticsearch VM is running: `gcloud compute instances list`
- [ ] API responds: `curl $API_URL/`
- [ ] UI loads: `open $UI_URL`
- [ ] Can add source via Admin UI
- [ ] Can run query and get response
- [ ] Metrics show data

---

## 🔧 Quick Fixes

### 500 Error in UI

```bash
# 1. Check API logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api AND severity>=ERROR" \
  --limit=20 --format="value(timestamp,textPayload)" \
  --project=gcp-poc-474818

# 2. Common fixes:
# - Wrong model name → Update rag_llm_abstraction.py
# - Dimension mismatch → Rebuild with unified 768 dims
# - ES not ready → Restart ES VM
```

### Local Docker Issues

```bash
# Rebuild without cache
docker-compose down -v
docker pull python:3.11-slim 
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.12
docker-compose build --no-cache
docker-compose up -d --build

# If Colima (macOS Docker):
colima stop && colima start --memory 8 --cpu 4
```

### Quick Local Test

```bash
docker-compose down
docker-compose up -d
sleep 30  # Wait for startup
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is python?"}' | jq .
```

### Quick GCP Deploy

```bash
# From project root
gcloud builds submit --config cloudbuild.yaml --project=gcp-poc-474818
cd terraform && terraform apply -auto-approve
```

---

## 🆘 Getting More Help

### Environment Info

```bash
# Capture system info for debugging
echo "=== Local Environment ==="
docker --version
docker-compose --version
python3 --version
ollama --version

echo ""
echo "=== Docker Containers ==="
docker-compose ps

echo ""
echo "=== Environment Variables ==="
docker exec rag-api env | grep -E "PROVIDER|PROJECT|OLLAMA|ELASTIC"

echo ""
echo "=== Elasticsearch ==="
curl -s http://localhost:9200/_cluster/health | jq .

echo ""
echo "=== Recent Errors ==="
docker logs rag-api 2>&1 | grep -i error | tail -10
```

### GCP Info

```bash
# Capture GCP state for debugging
echo "=== GCP Project ==="
gcloud config get-value project

echo ""
echo "=== Services ==="
gcloud run services list --project=gcp-poc-474818

echo ""
echo "=== Builds ==="
gcloud builds list --limit=3 --project=gcp-poc-474818

echo ""
echo "=== VMs ==="
gcloud compute instances list --project=gcp-poc-474818

echo ""
echo "=== Recent Errors ==="
gcloud logging read "severity>=ERROR" --limit=10 --project=gcp-poc-474818
```

---

## 📞 Support Resources

- **Local Development Guide**: `docs/LOCAL_DEVELOPMENT.md`
- **GCP Deployment Guide**: `docs/GCP_DEPLOYMENT.md`
- **Optimization Guide**: `docs/SIMILARITY_SEARCH_GUIDE.md`
- **GCP Documentation**: https://cloud.google.com/run/docs
- **Vertex AI Models**: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference