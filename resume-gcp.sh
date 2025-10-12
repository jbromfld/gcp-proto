#!/bin/bash
set -e

# Resume GCP deployment after pause
# This starts the Elasticsearch VM and verifies all services are healthy

echo "▶️  Resuming GCP RAG System..."
echo ""

PROJECT_ID="${GCP_PROJECT_ID:-gcp-poc-474818}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${REGION}-a"

echo "📋 Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Zone: $ZONE"
echo ""

# Start Elasticsearch VM
echo "🟢 Starting Elasticsearch VM..."
if gcloud compute instances describe elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
    STATUS=$(gcloud compute instances describe elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID --format='get(status)')
    
    if [ "$STATUS" = "TERMINATED" ]; then
        gcloud compute instances start elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID --quiet
        echo "✅ Elasticsearch VM started"
        echo "⏳ Waiting for Elasticsearch to initialize (60 seconds)..."
        sleep 60
    elif [ "$STATUS" = "RUNNING" ]; then
        echo "ℹ️  Elasticsearch VM is already running"
    else
        echo "⚠️  Elasticsearch VM is in unexpected state: $STATUS"
    fi
else
    echo "❌ Elasticsearch VM not found!"
    echo "   Run: cd terraform && terraform apply"
    exit 1
fi

# Get Elasticsearch internal IP
ES_IP=$(gcloud compute instances describe elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID --format='get(networkInterfaces[0].networkIP)')
echo ""
echo "📊 Elasticsearch Internal IP: $ES_IP"

# Health check
echo ""
echo "🏥 Running health checks..."

# Check Elasticsearch
echo -n "  • Elasticsearch: "
if timeout 10 bash -c "curl -s http://$ES_IP:9200/_cluster/health" &>/dev/null; then
    HEALTH=$(curl -s http://$ES_IP:9200/_cluster/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    echo "✅ $HEALTH"
else
    echo "⚠️  Not responding yet (may need more time)"
fi

# Check Cloud Run services
echo -n "  • API Service: "
API_URL=$(gcloud run services describe rag-api --region=$REGION --project=$PROJECT_ID --format='value(status.url)' 2>/dev/null || echo "")
if [ -n "$API_URL" ]; then
    if curl -s -o /dev/null -w "%{http_code}" "$API_URL/" | grep -q "200"; then
        echo "✅ Healthy"
    else
        echo "⚠️  May need a few minutes to reconnect to Elasticsearch"
    fi
else
    echo "⚠️  Not found"
fi

echo -n "  • UI Service: "
UI_URL=$(gcloud run services describe rag-ui --region=$REGION --project=$PROJECT_ID --format='value(status.url)' 2>/dev/null || echo "")
if [ -n "$UI_URL" ]; then
    if curl -s -o /dev/null -w "%{http_code}" "$UI_URL/" | grep -q "200"; then
        echo "✅ Healthy"
    else
        echo "⚠️  May need a few minutes"
    fi
else
    echo "⚠️  Not found"
fi

echo ""
echo "✅ GCP deployment resumed!"
echo ""
echo "🌐 Service URLs:"
echo "  • UI:  ${UI_URL}"
echo "  • API: ${API_URL}"
echo ""
echo "📌 To pause again: ./pause-gcp.sh"
echo "📌 To fully destroy: cd terraform && terraform destroy"

