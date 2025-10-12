#!/bin/bash
set -e

# Pause GCP deployment to minimize costs
# This stops the Elasticsearch VM but keeps Cloud Run services (they auto-scale to $0)

echo "🛑 Pausing GCP RAG System..."
echo ""

PROJECT_ID="${GCP_PROJECT_ID:-gcp-poc-474818}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${REGION}-a"

echo "📋 Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Zone: $ZONE"
echo ""

# Stop Elasticsearch VM
echo "🔴 Stopping Elasticsearch VM..."
if gcloud compute instances describe elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
    STATUS=$(gcloud compute instances describe elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID --format='get(status)')
    
    if [ "$STATUS" = "RUNNING" ]; then
        gcloud compute instances stop elasticsearch-vm --zone=$ZONE --project=$PROJECT_ID --quiet
        echo "✅ Elasticsearch VM stopped"
    elif [ "$STATUS" = "TERMINATED" ]; then
        echo "ℹ️  Elasticsearch VM is already stopped"
    else
        echo "⚠️  Elasticsearch VM is in unexpected state: $STATUS"
    fi
else
    echo "⚠️  Elasticsearch VM not found (may have been destroyed)"
fi

echo ""
echo "✅ GCP deployment paused!"
echo ""
echo "💰 Cost Estimate While Paused:"
echo "  • Elasticsearch disk (50GB): ~$8/month"
echo "  • VPC Connector: ~$8/month"
echo "  • Cloud Run (idle): ~$0/month"
echo "  • Total: ~$16/month"
echo ""
echo "📌 To resume: ./resume-gcp.sh"

