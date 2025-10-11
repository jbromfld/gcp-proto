# Elasticsearch Setup for GCP

## Option 1: Elastic Cloud (Recommended)

### Why Elastic Cloud?
- ✅ Fully managed - no ops work
- ✅ Auto-scaling and high availability
- ✅ Hybrid search ready (BM25 + vector)
- ✅ Built-in security, backups, monitoring
- ✅ GCP region co-location
- ✅ Predictable pricing

### Setup Steps

1. **Create Account**
   - Visit: https://cloud.elastic.co/
   - Sign up for free trial (14 days)

2. **Create Deployment**
   ```
   - Click "Create deployment"
   - Name: rag-knowledge-base
   - Cloud provider: Google Cloud Platform
   - Region: us-central1 (match your Cloud Run region)
   - Version: 8.x (latest)
   - Hardware profile: "General Purpose"
   - Size: Start with 4GB RAM (~$95/mo)
   ```

3. **Get Credentials**
   - Copy the **Elasticsearch endpoint** (e.g., `https://xxx.es.us-central1.gcp.cloud.es.io:9243`)
   - Copy the **elastic user password** (save it securely!)
   - Optional: Download Cloud ID

4. **Store in Secret Manager**
   ```bash
   # Elasticsearch URL
   echo 'https://your-deployment.es.us-central1.gcp.cloud.es.io:9243' | \
       gcloud secrets versions add elasticsearch-url --data-file=-
   
   # Password
   echo 'your-elastic-password' | \
       gcloud secrets versions add elasticsearch-password --data-file=-
   ```

5. **Test Connection**
   ```bash
   # From Cloud Shell or local
   curl -u elastic:YOUR-PASSWORD \
       https://your-deployment.es.us-central1.gcp.cloud.es.io:9243/_cluster/health
   
   # Should return: {"status":"green",...}
   ```

### Pricing

| Tier | RAM | Storage | Price/month |
|------|-----|---------|-------------|
| Basic | 4GB | 120GB | ~$95 |
| Standard | 8GB | 240GB | ~$190 |
| Premium | 16GB | 480GB | ~$380 |

**Free trial**: 14 days on any tier

## Option 2: Self-Hosted on GKE

### Why Self-Hosted?
- ✅ More control over configuration
- ✅ Potentially lower cost at scale
- ❌ Requires Kubernetes expertise
- ❌ You manage updates, backups, scaling

### Setup with Elastic Cloud on Kubernetes (ECK)

1. **Create GKE Cluster**
   ```bash
   gcloud container clusters create rag-elasticsearch \
       --region=us-central1 \
       --num-nodes=3 \
       --machine-type=n2-standard-4 \
       --disk-size=100 \
       --enable-autoscaling \
       --min-nodes=3 \
       --max-nodes=6
   ```

2. **Install ECK Operator**
   ```bash
   kubectl create -f https://download.elastic.co/downloads/eck/2.10.0/crds.yaml
   kubectl apply -f https://download.elastic.co/downloads/eck/2.10.0/operator.yaml
   ```

3. **Deploy Elasticsearch**
   ```yaml
   # elasticsearch.yaml
   apiVersion: elasticsearch.k8s.elastic.co/v1
   kind: Elasticsearch
   metadata:
     name: rag-elasticsearch
   spec:
     version: 8.11.0
     nodeSets:
     - name: default
       count: 3
       config:
         node.store.allow_mmap: false
       podTemplate:
         spec:
           containers:
           - name: elasticsearch
             resources:
               limits:
                 memory: 4Gi
                 cpu: 2
       volumeClaimTemplates:
       - metadata:
           name: elasticsearch-data
         spec:
           accessModes:
           - ReadWriteOnce
           resources:
             requests:
               storage: 100Gi
   ```

   ```bash
   kubectl apply -f elasticsearch.yaml
   ```

4. **Get Credentials**
   ```bash
   # Get password
   PASSWORD=$(kubectl get secret rag-elasticsearch-es-elastic-user \
       -o go-template='{{.data.elastic | base64decode}}')
   
   # Get service URL
   kubectl get service rag-elasticsearch-es-http
   
   # Internal URL: rag-elasticsearch-es-http.default.svc.cluster.local:9200
   ```

5. **Configure VPC Connector**
   
   Cloud Run needs VPC connector to reach GKE:
   
   ```bash
   # Update .env.gcp
   USE_VPC_CONNECTOR=true
   
   # Or in terraform
   use_vpc_connector = true
   ```

### GKE Costs

| Component | Monthly Cost |
|-----------|--------------|
| GKE cluster | $75 (management) |
| 3x n2-standard-4 nodes | ~$360 |
| 300GB persistent disk | ~$30 |
| **Total** | **~$465/mo** |

**Note**: Self-hosted is more expensive at small scale. Consider Elastic Cloud unless you have specific requirements.

## Comparison

| Feature | Elastic Cloud | GKE Self-Hosted |
|---------|---------------|-----------------|
| Setup time | 5 minutes | 1-2 hours |
| Ops burden | None | High |
| Cost (small) | $95/mo | $465/mo |
| Cost (large) | $190-380/mo | $200-300/mo |
| Flexibility | Medium | High |
| Security | Built-in | You configure |
| Backups | Automatic | You configure |

**Recommendation**: Use **Elastic Cloud** for development and most production workloads.

## Next Steps

After setting up Elasticsearch:
1. Update secrets in GCP
2. Run `./deploy.sh all`
3. Access your UI and start querying!

