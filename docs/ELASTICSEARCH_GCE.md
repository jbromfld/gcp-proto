# Self-Hosted Elasticsearch on GCE (Cost-Effective)

## Why GCE Instead of Elastic Cloud?

| Feature | Elastic Cloud | GCE Self-Hosted |
|---------|---------------|-----------------|
| **Cost (small)** | $95/mo minimum | $12-30/mo |
| **Free tier eligible** | No | Yes! (1 e2-micro free) |
| **Setup time** | 5 minutes | 10 minutes |
| **Ops burden** | None | Low (automated) |

## Free Tier Option! 🎉

Google Cloud's **always-free tier** includes:
- 1x e2-micro VM (0.25 vCPU, 1GB RAM)
- 30GB standard persistent disk
- **Perfect for development and small workloads!**

## Quick Setup

### Option 1: Using Terraform (Automated)

```bash
cd terraform

# Edit terraform.tfvars
cat > terraform.tfvars <<EOF
project_id = "your-project-id"
region = "us-central1"
use_gce_elasticsearch = true
elasticsearch_machine_type = "e2-micro"  # FREE TIER!
elasticsearch_disk_size_gb = 30  # FREE TIER!
EOF

# Deploy
terraform init
terraform apply
```

### Option 2: Manual Setup (5 commands)

```bash
# 1. Create persistent disk for data
gcloud compute disks create elasticsearch-data \
    --size=30GB \
    --type=pd-standard \
    --zone=us-central1-a

# 2. Create VM with startup script
gcloud compute instances create elasticsearch \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --disk=name=elasticsearch-data,device-name=elasticsearch-data,mode=rw \
    --tags=elasticsearch \
    --metadata=startup-script='#!/bin/bash
    # Format and mount data disk
    if ! mount | grep -q /var/lib/elasticsearch; then
      mkfs.ext4 -F /dev/disk/by-id/google-elasticsearch-data || true
      mkdir -p /var/lib/elasticsearch
      mount /dev/disk/by-id/google-elasticsearch-data /var/lib/elasticsearch
      echo "/dev/disk/by-id/google-elasticsearch-data /var/lib/elasticsearch ext4 defaults 0 0" >> /etc/fstab
    fi
    
    # Install Docker
    curl -fsSL https://get.docker.com | sh
    
    # Run Elasticsearch
    docker run -d \
      --name elasticsearch \
      --restart unless-stopped \
      -p 9200:9200 \
      -e "discovery.type=single-node" \
      -e "xpack.security.enabled=false" \
      -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
      -v /var/lib/elasticsearch:/usr/share/elasticsearch/data \
      docker.elastic.co/elasticsearch/elasticsearch:8.11.0'

# 3. Create firewall rule
gcloud compute firewall-rules create allow-elasticsearch \
    --allow=tcp:9200 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=elasticsearch

# 4. Get internal IP
ELASTICSEARCH_IP=$(gcloud compute instances describe elasticsearch \
    --zone=us-central1-a \
    --format='value(networkInterfaces[0].networkIP)')

echo "Elasticsearch URL: http://$ELASTICSEARCH_IP:9200"

# 5. Store in Secret Manager
echo "http://$ELASTICSEARCH_IP:9200" | \
    gcloud secrets versions add elasticsearch-url --data-file=-
```

### Option 3: Using Deploy Script

```bash
# Edit .env.gcp
USE_GCE_ELASTICSEARCH=true
ELASTICSEARCH_MACHINE_TYPE=e2-micro  # FREE!

# Run setup
./setup-gcp.sh --with-elasticsearch
```

## Machine Type Options

| Type | vCPU | RAM | Cost/month | Use Case |
|------|------|-----|------------|----------|
| **e2-micro** | 0.25-2 | 1GB | **FREE** | Dev, low traffic |
| e2-small | 0.5-2 | 2GB | $12 | Testing, demos |
| e2-medium | 1-2 | 4GB | $25 | Small production |
| e2-standard-2 | 2 | 8GB | $50 | Medium production |
| n2-standard-4 | 4 | 16GB | $180 | High volume |

## Cost Breakdown (e2-micro)

| Component | Monthly Cost |
|-----------|--------------|
| VM (e2-micro) | **FREE** (always-free tier) |
| Disk (30GB) | **FREE** (always-free tier) |
| Egress | **FREE** (first 1GB/mo) |
| **TOTAL** | **$0** 🎉 |

## Cost Breakdown (e2-medium - Recommended)

| Component | Monthly Cost |
|-----------|--------------|
| VM (e2-medium) | $25 |
| Disk (50GB pd-balanced) | $5 |
| Egress | ~$1 |
| **TOTAL** | **~$30-35/mo** |

Compare to Elastic Cloud: $95/mo minimum!

## Performance Notes

### e2-micro (FREE)
- ✅ Good for: Dev, testing, <100 docs, <10 queries/day
- ❌ Limited: 1GB RAM restricts index size
- 💡 Tip: Use for proof-of-concept, then upgrade

### e2-medium ($25/mo)
- ✅ Good for: Production, 10K docs, 100-1K queries/day
- ✅ 4GB RAM supports larger indices
- ✅ Good performance for most use cases

## Accessing Elasticsearch

### From Cloud Run (Internal)
Cloud Run services automatically use internal IP:
```
http://10.128.0.X:9200
```

### From Your Computer (External)
```bash
# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe elasticsearch \
    --zone=us-central1-a \
    --format='value(networkInterfaces[0].accessConfig[0].natIP)')

# Test
curl http://$EXTERNAL_IP:9200

# Access Kibana (if installed)
open http://$EXTERNAL_IP:5601
```

## Security (Production)

### 1. Enable Authentication
Update startup script to enable xpack.security:
```bash
-e "xpack.security.enabled=true" \
-e "ELASTIC_PASSWORD=your-secure-password"
```

### 2. Restrict Firewall
```bash
# Only allow Cloud Run VPC
gcloud compute firewall-rules update allow-elasticsearch \
    --source-ranges=10.0.0.0/8 \
    --no-enable-logging

# Or use VPC Service Controls
```

### 3. Use HTTPS
Add nginx reverse proxy with Let's Encrypt:
```bash
# Install nginx and certbot on VM
# Configure SSL termination
```

## Backup & Recovery

### Automated Snapshots
```bash
# Create snapshot schedule
gcloud compute resource-policies create snapshot-schedule elasticsearch-backup \
    --region=us-central1 \
    --max-retention-days=7 \
    --daily-schedule \
    --start-time=02:00

# Attach to disk
gcloud compute disks add-resource-policies elasticsearch-data \
    --zone=us-central1-a \
    --resource-policies=elasticsearch-backup
```

### Manual Backup
```bash
# Create snapshot
gcloud compute disks snapshot elasticsearch-data \
    --zone=us-central1-a \
    --snapshot-names=elasticsearch-backup-$(date +%Y%m%d)

# Restore from snapshot
gcloud compute disks create elasticsearch-data-restored \
    --source-snapshot=elasticsearch-backup-YYYYMMDD \
    --zone=us-central1-a
```

## Monitoring

### Health Check
```bash
# Check if running
gcloud compute instances describe elasticsearch \
    --zone=us-central1-a \
    --format='value(status)'

# Check Elasticsearch health
INTERNAL_IP=$(gcloud compute instances describe elasticsearch \
    --zone=us-central1-a \
    --format='value(networkInterfaces[0].networkIP)')

curl http://$INTERNAL_IP:9200/_cluster/health
```

### Logs
```bash
# SSH to instance
gcloud compute ssh elasticsearch --zone=us-central1-a

# View Elasticsearch logs
docker logs elasticsearch -f

# View system logs
sudo journalctl -u docker -f
```

## Scaling Up

### Upgrade Machine Type (Zero Downtime)
```bash
# Stop instance
gcloud compute instances stop elasticsearch --zone=us-central1-a

# Change machine type
gcloud compute instances set-machine-type elasticsearch \
    --zone=us-central1-a \
    --machine-type=e2-standard-2

# Start instance
gcloud compute instances start elasticsearch --zone=us-central1-a
```

### Add More Disk Space
```bash
# Resize disk (can only increase)
gcloud compute disks resize elasticsearch-data \
    --size=100GB \
    --zone=us-central1-a

# SSH and resize filesystem
gcloud compute ssh elasticsearch --zone=us-central1-a
sudo resize2fs /dev/disk/by-id/google-elasticsearch-data
```

## When to Use Elastic Cloud Instead

Consider Elastic Cloud if:
- ❌ You need 99.9%+ SLA guarantees
- ❌ Multi-region/geo replication required
- ❌ Team doesn't want to manage infrastructure
- ❌ Need advanced features (ML, APM, SIEM)
- ❌ Handling millions of documents

Otherwise, **GCE is perfect** for most use cases and **70% cheaper**!

## Cost Comparison

| Workload | Elastic Cloud | GCE (e2-medium) | Savings |
|----------|---------------|-----------------|---------|
| Dev/Test | $95/mo | **FREE** (e2-micro) | 100% |
| Small Prod | $95/mo | $30/mo | 68% |
| Medium Prod | $190/mo | $50-80/mo | 58-74% |

## Next Steps

After deployment:
1. Verify Elasticsearch is running
2. Update Cloud Run services with internal IP
3. Test queries
4. Set up automated backups
5. Monitor performance

Full deployment guide: `DEPLOY_GCP.md`

