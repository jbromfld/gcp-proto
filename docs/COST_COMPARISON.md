# 💰 Cost Comparison: Elasticsearch Options

## Summary

| Option | Setup Time | Monthly Cost | Free Tier | Best For |
|--------|------------|--------------|-----------|----------|
| **GCE e2-micro** | 10 min | **$0** | ✅ Yes | Dev, POC, low traffic |
| **GCE e2-medium** | 10 min | **$30** | ❌ No | Production <10K docs |
| Elastic Cloud | 5 min | **$95** | ⚠️ 14-day trial | Enterprise, no-ops |
| GKE Cluster | 2 hours | **$200+** | ❌ No | Large scale, HA |

## Detailed Breakdown

### Option 1: GCE e2-micro (FREE! 🎉)

**Cost:**
- VM: **$0** (Google Cloud always-free tier)
- Disk (30GB): **$0** (always-free tier)
- Network egress: **$0** (first 1GB/mo free)
- **Total: $0/month**

**Specs:**
- 0.25-2 vCPU (burstable)
- 1GB RAM
- 30GB persistent disk
- ~10K documents capacity
- ~100 queries/day

**Setup:**
```bash
cd terraform
terraform apply -var="elasticsearch_machine_type=e2-micro"
```

**Limitations:**
- Single instance (no HA)
- Limited RAM (1GB)
- Not for high-traffic production
- Perfect for development!

---

### Option 2: GCE e2-medium (~$30/mo)

**Cost:**
- VM (e2-medium): $24.50/mo
- Disk (50GB pd-balanced): $5/mo
- Network egress: ~$1/mo
- **Total: ~$30/month**

**Specs:**
- 2 vCPU
- 4GB RAM
- 50GB persistent disk
- ~100K documents capacity
- ~1-5K queries/day

**Setup:**
```bash
cd terraform
terraform apply -var="elasticsearch_machine_type=e2-medium" \
                -var="elasticsearch_disk_size_gb=50"
```

**Best for:**
- Small to medium production
- 100-10K documents
- 100-5K queries/day
- Single region

---

### Option 3: Elastic Cloud ($95+/mo)

**Cost:**
- Basic (4GB RAM, 120GB): $95/mo
- Standard (8GB RAM, 240GB): $190/mo
- Premium (16GB RAM, 480GB): $380/mo

**Specs (Basic tier):**
- 4GB RAM
- 120GB storage
- Multi-zone HA
- Auto-scaling
- Managed backups
- 99.9% SLA

**Setup:**
```bash
# Sign up at cloud.elastic.co
# Choose GCP, us-central1
# Copy credentials to Secret Manager
```

**Best for:**
- Enterprise with budget
- Need 99.9%+ SLA
- No DevOps team
- Advanced features (ML, APM)

---

### Option 4: GKE Cluster ($200+/mo)

**Cost:**
- GKE management: $75/mo
- 3x e2-standard-2 nodes: $150/mo
- Persistent disks (300GB): $30/mo
- **Total: ~$255/month**

**Specs:**
- 3-node cluster (HA)
- 6 vCPU total, 24GB RAM
- Auto-scaling
- Multi-zone
- Full control

**Best for:**
- Large scale (millions of docs)
- Need full control
- Multi-region deployment
- Part of existing Kubernetes stack

---

## Full System Cost

### Development/POC (Minimal)

| Component | Cost |
|-----------|------|
| Elasticsearch (GCE e2-micro) | **$0** (free tier) |
| Cloud Run API/UI | **$0** (free tier) |
| Vertex AI (100 queries/mo) | **$0.50** |
| **TOTAL** | **~$0.50/mo** 🎉 |

### Small Production (Recommended)

| Component | Cost |
|-----------|------|
| Elasticsearch (GCE e2-medium) | $30 |
| Cloud Run API/UI | $10-20 |
| Vertex AI (1K queries/day) | $5-10 |
| **TOTAL** | **$45-60/mo** |

### Medium Production

| Component | Cost |
|-----------|------|
| Elasticsearch (GCE e2-standard-2) | $50 |
| Cloud Run (scaled) | $30-50 |
| Vertex AI (10K queries/day) | $50-100 |
| **TOTAL** | **$130-200/mo** |

### Enterprise

| Component | Cost |
|-----------|------|
| **Elastic Cloud** (managed) | $95-380 |
| Cloud Run (scaled) | $50-150 |
| Vertex AI (100K queries/day) | $500-1000 |
| **TOTAL** | **$645-1530/mo** |

## Savings Comparison

| Workload | With Elastic Cloud | With GCE | **Savings** |
|----------|-------------------|----------|-------------|
| Dev/POC | $100/mo | **$0** | **100%** |
| Small Prod | $110/mo | **$50/mo** | **55%** |
| Medium Prod | $180/mo | **$150/mo** | **17%** |

## Recommendation by Use Case

| Use Case | Recommended Option | Cost |
|----------|-------------------|------|
| **POC/Demo** | GCE e2-micro | **FREE** |
| **Startup MVP** | GCE e2-medium | $45-60/mo |
| **Growing SaaS** | GCE e2-standard-2 | $130-200/mo |
| **Enterprise** | Elastic Cloud | $645+/mo |

## How to Switch

### Start Free, Scale Up

1. **Day 1**: Deploy with e2-micro (FREE)
2. **Month 1**: Upgrade to e2-medium ($30/mo)
3. **Month 6**: Switch to Elastic Cloud if needed ($95/mo)

```bash
# Upgrade machine type
gcloud compute instances stop elasticsearch --zone=us-central1-a
gcloud compute instances set-machine-type elasticsearch \
    --machine-type=e2-medium --zone=us-central1-a
gcloud compute instances start elasticsearch --zone=us-central1-a
```

### Migrate to Elastic Cloud Later

```bash
# Snapshot data from GCE
# Restore to Elastic Cloud deployment
# Update Cloud Run environment variables
# Done!
```

## Our Recommendation

**Start with GCE e2-micro (FREE)** → Validate your use case → Upgrade as needed

**Pays $0 during:**
- Development
- POC/Demo
- MVP testing
- Low traffic (<100 queries/day)

**Then scale to e2-medium ($30/mo)** when you have real users and budget.

