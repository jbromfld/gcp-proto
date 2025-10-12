## 500 In UI
```
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api AND severity>=ERROR" --limit=20 --format="value(timestamp,textPayload)" --project=gcp-poc-474818
```