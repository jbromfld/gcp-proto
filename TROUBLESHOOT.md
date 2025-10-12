## 500 In UI
```
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api AND severity>=ERROR" --limit=20 --format="value(timestamp,textPayload)" --project=gcp-poc-474818
```

```
docker-compose down -v
docker pull python:3.11-slim 
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.12
docker-compose build --no-cache
docker-compose up -d --build
```

```
colima stop && colima start --memory 8 --cpu 4
```

## Quick test for local
```
docker-compose down
docker-compose up -d
# Wait 30 seconds for startup
sleep 30
# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is python?"}' | jq .
```