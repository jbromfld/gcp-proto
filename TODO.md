```
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query": "What is Python?", "max_results": 3}'
```
```
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d --build
```