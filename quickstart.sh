
make setup        # Installs everything, starts ES + Ollama

# Start services
make start        # API + UI

# Or use Docker
docker-compose up -d

# Access UI
open http://localhost:8501