# Makefile - Convenient commands

.PHONY: setup start stop test clean deploy

# Setup environment
setup:
	@bash setup.sh

# Start all services
start:
	@bash start_local.sh

# Stop all services
stop:
	@docker stop rag-elasticsearch rag-ollama || true
	@docker rm rag-elasticsearch rag-ollama || true
	@pkill -f "python rag_api.py" || true
	@pkill -f "streamlit run" || true

# Run tests
test:
	@python test_setup.py
	@pytest tests/ -v

# Clean up
clean:
	@docker stop rag-elasticsearch rag-ollama || true
	@docker rm rag-elasticsearch rag-ollama || true
	@docker volume rm ollama-data || true
	@rm -rf venv __pycache__ .pytest_cache logs/*

# Full cleanup including ES data
clean-all: clean
	@docker volume rm rag-knowledge-search_es-data || true

# Deploy to Docker Compose
deploy:
	@docker-compose up -d
	@echo "Waiting for services..."
	@sleep 10
	@docker-compose ps

# View logs
logs:
	@docker-compose logs -f

# Switch to Vertex AI
use-vertex:
	@sed -i '' 's/EMBEDDING_PROVIDER=.*/EMBEDDING_PROVIDER=vertex/' .env
	@sed -i '' 's/LLM_PROVIDER=.*/LLM_PROVIDER=vertex/' .env
	@echo "⚠️  Don't forget to set GOOGLE_PROJECT_ID and credentials!"

# Switch to Azure
use-azure:
	@sed -i '' 's/EMBEDDING_PROVIDER=.*/EMBEDDING_PROVIDER=azure/' .env
	@sed -i '' 's/LLM_PROVIDER=.*/LLM_PROVIDER=azure/' .env
	@echo "⚠️  Don't forget to set AZURE_OPENAI_* environment variables!"

# Switch to local
use-local:
	@sed -i '' 's/EMBEDDING_PROVIDER=.*/EMBEDDING_PROVIDER=local/' .env
	@sed -i '' 's/LLM_PROVIDER=.*/LLM_PROVIDER=local/' .env
	@echo "✓ Using local models (free)"

# Help
help:
	@echo "RAG Knowledge Search - Available Commands:"
	@echo ""
	@echo "  make setup        - Initial setup (run once)"
	@echo "  make start        - Start all services"
	@echo "  make stop         - Stop all services"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up containers"
	@echo "  make deploy       - Deploy with Docker Compose"
	@echo "  make logs         - View service logs"
	@echo "  make use-local    - Switch to local models (free)"
	@echo "  make use-vertex   - Switch to Google Vertex AI"
	@echo "  make use-azure    - Switch to Azure OpenAI"
	@echo ""