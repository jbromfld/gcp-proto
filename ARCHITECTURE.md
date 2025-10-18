# RAG System Architecture

## System Overview

This is a production-ready Retrieval-Augmented Generation (RAG) system that enables semantic search over internal documentation with AI-powered answers. The system combines vector search with large language models to provide accurate, contextual responses based on your knowledge base.

The system supports both local development (using Ollama and sentence-transformers) and GCP production deployment (using Vertex AI and managed Elasticsearch), with automatic document ingestion, chunking, and vector indexing. It features a modern web UI for querying, comprehensive admin tools for source management, and robust infrastructure with auto-scaling Cloud Run services connected via VPC to a self-hosted Elasticsearch cluster.

The architecture diagram shows the complete data flow from document ingestion through web scraping, chunking with embeddings, vector storage, and finally query processing with LLM generation - all designed to stay within token limits and provide accurate, contextual answers based on your knowledge base.

## Architecture Flow

```mermaid
graph TB
    %% User Interface Layer
    UI[Streamlit UI<br/>rag_ui.py] --> API[FastAPI Backend<br/>rag_api.py]
    
    %% Core RAG Engine
    API --> RAG[RAG Service<br/>rag_service.py]
    RAG --> EMB[Embedding Service<br/>rag_embeddings.py]
    RAG --> LLM[LLM Service<br/>rag_llm_abstraction.py]
    
    %% Vector Search
    EMB --> ES[Elasticsearch<br/>Vector Store]
    RAG --> ES
    
    %% Document Processing
    ETL[ETL Pipeline<br/>rag_etl_pipeline.py] --> CHUNK[Document Chunker<br/>300 words, 30 overlap]
    CHUNK --> EMB
    ETL --> ES
    
    %% Data Sources
    SCRAPER[Web Scraper<br/>Recursive crawling] --> ETL
    SOURCES[Source Manager<br/>rag_sources.py] --> SCRAPER
    
    %% Providers (Local vs GCP)
    subgraph "Local Development"
        LOCAL_LLM[Ollama<br/>llama3.2]
        LOCAL_EMB[sentence-transformers<br/>all-mpnet-base-v2]
        LOCAL_ES[Elasticsearch Docker<br/>Port 9200]
    end
    
    subgraph "GCP Production"
        VERTEX_LLM[Vertex AI<br/>gemini-2.5-pro]
        VERTEX_EMB[Vertex AI<br/>text-embedding-004]
        GCE_ES[Elasticsearch GCE<br/>e2-medium VM]
    end
    
    %% Provider Selection
    LLM -.-> LOCAL_LLM
    LLM -.-> VERTEX_LLM
    EMB -.-> LOCAL_EMB
    EMB -.-> VERTEX_EMB
    ES -.-> LOCAL_ES
    ES -.-> GCE_ES
    
    %% Infrastructure
    subgraph "GCP Infrastructure"
        CR_API[Cloud Run API<br/>2 CPU, 4GB RAM]
        CR_UI[Cloud Run UI<br/>1 CPU, 2GB RAM]
        CR_ETL[Cloud Run ETL<br/>2 CPU, 4GB RAM]
        VPC[VPC Connector<br/>e2-micro]
        SCHED[Cloud Scheduler<br/>Daily ETL trigger]
    end
    
    %% Deployment
    API -.-> CR_API
    UI -.-> CR_UI
    ETL -.-> CR_ETL
    CR_API --> VPC
    CR_ETL --> VPC
    VPC --> GCE_ES
    SCHED --> CR_ETL
    
    %% Feedback Loop
    API --> FEEDBACK[Feedback Store<br/>rag_evaluation.py]
    FEEDBACK --> METRICS[Metrics & Analytics]
    
    %% Styling
    classDef userLayer fill:#e1f5fe
    classDef coreLayer fill:#f3e5f5
    classDef dataLayer fill:#e8f5e8
    classDef infraLayer fill:#fff3e0
    classDef providerLayer fill:#fce4ec
    
    class UI,API userLayer
    class RAG,EMB,LLM coreLayer
    class ES,ETL,CHUNK,SCRAPER,SOURCES dataLayer
    class CR_API,CR_UI,CR_ETL,VPC,SCHED infraLayer
    class LOCAL_LLM,LOCAL_EMB,LOCAL_ES,VERTEX_LLM,VERTEX_EMB,GCE_ES providerLayer
```

## Key Components

### Core Services
- **RAG Service**: Orchestrates retrieval and generation
- **Embedding Service**: Multi-provider embeddings (local/GCP)
- **LLM Service**: Multi-provider language models (Ollama/Vertex AI)
- **ETL Pipeline**: Document processing and indexing

### Data Flow
1. **Ingestion**: Web scraper → Document chunker → Embeddings → Vector store
2. **Query**: User query → Embedding → Vector search → LLM generation → Response
3. **Feedback**: User feedback → Metrics → System improvement

### Deployment Options
- **Local**: Docker Compose with Ollama and sentence-transformers
- **GCP**: Cloud Run services with Vertex AI and GCE Elasticsearch

## Configuration

### Chunking Strategy
- **Size**: 300 words (reduced from 500 to avoid token limits)
- **Overlap**: 30 words for context preservation
- **Token Limit**: Stays under 20,000 tokens for Vertex AI embeddings

### Model Configuration
- **Local LLM**: Ollama llama3.2 (3B parameters)
- **Production LLM**: Vertex AI gemini-2.5-pro
- **Embeddings**: 768 dimensions for compatibility

### Infrastructure
- **Elasticsearch**: 8.11 with HNSW indexing
- **VPC**: Connects Cloud Run to GCE Elasticsearch
- **Scaling**: Auto-scaling Cloud Run (0-10 instances)

---

*Last Updated: October 18, 2025*
