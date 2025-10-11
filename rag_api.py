"""
FastAPI backend for RAG system
Provides REST API for queries, feedback, and metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta
import logging
import os

from elasticsearch import Elasticsearch

from rag_embeddings import EmbeddingFactory, EMBEDDING_CONFIGS
from rag_llm_abstraction import LLMFactory, LLM_CONFIGS
from rag_service import RAGService, HybridSearchEngine
from rag_evaluation import FeedbackStore, FeedbackType
from rag_etl_pipeline import (
    ETLPipeline, ElasticsearchIndexer, DocumentChunker, ScheduledETL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG Knowledge Search API",
    description="Semantic search over internal documentation with LLM-powered answers",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_service: Optional[RAGService] = None
feedback_store: Optional[FeedbackStore] = None
scheduled_etl: Optional[ScheduledETL] = None


# === Request/Response Models ===

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's search query", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    top_k: int = Field(3, ge=1, le=10, description="Number of results to retrieve")


class DocumentResult(BaseModel):
    title: str
    content: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    retrieved_docs: List[DocumentResult]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    cost_estimate: Optional[float] = None


class FeedbackRequest(BaseModel):
    query_id: str
    feedback_type: str = Field(..., description="thumbs_up, thumbs_down, or rating")
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = Field(None, description="Optional feedback comment")


class IngestRequest(BaseModel):
    urls: List[str] = Field(..., description="URLs to scrape and index")
    max_pages_per_url: int = Field(50, ge=1, le=200)


class MetricsResponse(BaseModel):
    period_start: str = Field(..., description="Start time of the metrics period")
    period_end: str = Field(..., description="End time of the metrics period")
    total_queries: int = Field(..., description="Total number of queries processed")
    thumbs_up_rate: float = Field(..., description="Rate of positive feedback")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    p95_latency_ms: float = Field(..., description="95th percentile latency in milliseconds")
    total_cost: float = Field(..., description="Total cost of queries")


class ConfigRequest(BaseModel):
    embedding_provider: str = Field(..., description="vertex, azure, or local")
    llm_provider: str = Field(..., description="vertex, azure, or local")
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None


# === Startup/Shutdown ===

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_service, feedback_store, scheduled_etl
    
    try:
        # Connect to Elasticsearch
        es_url = os.environ.get('ELASTICSEARCH_URL', 'http://elasticsearch:9200')
        es_client = Elasticsearch([es_url])
        logging.info(f"Connecting to Elasticsearch at {es_url}")
        
        # Default to local models for POC
        embedding_config = EMBEDDING_CONFIGS['local_minilm']
        llm_config = LLM_CONFIGS['local_llama']
        
        # Create components
        embedder = EmbeddingFactory.create(embedding_config)
        llm = LLMFactory.create(llm_config)
        
        # Create index if needed
        indexer = ElasticsearchIndexer(es_client, index_name="knowledge_base")
        indexer.create_index(embedding_dim=embedder.dimensions)
        
        # Create search engine
        search_engine = HybridSearchEngine(
            es_client=es_client,
            embedder=embedder,
            index_name="knowledge_base"
        )
        
        # Create feedback store
        feedback_store = FeedbackStore(es_client)
        
        # Create RAG service
        rag_service = RAGService(
            search_engine=search_engine,
            llm=llm,
            feedback_store=feedback_store
        )
        
        # Setup scheduled ETL (24 hour cadence)
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        etl_pipeline = ETLPipeline(embedder, indexer, chunker)
        
        scrape_urls = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://fastapi.tiangolo.com/"
        ]
        
        scheduled_etl = ScheduledETL(
            pipeline=etl_pipeline,
            scrape_urls=scrape_urls,
            interval_hours=24,
            max_pages_per_url=50
        )
        
        logger.info("RAG system initialized successfully")
        logger.info(f"Using embedding model: {embedding_config.model_name}")
        logger.info(f"Using LLM: {llm_config.model_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        raise


# === API Endpoints ===

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "RAG Knowledge Search",
        "version": "1.0.0"
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a user query through the RAG pipeline
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        response = rag_service.query(
            user_query=request.query,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        return QueryResponse(
            query_id=response.query_id,
            answer=response.answer,
            retrieved_docs=[
                DocumentResult(
                    title=doc.title,
                    content=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    source_url=doc.source_url,
                    score=doc.score
                )
                for doc in response.retrieved_docs
            ],
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            model_used=response.model_used,
            cost_estimate=response.cost_estimate
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Submit user feedback for a query
    """
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        # Map feedback type string to enum
        feedback_type_map = {
            "thumbs_up": FeedbackType.THUMBS_UP,
            "thumbs_down": FeedbackType.THUMBS_DOWN,
            "rating": FeedbackType.RATING
        }
        
        feedback_type = feedback_type_map.get(request.feedback_type)
        if not feedback_type:
            raise HTTPException(status_code=400, detail="Invalid feedback type")
        
        feedback_store.add_feedback(
            query_id=request.query_id,
            feedback_type=feedback_type,
            rating=request.rating,
            comment=request.comment
        )
        
        return {"status": "success", "message": "Feedback recorded"}
    
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics_endpoint(days: int = 7):
    """
    Get system metrics for the specified time period
    """
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        metrics = feedback_store.compute_metrics(start_time, end_time)
        
        return MetricsResponse(
            period_start=metrics.period_start,
            period_end=metrics.period_end,
            total_queries=metrics.total_queries,
            thumbs_up_rate=metrics.thumbs_up_rate,
            avg_response_time_ms=metrics.avg_response_time_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            total_cost=metrics.total_cost
        )
    
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/popular-queries")
async def popular_queries_endpoint(limit: int = 10):
    """Get most popular queries"""
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        queries = feedback_store.get_top_queries(limit=limit)
        return {"popular_queries": queries}
    
    except Exception as e:
        logger.error(f"Failed to get popular queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/low-rated-queries")
async def low_rated_queries_endpoint(limit: int = 10):
    """Get queries with negative feedback"""
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        queries = feedback_store.get_low_rated_queries(limit=limit)
        return {"low_rated_queries": queries}
    
    except Exception as e:
        logger.error(f"Failed to get low-rated queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger manual ingestion of URLs
    """
    if not scheduled_etl:
        raise HTTPException(status_code=503, detail="ETL system not initialized")
    
    try:
        # Run in background
        background_tasks.add_task(
            scheduled_etl.pipeline.run_scrape_and_index,
            request.urls,
            request.max_pages_per_url
        )
        
        return {
            "status": "processing",
            "message": f"Started ingestion of {len(request.urls)} URLs"
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger-etl")
async def trigger_etl_endpoint(background_tasks: BackgroundTasks):
    """
    Manually trigger scheduled ETL run
    """
    if not scheduled_etl:
        raise HTTPException(status_code=503, detail="ETL system not initialized")
    
    background_tasks.add_task(scheduled_etl.run_once)
    
    return {
        "status": "scheduled",
        "message": "ETL run scheduled"
    }


@app.post("/api/configure")
async def configure_endpoint(request: ConfigRequest):
    """
    Reconfigure embedding and LLM providers
    (Requires restart in production)
    """
    return {
        "status": "accepted",
        "message": "Configuration updated. Restart required.",
        "config": {
            "embedding_provider": request.embedding_provider,
            "llm_provider": request.llm_provider
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)