"""
Simple web server for ETL operations on Cloud Run
Responds to HTTP POST /trigger to run ETL pipeline
"""
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks
from rag_etl_pipeline import ETLPipeline, ElasticsearchIndexer, DocumentChunker
from rag_embeddings import EmbeddingFactory, EMBEDDING_CONFIGS
from elasticsearch_client import create_elasticsearch_client
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG ETL Service")

# Global ETL pipeline
etl_pipeline = None

@app.on_event("startup")
async def startup():
    """Initialize ETL pipeline on startup"""
    global etl_pipeline
    
    try:
        # Connect to Elasticsearch
        es_client = create_elasticsearch_client()
        logger.info("Connected to Elasticsearch")
        
        # Initialize embedding provider
        embedding_provider = os.environ.get('EMBEDDING_PROVIDER', 'vertex')
        if embedding_provider == 'vertex':
            embedder = EmbeddingFactory.create(EMBEDDING_CONFIGS['vertex_gecko'])
        else:
            embedder = EmbeddingFactory.create(EMBEDDING_CONFIGS['local_mpnet'])  # 768 dims (matches Vertex AI)
        
        logger.info(f"Initialized embeddings: {embedding_provider}")
        
        # Create index
        indexer = ElasticsearchIndexer(es_client)
        indexer.create_index(embedding_dim=embedder.dimensions)
        
        # Setup pipeline
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        etl_pipeline = ETLPipeline(embedder, indexer, chunker)
        
        logger.info("ETL pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ETL pipeline: {e}")
        raise

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-etl"}

@app.post("/trigger")
async def trigger_etl(background_tasks: BackgroundTasks, urls: Optional[List[str]] = None):
    """
    Trigger ETL pipeline execution
    URLs must be provided in request body
    """
    if not etl_pipeline:
        return {"error": "ETL pipeline not initialized"}, 500
    
    if not urls:
        return {"error": "No URLs provided in request body"}, 400
    
    # Run ETL in background
    background_tasks.add_task(run_etl, urls)
    
    return {
        "status": "started",
        "message": "ETL pipeline triggered",
        "urls": urls
    }

def run_etl(urls: list):
    """Run the ETL pipeline"""
    assert etl_pipeline is not None, "ETL pipeline not initialized"
    try:
        logger.info(f"Starting ETL for {len(urls)} URLs")
        docs, chunks = etl_pipeline.run_scrape_and_index(
            urls,
            max_pages_per_url=10
        )
        logger.info(f"ETL complete: {docs} docs, {chunks} chunks")
    except Exception as e:
        logger.error(f"ETL failed: {e}")

@app.get("/status")
async def status():
    """Get ETL pipeline status"""
    return {
        "pipeline_initialized": etl_pipeline is not None
    }

