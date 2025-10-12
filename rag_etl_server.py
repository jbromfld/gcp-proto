"""
Simple web server for ETL operations on Cloud Run
Responds to HTTP POST /trigger to run ETL pipeline
"""
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
            embedder = EmbeddingFactory.create(EMBEDDING_CONFIGS['local_minilm'])
        
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
async def trigger_etl(background_tasks: BackgroundTasks):
    """Trigger ETL pipeline execution"""
    if not etl_pipeline:
        return {"error": "ETL pipeline not initialized"}, 500
    
    # Get URLs from environment
    scrape_urls_str = os.environ.get('SCRAPE_URLS', '')
    scrape_urls = [url.strip() for url in scrape_urls_str.split(',') if url.strip()]
    
    if not scrape_urls:
        return {"error": "No SCRAPE_URLS configured"}, 400
    
    # Run ETL in background
    background_tasks.add_task(run_etl, scrape_urls)
    
    return {
        "status": "started",
        "message": "ETL pipeline triggered",
        "urls": scrape_urls
    }

def run_etl(urls: list):
    """Run the ETL pipeline"""
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
        "pipeline_initialized": etl_pipeline is not None,
        "scrape_urls": os.environ.get('SCRAPE_URLS', '').split(',')
    }

