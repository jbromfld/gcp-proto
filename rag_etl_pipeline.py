"""
ETL Pipeline for scraping, chunking, embedding, and indexing documents
Supports scheduled re-scraping (default: 24 hours)
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
import numpy as np

from rag_embeddings import EmbeddingProvider, EmbeddingFactory, EMBEDDING_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation"""
    doc_id: str
    title: str
    content: str
    source_url: str
    timestamp: str
    content_hash: str
    metadata: Optional[Dict] = None


@dataclass
class Chunk:
    """Document chunk with embeddings"""
    chunk_id: str
    doc_id: str
    title: str
    content: str
    embedding: np.ndarray
    source_url: str
    timestamp: str
    chunk_index: int
    total_chunks: int


class DocumentScraper:
    """Scrapes documentation websites"""
    
    @staticmethod
    def scrape_page(url: str) -> Optional[Document]:
        """Scrape a single page"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, nav, footer
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get title
            title = soup.find('title')
            title = title.get_text().strip() if title else url
            
            # Get main content
            main = soup.find('main') or soup.find('article') or soup.find('body')
            if not main:
                return None
            
            content = main.get_text(separator='\n', strip=True)
            
            # Create content hash for change detection
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            doc_id = hashlib.md5(url.encode()).hexdigest()
            
            return Document(
                doc_id=doc_id,
                title=title,
                content=content,
                source_url=url,
                timestamp=datetime.now().isoformat(),
                content_hash=content_hash
            )
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    @staticmethod
    def scrape_sitemap(base_url: str, max_pages: int = 100) -> List[Document]:
        """Scrape multiple pages from a sitemap or crawl"""
        # For POC, we'll implement simple link following
        # In production, use Scrapy or sitemap.xml parsing
        
        documents = []
        visited = set()
        to_visit = [base_url]
        
        while to_visit and len(documents) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            doc = DocumentScraper.scrape_page(url)
            
            if doc:
                documents.append(doc)
                logger.info(f"Scraped: {doc.title[:50]}... ({len(documents)}/{max_pages})")
            
            # Rate limiting
            time.sleep(0.5)
        
        return documents


class DocumentChunker:
    """Chunks documents with overlap for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, doc: Document) -> List[str]:
        """Chunk document with overlap"""
        # Split by sentences (simple approach)
        sentences = self._split_sentences(doc.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save chunk
                chunks.append(' '.join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter"""
        import re
        # Simple sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class ElasticsearchIndexer:
    """Manages Elasticsearch index for vector search"""
    
    def __init__(self, es_client: Elasticsearch, index_name: str = "knowledge_base"):
        self.es = es_client
        self.index_name = index_name
    
    def create_index(self, embedding_dim: int):
        """Create index with vector field"""
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100
                        }
                    },
                    "source_url": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "content_hash": {"type": "keyword"}
                }
            }
        }
        
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists")
        else:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index {self.index_name}")
    
    def index_chunks(self, chunks: List[Chunk]):
        """Bulk index chunks"""
        from elasticsearch.helpers import bulk
        
        actions = []
        for chunk in chunks:
            action = {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "embedding": chunk.embedding.tolist(),
                    "source_url": chunk.source_url,
                    "timestamp": chunk.timestamp,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks
                }
            }
            actions.append(action)
        
        success, failed = bulk(self.es, actions, raise_on_error=False)
        logger.info(f"Indexed {success} chunks, {failed} failed")
    
    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get stored content hash for change detection"""
        try:
            result = self.es.search(
                index=self.index_name,
                body={
                    "query": {"term": {"doc_id": doc_id}},
                    "size": 1,
                    "_source": ["content_hash"]
                }
            )
            if result['hits']['hits']:
                return result['hits']['hits'][0]['_source'].get('content_hash')
        except:
            pass
        return None
    
    def delete_document_chunks(self, doc_id: str):
        """Delete all chunks for a document"""
        self.es.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}}
        )


class ETLPipeline:
    """Complete ETL pipeline"""
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        indexer: ElasticsearchIndexer,
        chunker: DocumentChunker
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.chunker = chunker
    
    def process_documents(self, documents: List[Document], force_reindex: bool = False):
        """Process documents: chunk, embed, index"""
        all_chunks = []
        
        for doc in documents:
            # Check if document changed
            if not force_reindex:
                stored_hash = self.indexer.get_document_hash(doc.doc_id)
                if stored_hash == doc.content_hash:
                    logger.info(f"Skipping unchanged document: {doc.title[:50]}")
                    continue
                elif stored_hash:
                    logger.info(f"Document changed, re-indexing: {doc.title[:50]}")
                    self.indexer.delete_document_chunks(doc.doc_id)
            
            # Chunk document
            chunk_texts = self.chunker.chunk_document(doc)
            
            if not chunk_texts:
                logger.warning(f"No chunks for document: {doc.title}")
                continue
            
            # Embed chunks
            embeddings = self.embedder.embed_texts(chunk_texts)
            
            # Create Chunk objects
            for idx, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                chunk_id = f"{doc.doc_id}_{idx}"
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=text,
                    embedding=embedding,
                    source_url=doc.source_url,
                    timestamp=doc.timestamp,
                    chunk_index=idx,
                    total_chunks=len(chunk_texts)
                )
                all_chunks.append(chunk)
            
            logger.info(f"Processed {doc.title[:50]}: {len(chunk_texts)} chunks")
        
        # Bulk index
        if all_chunks:
            self.indexer.index_chunks(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} total chunks")
        
        return len(all_chunks)
    
    def run_scrape_and_index(self, urls: List[str], max_pages_per_url: int = 50):
        """Scrape URLs and index"""
        all_docs = []
        
        for url in urls:
            logger.info(f"Scraping {url}")
            docs = DocumentScraper.scrape_sitemap(url, max_pages=max_pages_per_url)
            all_docs.extend(docs)
        
        logger.info(f"Scraped {len(all_docs)} total documents")
        
        chunks_indexed = self.process_documents(all_docs)
        return len(all_docs), chunks_indexed


class ScheduledETL:
    """Scheduled ETL runner with 24-hour cadence"""
    
    def __init__(
        self,
        pipeline: ETLPipeline,
        scrape_urls: List[str],
        interval_hours: int = 24,
        max_pages_per_url: int = 50
    ):
        self.pipeline = pipeline
        self.scrape_urls = scrape_urls
        self.interval_hours = interval_hours
        self.max_pages_per_url = max_pages_per_url
        self.last_run: Optional[datetime] = None
    
    def should_run(self) -> bool:
        """Check if enough time has passed"""
        if self.last_run is None:
            return True
        
        elapsed = datetime.now() - self.last_run
        return elapsed >= timedelta(hours=self.interval_hours)
    
    def run_once(self):
        """Run ETL once"""
        logger.info("=" * 60)
        logger.info(f"Starting scheduled ETL run at {datetime.now()}")
        logger.info("=" * 60)
        
        try:
            docs, chunks = self.pipeline.run_scrape_and_index(
                self.scrape_urls,
                max_pages_per_url=self.max_pages_per_url
            )
            
            self.last_run = datetime.now()
            
            logger.info("=" * 60)
            logger.info(f"ETL completed: {docs} docs, {chunks} chunks")
            logger.info(f"Next run: {self.last_run + timedelta(hours=self.interval_hours)}")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.error(f"ETL failed: {e}", exc_info=True)
            return False
    
    def run_loop(self):
        """Run ETL in a loop (for background process)"""
        import time
        
        while True:
            if self.should_run():
                self.run_once()
            else:
                next_run = self.last_run + timedelta(hours=self.interval_hours)
                wait_seconds = (next_run - datetime.now()).total_seconds()
                logger.info(f"Waiting {wait_seconds/3600:.1f} hours until next run")
                time.sleep(min(wait_seconds, 3600))  # Check every hour max


# Example usage
if __name__ == "__main__":
    # Setup
    es_url = os.environ.get('ELASTICSEARCH_URL', 'http://elasticsearch:9200')
    es_client = Elasticsearch([es_url])
    logging.info(f"Connected to Elasticsearch at {es_url}")
    
    # Use local embeddings for testing
    embedding_config = EMBEDDING_CONFIGS['local_minilm']
    embedder = EmbeddingFactory.create(embedding_config)
    
    indexer = ElasticsearchIndexer(es_client, index_name="knowledge_base")
    indexer.create_index(embedding_dim=embedder.dimensions)
    
    chunker = DocumentChunker(chunk_size=500, overlap=50)
    
    pipeline = ETLPipeline(embedder, indexer, chunker)
    
    # URLs to scrape (public documentation sites)
    scrape_urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://fastapi.tiangolo.com/tutorial/"
    ]
    
    # Option 1: Run once
    docs, chunks = pipeline.run_scrape_and_index(scrape_urls, max_pages_per_url=10)
    print(f"Indexed {docs} documents, {chunks} chunks")
    
    # Option 2: Run on schedule (24 hours)
    # scheduler = ScheduledETL(pipeline, scrape_urls, interval_hours=24)
    # scheduler.run_loop()  # Runs forever