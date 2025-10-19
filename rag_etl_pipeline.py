"""
ETL Pipeline for scraping, chunking, embedding, and indexing documents
Supports scheduled re-scraping (default: 24 hours)
"""

import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import fnmatch
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
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, nav, footer
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else url
            
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
        
        documents: List[Document] = []
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


class RecursiveWebScraper:
    """
    Enhanced scraper with recursive crawling, depth limits, and pattern matching
    Supports wildcards like docs.python.com/* for full site scraping
    """
    
    def __init__(self, max_depth: int = 2, max_pages: int = 50, 
                 include_patterns: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or ['*/search*', '*/login*', '*/logout*']
        self.visited: Set[str] = set()
        self.documents: List[Document] = []
    
    def _should_crawl_url(self, url: str, base_domain: str) -> bool:
        """Determine if a URL should be crawled based on patterns and domain"""
        parsed = urlparse(url)
        
        # Must be same domain
        if not parsed.netloc or parsed.netloc != base_domain:
            return False
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(url, pattern):
                return False
        
        # If include patterns specified, URL must match at least one
        if self.include_patterns:
            return any(fnmatch.fnmatch(url, pattern) for pattern in self.include_patterns)
        
        return True
    
    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract and normalize links from a page"""
        links: List[str] = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Ensure href is a string
            if not isinstance(href, str):
                continue
                
            # Skip anchors, mailto, tel, javascript
            if href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                continue
            
            # Make absolute URL
            absolute_url = urljoin(current_url, href)
            
            # Remove fragment
            absolute_url = absolute_url.split('#')[0]
            
            links.append(absolute_url)
        
        return [link for link in links if isinstance(link, str)]
    
    def crawl(self, start_url: str) -> List[Document]:
        """
        Recursively crawl starting from start_url
        
        Examples:
            - crawl("https://docs.python.org/3/tutorial/") # Single page
            - crawl("https://docs.python.org/3/*") # All pages under /3/
        """
        # Handle wildcard URLs
        if '*' in start_url:
            # Convert wildcard to base URL and pattern
            base_url = start_url.split('*')[0].rstrip('/')
            self.include_patterns.append(start_url)
            logger.info(f"Wildcard detected: Starting from {base_url} with pattern {start_url}")
        else:
            base_url = start_url
        
        base_domain = urlparse(base_url).netloc
        
        # Start crawling
        self._crawl_recursive(base_url, base_domain, depth=0)
        
        logger.info(f"Crawl complete: {len(self.documents)} documents, {len(self.visited)} pages visited")
        return self.documents
    
    def _crawl_recursive(self, url: str, base_domain: str, depth: int):
        """Recursively crawl pages"""
        # Check limits
        if depth > self.max_depth:
            return
        if len(self.documents) >= self.max_pages:
            return
        if url in self.visited:
            return
        
        # Mark as visited
        self.visited.add(url)
        
        # Scrape the page
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Create document
            doc = self._create_document(url, soup)
            if doc:
                self.documents.append(doc)
                logger.info(f"[Depth {depth}] Scraped: {doc.title[:60]}... ({len(self.documents)}/{self.max_pages})")
            
            # Extract and crawl links if not at max depth
            if depth < self.max_depth and len(self.documents) < self.max_pages:
                links = self._extract_links(soup, url)
                
                for link in links:
                    if self._should_crawl_url(link, base_domain):
                        time.sleep(0.5)  # Rate limiting
                        self._crawl_recursive(link, base_domain, depth + 1)
                        
                        if len(self.documents) >= self.max_pages:
                            break
        
        except Exception as e:
            logger.warning(f"Error crawling {url}: {e}")
    
    def _create_document(self, url: str, soup: BeautifulSoup) -> Optional[Document]:
        """Create a Document from scraped content"""
        try:
            # Remove script, style, nav, footer
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Get title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else url
            
            # Get main content
            main = soup.find('main') or soup.find('article') or soup.find('body')
            if not main:
                return None
            
            content = main.get_text(separator='\n', strip=True)
            
            # Skip if too short
            if len(content) < 100:
                return None
            
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
            logger.error(f"Error creating document from {url}: {e}")
            return None


class DocumentChunker:
    """Document chunker with token-aware splitting to avoid LLM limits"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, max_tokens: int = 15000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_tokens = max_tokens
    
    def chunk_document(self, doc: Document) -> List[str]:
        """Chunk document with token limits to avoid LLM token limits"""
        words = doc.content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunk_text = ' '.join(chunk_words)
                
                # Check if chunk exceeds token limit (rough estimate: 4 chars per token)
                estimated_tokens = len(chunk_text) / 4
                if estimated_tokens > self.max_tokens:
                    # Split large chunk further
                    sub_chunks = self._split_large_chunk(chunk_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_text)
        
        return chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a chunk that exceeds token limits"""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed limit, finalize current chunk
            if current_length + sentence_length > self.max_tokens * 4 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    


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
        
        try:
            self.es.indices.create(index=self.index_name, mappings=mapping["mappings"])
            logger.info(f"Created index {self.index_name}")
        except Exception as e:
            if 'resource_already_exists_exception' in str(e):
                logger.info(f"Index {self.index_name} already exists")
            else:
                raise
    
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
        except Exception:
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
        """Process documents: chunk, embed, index
        
        Returns:
            Tuple of (doc_count, chunk_count)
        """
        all_chunks = []
        processed_docs = 0
        
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
            
            processed_docs += 1
            logger.info(f"Processed {doc.title[:50]}: {len(chunk_texts)} chunks")
        
        # Bulk index
        if all_chunks:
            self.indexer.index_chunks(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} total chunks from {processed_docs} documents")
        
        return (processed_docs, len(all_chunks))
    
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
    
    chunker = DocumentChunker(chunk_size=500, overlap=50, max_tokens=15000)
    
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

