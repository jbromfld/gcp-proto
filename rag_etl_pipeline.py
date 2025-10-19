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


class TokenCounter:
    """Accurate token counting for different models"""
    
    # Token estimation ratios for different content types
    TOKEN_RATIOS = {
        'code': 2.5,      # Code has more tokens per character
        'text': 4.0,      # Regular text
        'markdown': 3.5,  # Markdown with formatting
        'html': 2.8,      # HTML with tags
    }
    
    @staticmethod
    def estimate_tokens(text: str, content_type: str = 'text') -> int:
        """
        Estimate token count for text based on content type.
        More accurate than simple character/4 ratio.
        """
        if not text:
            return 0
            
        # Clean text for better estimation
        cleaned_text = TokenCounter._clean_text_for_counting(text)
        
        # Base estimation
        char_count = len(cleaned_text)
        ratio = TokenCounter.TOKEN_RATIOS.get(content_type, 4.0)
        base_tokens = char_count / ratio
        
        # Adjust for content complexity
        complexity_factor = TokenCounter._calculate_complexity_factor(cleaned_text)
        
        return int(base_tokens * complexity_factor)
    
    @staticmethod
    def _clean_text_for_counting(text: str) -> str:
        """Clean text for more accurate token counting"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags (but keep content)
        text = re.sub(r'<[^>]+>', '', text)
        # Remove markdown formatting
        text = re.sub(r'[*_`#]+', '', text)
        return text.strip()
    
    @staticmethod
    def _calculate_complexity_factor(text: str) -> float:
        """Calculate complexity factor based on text characteristics"""
        factor = 1.0
        
        # More punctuation = more tokens
        punct_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        if punct_ratio > 0.1:
            factor += 0.2
        
        # More numbers = more tokens
        num_ratio = len(re.findall(r'\d', text)) / max(len(text), 1)
        if num_ratio > 0.1:
            factor += 0.1
        
        # More special characters = more tokens
        special_ratio = len(re.findall(r'[^\w\s\.,!?;:]', text)) / max(len(text), 1)
        if special_ratio > 0.05:
            factor += 0.1
        
        return min(factor, 1.5)  # Cap at 1.5x
    
    @staticmethod
    def is_within_token_limit(text: str, max_tokens: int, content_type: str = 'text') -> bool:
        """Check if text is within token limit"""
        return TokenCounter.estimate_tokens(text, content_type) <= max_tokens


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
    """Advanced document chunker with dynamic tokenization and smart boundaries"""
    
    def __init__(self, 
                 chunk_size: int = 100, 
                 overlap: int = 10, 
                 max_tokens: int = 15000,
                 min_chunk_size: int = 20,
                 content_type: str = 'text'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_tokens = max_tokens
        self.min_chunk_size = min_chunk_size
        self.content_type = content_type
    
    def chunk_document(self, doc: Document) -> List[str]:
        """Chunk document with smart boundaries and token limits"""
        # Detect content type for better token estimation
        content_type = self._detect_content_type(doc.content)
        
        # Split into semantic units (paragraphs, sections, etc.)
        semantic_units = self._split_into_semantic_units(doc.content)
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_word_count = 0
        
        for unit in semantic_units:
            unit_words = len(unit.split())
            potential_chunk = ' '.join(current_chunk + [unit])
            
            # Check both word count and token limits
            estimated_tokens = TokenCounter.estimate_tokens(potential_chunk, content_type)
            
            # If adding this unit would exceed limits, finalize current chunk
            if (current_word_count + unit_words > self.chunk_size or 
                estimated_tokens > self.max_tokens) and current_chunk:
                
                # Validate and save current chunk
                chunk_text = ' '.join(current_chunk)
                if self._validate_chunk(chunk_text, content_type):
                    chunks.append(chunk_text)
                
                # Create overlap for next chunk
                current_chunk = self._create_overlap(current_chunk)
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(unit)
            current_word_count += unit_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self._validate_chunk(chunk_text, content_type):
                chunks.append(chunk_text)
        
        return chunks
    
    def _detect_content_type(self, content: str) -> str:
        """Detect content type for better token estimation"""
        # Check for code patterns
        if re.search(r'```|def |class |import |from ', content):
            return 'code'
        # Check for HTML
        elif re.search(r'<[^>]+>', content):
            return 'html'
        # Check for markdown
        elif re.search(r'#{1,6}\s|[*_`]|\[.*\]\(.*\)', content):
            return 'markdown'
        else:
            return 'text'
    
    def _split_into_semantic_units(self, content: str) -> List[str]:
        """Split content into semantic units (paragraphs, sections, etc.)"""
        # First, try to split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        units = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Special handling for feature lists
            if self._is_feature_list(paragraph):
                # Keep feature lists as single units to preserve context
                units.append(paragraph)
            elif len(paragraph.split()) > self.chunk_size * 2:
                sentences = self._split_sentences(paragraph)
                units.extend(sentences)
            else:
                units.append(paragraph)
        
        return units
    
    def _is_feature_list(self, text: str) -> bool:
        """Detect if text is a feature list that should be kept together"""
        feature_indicators = [
            r'features?\s*:',
            r'benefits?\s*:',
            r'advantages?\s*:',
            r'capabilities?\s*:',
            r'key\s+features?',
            r'what\s+you\s+get',
            r'gives\s+you',
            r'provides?\s+you'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in feature_indicators)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting that handles various formats"""
        # Handle different sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up and filter
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_overlap(self, current_chunk: List[str]) -> List[str]:
        """Create overlap from current chunk for next chunk"""
        overlap_units = []
        overlap_word_count = 0
        
        # Take last few units for overlap
        for unit in reversed(current_chunk):
            unit_words = len(unit.split())
            if overlap_word_count + unit_words <= self.overlap:
                overlap_units.insert(0, unit)
                overlap_word_count += unit_words
            else:
                break
        
        return overlap_units
    
    def _validate_chunk(self, chunk: str, content_type: str) -> bool:
        """Validate that chunk meets requirements"""
        word_count = len(chunk.split())
        token_count = TokenCounter.estimate_tokens(chunk, content_type)
        
        # Check minimum size
        if word_count < self.min_chunk_size:
            return False
        
        # Check token limit
        if token_count > self.max_tokens:
            logger.warning(f"Chunk exceeds token limit: {token_count} > {self.max_tokens}")
            return False
        
        return True
    


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
    
    chunker = DocumentChunker(
        chunk_size=100, 
        overlap=10, 
        max_tokens=15000,
        min_chunk_size=20
    )
    
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


def test_chunking_with_large_document():
    """Test the chunker with a large document to ensure token limits are respected"""
    # Create a test document that simulates the Python glossary
    large_content = """
    # Python Glossary Test Document
    
    This is a test document that simulates a large documentation page like the Python glossary.
    It contains multiple sections with detailed explanations that would normally exceed token limits.
    
    ## Section 1: Basic Concepts
    
    Python is a high-level programming language known for its simplicity and readability. 
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    
    ## Section 2: Data Types
    
    Python has several built-in data types including integers, floats, strings, lists, tuples, dictionaries, and sets.
    Each data type has specific characteristics and methods that make them suitable for different use cases.
    
    ## Section 3: Control Structures
    
    Python provides various control structures including if statements, for loops, while loops, and try-except blocks.
    These structures allow developers to control the flow of execution in their programs.
    
    ## Section 4: Functions and Classes
    
    Functions in Python are defined using the def keyword and can accept parameters and return values.
    Classes are defined using the class keyword and support inheritance, encapsulation, and polymorphism.
    
    ## Section 5: Advanced Features
    
    Python includes many advanced features such as decorators, generators, context managers, and metaclasses.
    These features provide powerful tools for creating sophisticated and maintainable code.
    
    ## Section 6: Standard Library
    
    Python comes with a comprehensive standard library that includes modules for file I/O, networking, 
    data processing, web development, and many other common programming tasks.
    
    ## Section 7: Third-Party Packages
    
    The Python ecosystem includes thousands of third-party packages available through PyPI.
    Popular packages include NumPy for numerical computing, Django for web development, and Pandas for data analysis.
    
    ## Section 8: Best Practices
    
    Python development follows several best practices including PEP 8 style guidelines, 
    proper documentation with docstrings, and comprehensive testing with frameworks like pytest.
    
    ## Section 9: Performance Considerations
    
    While Python is not the fastest language, there are many ways to optimize performance including
    using NumPy for numerical operations, Cython for compiled extensions, and profiling tools for optimization.
    
    ## Section 10: Conclusion
    
    Python is a versatile and powerful programming language that continues to grow in popularity.
    Its simplicity, readability, and extensive ecosystem make it an excellent choice for many programming tasks.
    """ * 10  # Repeat content to make it very large
    
    # Create test document
    test_doc = Document(
        doc_id="test_large_doc",
        url="https://example.com/large-doc",
        title="Large Test Document",
        content=large_content,
        source_url="https://example.com/large-doc",
        scraped_at=datetime.now().isoformat()
    )
    
    # Test chunker
    chunker = DocumentChunker(
        chunk_size=100,
        overlap=10,
        max_tokens=15000,
        min_chunk_size=20
    )
    
    chunks = chunker.chunk_document(test_doc)
    
    print(f"Created {len(chunks)} chunks from large document")
    
    # Validate all chunks
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        token_count = TokenCounter.estimate_tokens(chunk, 'text')
        
        print(f"Chunk {i+1}: {word_count} words, ~{token_count} tokens")
        
        if token_count > 15000:
            print(f"  WARNING: Chunk {i+1} exceeds token limit!")
            return False
    
    print("✅ All chunks are within token limits!")
    return True


if __name__ == "__main__":
    # Test the chunking with large documents
    print("Testing chunking with large documents...")
    test_chunking_with_large_document()