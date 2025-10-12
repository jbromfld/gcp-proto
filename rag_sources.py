"""
Knowledge Source Management
Tracks and manages knowledge sources (URLs, scraping patterns, ingestion history)
"""
from elasticsearch import Elasticsearch
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Types of knowledge sources"""
    WEB_PAGE = "web_page"
    WEB_RECURSIVE = "web_recursive"
    DOCUMENTATION = "documentation"


class SourceStatus(str, Enum):
    """Status of source ingestion"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"


@dataclass
class KnowledgeSource:
    """Represents a knowledge source"""
    source_id: str
    url: str
    source_type: SourceType
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Scraping configuration
    max_depth: int = 1  # For recursive scraping
    max_pages: int = 50
    include_patterns: Optional[List[str]] = None  # e.g., ["*/docs/*"]
    exclude_patterns: Optional[List[str]] = None  # e.g., ["*/blog/*"]
    
    # Status tracking
    status: SourceStatus = SourceStatus.PENDING
    last_ingested: Optional[str] = None
    next_scheduled: Optional[str] = None
    
    # Statistics
    pages_scraped: int = 0
    chunks_created: int = 0
    error_message: Optional[str] = None
    
    # Metadata
    created_at: str = None
    updated_at: str = None
    created_by: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


class SourceManager:
    """Manages knowledge sources in Elasticsearch"""
    
    def __init__(self, es_client: Elasticsearch, index_name: str = "knowledge_sources"):
        self.es = es_client
        self.index_name = index_name
        self._create_index()
    
    def _create_index(self):
        """Create index for knowledge sources"""
        mapping = {
            "mappings": {
                "properties": {
                    "source_id": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "max_depth": {"type": "integer"},
                    "max_pages": {"type": "integer"},
                    "include_patterns": {"type": "keyword"},
                    "exclude_patterns": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "last_ingested": {"type": "date"},
                    "next_scheduled": {"type": "date"},
                    "pages_scraped": {"type": "integer"},
                    "chunks_created": {"type": "integer"},
                    "error_message": {"type": "text"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "created_by": {"type": "keyword"}
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
    
    def add_source(self, source: KnowledgeSource) -> Dict:
        """Add a new knowledge source"""
        try:
            result = self.es.index(
                index=self.index_name,
                id=source.source_id,
                document=asdict(source)
            )
            logger.info(f"Added source: {source.source_id}")
            return {"success": True, "source_id": source.source_id}
        except Exception as e:
            logger.error(f"Failed to add source: {e}")
            return {"success": False, "error": str(e)}
    
    def update_source(self, source_id: str, updates: Dict) -> Dict:
        """Update an existing source"""
        try:
            updates['updated_at'] = datetime.utcnow().isoformat()
            result = self.es.update(
                index=self.index_name,
                id=source_id,
                doc=updates
            )
            logger.info(f"Updated source: {source_id}")
            return {"success": True, "source_id": source_id}
        except Exception as e:
            logger.error(f"Failed to update source: {e}")
            return {"success": False, "error": str(e)}
    
    def get_source(self, source_id: str) -> Optional[KnowledgeSource]:
        """Get a source by ID"""
        try:
            result = self.es.get(index=self.index_name, id=source_id)
            return KnowledgeSource(**result['_source'])
        except Exception as e:
            logger.error(f"Failed to get source {source_id}: {e}")
            return None
    
    def list_sources(self, status: Optional[SourceStatus] = None, limit: int = 100) -> List[KnowledgeSource]:
        """List all sources, optionally filtered by status"""
        try:
            query = {"match_all": {}}
            if status:
                query = {"term": {"status": status.value}}
            
            result = self.es.search(
                index=self.index_name,
                query=query,
                size=limit,
                sort=[{"created_at": {"order": "desc"}}]
            )
            
            sources = []
            for hit in result['hits']['hits']:
                sources.append(KnowledgeSource(**hit['_source']))
            
            return sources
        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []
    
    def delete_source(self, source_id: str) -> Dict:
        """Delete a source"""
        try:
            self.es.delete(index=self.index_name, id=source_id)
            logger.info(f"Deleted source: {source_id}")
            return {"success": True, "source_id": source_id}
        except Exception as e:
            logger.error(f"Failed to delete source: {e}")
            return {"success": False, "error": str(e)}
    
    def update_ingestion_stats(self, source_id: str, pages_scraped: int, chunks_created: int, 
                               status: SourceStatus, error_message: Optional[str] = None) -> Dict:
        """Update ingestion statistics for a source"""
        updates = {
            "pages_scraped": pages_scraped,
            "chunks_created": chunks_created,
            "status": status.value,
            "last_ingested": datetime.utcnow().isoformat()
        }
        
        if error_message:
            updates["error_message"] = error_message
        
        return self.update_source(source_id, updates)
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        try:
            result = self.es.search(
                index=self.index_name,
                size=0,
                aggs={
                    "by_status": {
                        "terms": {"field": "status"}
                    },
                    "total_pages": {
                        "sum": {"field": "pages_scraped"}
                    },
                    "total_chunks": {
                        "sum": {"field": "chunks_created"}
                    }
                }
            )
            
            return {
                "total_sources": result['hits']['total']['value'],
                "by_status": {
                    bucket['key']: bucket['doc_count']
                    for bucket in result['aggregations']['by_status']['buckets']
                },
                "total_pages_scraped": int(result['aggregations']['total_pages']['value']),
                "total_chunks_created": int(result['aggregations']['total_chunks']['value'])
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

