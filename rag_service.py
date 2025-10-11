"""
Main RAG service with hybrid search (vector + keyword)
Integrates all components: embeddings, LLM, search, evaluation
"""

import time
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from elasticsearch import Elasticsearch
import numpy as np

from rag_embeddings import EmbeddingProvider
from rag_llm_abstraction import LLMProvider, LLMResponse
from rag_evaluation import FeedbackStore, QueryLog

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result"""
    doc_id: str
    chunk_id: str
    title: str
    content: str
    source_url: str
    score: float
    chunk_index: int


@dataclass
class RAGResponse:
    """Complete RAG response"""
    query_id: str
    answer: str
    retrieved_docs: List[SearchResult]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    cost_estimate: Optional[float] = None
    tokens_used: Optional[int] = None


class HybridSearchEngine:
    """Hybrid search combining vector similarity and keyword matching"""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        embedder: EmbeddingProvider,
        index_name: str = "knowledge_base"
    ):
        self.es = es_client
        self.embedder = embedder
        self.index_name = index_name
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank: bool = True
    ) -> List[SearchResult]:
        """
        Hybrid search with vector and keyword queries
        
        Args:
            query: User query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            rerank: Whether to rerank results
        """
        start = time.time()
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Build hybrid query
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Vector similarity search
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {
                                        "query_vector": query_embedding.tolist()
                                    }
                                },
                                "boost": vector_weight
                            }
                        },
                        # Keyword search (BM25)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "title^3"],
                                "type": "best_fields",
                                "boost": keyword_weight
                            }
                        }
                    ]
                }
            },
            "size": top_k * 2 if rerank else top_k,  # Get more for reranking
            "_source": ["doc_id", "chunk_id", "title", "content", "source_url", "chunk_index"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        # Parse results
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            results.append(SearchResult(
                doc_id=source['doc_id'],
                chunk_id=source['chunk_id'],
                title=source['title'],
                content=source['content'],
                source_url=source['source_url'],
                score=hit['_score'],
                chunk_index=source['chunk_index']
            ))
        
        # Optional: Rerank results
        if rerank and len(results) > top_k:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        retrieval_time = (time.time() - start) * 1000
        logger.info(f"Retrieved {len(results)} results in {retrieval_time:.0f}ms")
        
        return results
    
    def _rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Simple reranking based on query term overlap
        In production, use a cross-encoder model like Cohere Rerank or Vertex AI Ranking
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            title_terms = set(result.title.lower().split())
            
            # Compute overlap
            content_overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
            title_overlap = len(query_terms & title_terms) / len(query_terms) if query_terms else 0
            
            # Adjust score
            result.score = result.score * (1 + content_overlap * 0.3 + title_overlap * 0.5)
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def search_with_context_limit(
        self,
        query: str,
        max_context_tokens: int = 2000,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search with token limit to prevent context overflow"""
        results = self.search(query, top_k=top_k)
        
        # Estimate tokens (rough: 1 token ~= 4 chars)
        selected_results = []
        total_tokens = 0
        
        for result in results:
            result_tokens = len(result.content) // 4
            
            if total_tokens + result_tokens <= max_context_tokens:
                selected_results.append(result)
                total_tokens += result_tokens
            else:
                break
        
        logger.info(f"Selected {len(selected_results)} docs within {total_tokens} token limit")
        return selected_results


class RAGService:
    """Main RAG service orchestrating all components"""
    
    def __init__(
        self,
        search_engine: HybridSearchEngine,
        llm: LLMProvider,
        feedback_store: Optional[FeedbackStore] = None
    ):
        self.search_engine = search_engine
        self.llm = llm
        self.feedback_store = feedback_store
    
    def query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        top_k: int = 3,
        max_context_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        Process user query through RAG pipeline
        
        Args:
            user_query: User's question
            user_id: Optional user identifier for analytics
            top_k: Number of documents to retrieve
            max_context_tokens: Maximum tokens for context
            system_prompt: Optional system prompt override
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 1. Retrieval
        retrieval_start = time.time()
        retrieved_docs = self.search_engine.search_with_context_limit(
            user_query,
            max_context_tokens=max_context_tokens,
            top_k=top_k
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not retrieved_docs:
            logger.warning(f"No documents retrieved for query: {user_query}")
            return self._create_no_results_response(
                query_id, user_query, retrieval_time
            )
        
        # 2. Prepare context
        context = [
            {
                "title": doc.title,
                "content": doc.content,
                "source": doc.source_url
            }
            for doc in retrieved_docs
        ]
        
        # 3. Generate response
        generation_start = time.time()
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        llm_response = self.llm.generate_with_context(
            query=user_query,
            context=context,
            system_prompt=system_prompt
        )
        generation_time = (time.time() - generation_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # 4. Create response
        rag_response = RAGResponse(
            query_id=query_id,
            answer=llm_response.content,
            retrieved_docs=retrieved_docs,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            model_used=llm_response.model,
            cost_estimate=llm_response.cost_estimate,
            tokens_used=llm_response.tokens_used
        )
        
        # 5. Log for evaluation
        if self.feedback_store:
            self._log_query(
                query_id=query_id,
                user_query=user_query,
                user_id=user_id,
                retrieved_docs=retrieved_docs,
                llm_response=llm_response,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time
            )
        
        logger.info(
            f"Query processed in {total_time:.0f}ms "
            f"(retrieval: {retrieval_time:.0f}ms, generation: {generation_time:.0f}ms)"
        )
        
        return rag_response
    
    def _create_no_results_response(
        self,
        query_id: str,
        user_query: str,
        retrieval_time: float
    ) -> RAGResponse:
        """Create response when no documents are found"""
        fallback_answer = (
            "I couldn't find relevant information in the knowledge base to answer your question. "
            "This might be because:\n"
            "1. The topic isn't covered in our documentation\n"
            "2. The question needs to be phrased differently\n"
            "3. The information hasn't been indexed yet\n\n"
            "Please try rephrasing your question or contact support for assistance."
        )
        
        return RAGResponse(
            query_id=query_id,
            answer=fallback_answer,
            retrieved_docs=[],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=0,
            total_time_ms=retrieval_time,
            model_used="fallback"
        )
    
    def _log_query(
        self,
        query_id: str,
        user_query: str,
        user_id: Optional[str],
        retrieved_docs: List[SearchResult],
        llm_response: LLMResponse,
        retrieval_time: float,
        generation_time: float,
        total_time: float
    ):
        """Log query execution for evaluation"""
        query_log = QueryLog(
            query_id=query_id,
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            user_query=user_query,
            user_id=user_id,
            retrieved_doc_ids=[doc.doc_id for doc in retrieved_docs],
            retrieval_scores=[doc.score for doc in retrieved_docs],
            llm_response=llm_response.content,
            model_used=llm_response.model,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            cost_estimate=llm_response.cost_estimate,
            tokens_used=llm_response.tokens_used
        )
        
        self.feedback_store.log_query(query_log)
    
    @staticmethod
    def _get_default_system_prompt() -> str:
        """Default system prompt for RAG"""
        return """You are a helpful AI assistant with access to a knowledge base. 
Your role is to answer questions accurately based on the provided context.

Guidelines:
- Answer questions using ONLY information from the provided context
- Cite sources using [Document X] notation when referencing specific information
- If the context doesn't contain sufficient information, clearly state this
- Be concise but comprehensive
- If there's conflicting information in the context, acknowledge it
- Never make up information not present in the context"""


class QueryPreprocessor:
    """Preprocesses queries before search"""
    
    @staticmethod
    def preprocess(query: str) -> str:
        """
        Preprocess query:
        - Fix common typos
        - Expand acronyms
        - Rephrase for better retrieval
        """
        # Simple preprocessing for POC
        query = query.strip()
        
        # Expand common acronyms (customize for your domain)
        acronyms = {
            "API": "Application Programming Interface",
            "REST": "RESTful",
            "DB": "database",
            "ML": "machine learning",
            "AI": "artificial intelligence"
        }
        
        for acronym, expansion in acronyms.items():
            if acronym in query.upper():
                query = query + f" {expansion}"
        
        return query


# Example usage
if __name__ == "__main__":
    from rag_embeddings import EmbeddingFactory, EMBEDDING_CONFIGS
    from rag_llm_abstraction import LLMFactory, LLM_CONFIGS
    from rag_evaluation import FeedbackStore, FeedbackType
    
    # Setup components
    es_client = Elasticsearch(['http://localhost:9200'])
    
    # Use local models for testing
    embedder = EmbeddingFactory.create(EMBEDDING_CONFIGS['local_minilm'])
    llm = LLMFactory.create(LLM_CONFIGS['local_llama'])
    
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
    
    # Test query
    response = rag_service.query(
        user_query="What is FastAPI and how do I install it?",
        user_id="test_user_123",
        top_k=3
    )
    
    print("=" * 60)
    print(f"Query ID: {response.query_id}")
    print(f"Answer:\n{response.answer}\n")
    print(f"Retrieved {len(response.retrieved_docs)} documents:")
    for i, doc in enumerate(response.retrieved_docs, 1):
        print(f"  {i}. {doc.title[:50]}... (score: {doc.score:.3f})")
    print(f"\nPerformance:")
    print(f"  Total time: {response.total_time_ms:.0f}ms")
    print(f"  Retrieval: {response.retrieval_time_ms:.0f}ms")
    print(f"  Generation: {response.generation_time_ms:.0f}ms")
    print(f"  Cost: ${response.cost_estimate:.6f}" if response.cost_estimate else "  Cost: $0.00 (local)")
    print("=" * 60)
    
    # Simulate user feedback
    feedback_store.add_feedback(
        query_id=response.query_id,
        feedback_type=FeedbackType.THUMBS_UP
    )
    
    # Get metrics
    metrics = feedback_store.compute_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Total queries: {metrics.total_queries}")
    print(f"  Thumbs up rate: {metrics.thumbs_up_rate:.1%}")
    print(f"  Avg response time: {metrics.avg_total_time_ms:.0f}ms")