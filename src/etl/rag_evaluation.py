"""
Evaluation and feedback system for RAG pipeline
Tracks user feedback, metrics, and provides analytics
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from elasticsearch import Elasticsearch
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 stars


@dataclass
class QueryLog:
    """Complete log of a query execution"""
    query_id: str
    timestamp: str
    
    # Input
    user_query: str
    user_id: Optional[str] = None
    
    # Processing
    query_embedding: Optional[List[float]] = None
    retrieved_doc_ids: List[str] = None
    retrieved_docs: List[Dict] = None
    retrieval_scores: List[float] = None
    
    # Generation
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    model_used: Optional[str] = None
    
    # Performance
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    
    # Cost
    cost_estimate: Optional[float] = None
    tokens_used: Optional[int] = None
    
    # Feedback (added later)
    feedback_type: Optional[str] = None
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None
    feedback_timestamp: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics"""
    period_start: str
    period_end: str
    
    # Volume
    total_queries: int
    unique_users: int
    
    # Feedback
    thumbs_up_count: int
    thumbs_down_count: int
    thumbs_up_rate: float
    avg_rating: Optional[float] = None
    
    # Performance
    avg_total_time_ms: Optional[float] = None
    p50_time_ms: Optional[float] = None
    p95_time_ms: Optional[float] = None
    p99_time_ms: Optional[float] = None
    
    # Cost
    total_cost: Optional[float] = None
    avg_cost_per_query: Optional[float] = None
    
    # Popular queries
    top_queries: Optional[List[Dict]] = None
    low_rated_queries: Optional[List[Dict]] = None


class FeedbackStore:
    """Stores and retrieves feedback in Elasticsearch"""
    
    def __init__(self, es_client: Elasticsearch, index_name: str = "query_logs"):
        self.es = es_client
        self.index_name = index_name
        self._create_index()
    
    def _create_index(self):
        """Create index for query logs"""
        mapping = {
            "mappings": {
                "properties": {
                    "query_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "user_query": {"type": "text"},
                    "user_id": {"type": "keyword"},
                    "retrieved_doc_ids": {"type": "keyword"},
                    "retrieval_scores": {"type": "float"},
                    "llm_response": {"type": "text"},
                    "model_used": {"type": "keyword"},
                    "retrieval_time_ms": {"type": "float"},
                    "generation_time_ms": {"type": "float"},
                    "total_time_ms": {"type": "float"},
                    "cost_estimate": {"type": "float"},
                    "tokens_used": {"type": "integer"},
                    "feedback_type": {"type": "keyword"},
                    "feedback_rating": {"type": "integer"},
                    "feedback_comment": {"type": "text"},
                    "feedback_timestamp": {"type": "date"}
                }
            }
        }
        
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created feedback index: {self.index_name}")
    
    def log_query(self, query_log: QueryLog):
        """Log a query execution"""
        doc = {k: v for k, v in asdict(query_log).items() if v is not None}
        self.es.index(index=self.index_name, id=query_log.query_id, body=doc)
    
    def add_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """Add user feedback to a query"""
        feedback_data = {
            "feedback_type": feedback_type.value,
            "feedback_timestamp": datetime.now().isoformat()
        }
        
        if rating is not None:
            feedback_data["feedback_rating"] = rating
        if comment:
            feedback_data["feedback_comment"] = comment
        
        self.es.update(
            index=self.index_name,
            id=query_id,
            body={"doc": feedback_data}
        )
        logger.info(f"Added {feedback_type.value} feedback to query {query_id}")
    
    def get_query_log(self, query_id: str) -> Optional[QueryLog]:
        """Retrieve a query log"""
        try:
            result = self.es.get(index=self.index_name, id=query_id)
            return QueryLog(**result['_source'])
        except:
            return None
    
    def compute_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> EvaluationMetrics:
        """Compute aggregate metrics for a time period"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=7)
        
        # Build query
        query = {
            "query": {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lte": end_time.isoformat()
                    }
                }
            },
            "size": 0,
            "aggs": {
                "total_queries": {"value_count": {"field": "query_id"}},
                "unique_users": {"cardinality": {"field": "user_id"}},
                "thumbs_up": {
                    "filter": {"term": {"feedback_type": "thumbs_up"}}
                },
                "thumbs_down": {
                    "filter": {"term": {"feedback_type": "thumbs_down"}}
                },
                "avg_rating": {"avg": {"field": "feedback_rating"}},
                "avg_time": {"avg": {"field": "total_time_ms"}},
                "time_percentiles": {
                    "percentiles": {
                        "field": "total_time_ms",
                        "percents": [50, 95, 99]
                    }
                },
                "total_cost": {"sum": {"field": "cost_estimate"}},
                "avg_cost": {"avg": {"field": "cost_estimate"}}
            }
        }
        
        result = self.es.search(index=self.index_name, body=query)
        aggs = result['aggregations']
        
        total_queries = aggs['total_queries']['value']
        thumbs_up = aggs['thumbs_up']['doc_count']
        thumbs_down = aggs['thumbs_down']['doc_count']
        
        thumbs_up_rate = thumbs_up / (thumbs_up + thumbs_down) if (thumbs_up + thumbs_down) > 0 else 0
        
        time_pcts = aggs['time_percentiles']['values']
        
        return EvaluationMetrics(
            period_start=start_time.isoformat(),
            period_end=end_time.isoformat(),
            total_queries=int(total_queries),
            unique_users=aggs['unique_users']['value'],
            thumbs_up_count=thumbs_up,
            thumbs_down_count=thumbs_down,
            thumbs_up_rate=thumbs_up_rate,
            avg_rating=aggs['avg_rating'].get('value'),
            avg_total_time_ms=aggs['avg_time'].get('value', 0),
            p50_time_ms=time_pcts.get('50.0', 0),
            p95_time_ms=time_pcts.get('95.0', 0),
            p99_time_ms=time_pcts.get('99.0', 0),
            total_cost=aggs['total_cost'].get('value', 0),
            avg_cost_per_query=aggs['avg_cost'].get('value', 0)
        )
    
    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most common queries"""
        query = {
            "size": 0,
            "aggs": {
                "top_queries": {
                    "terms": {
                        "field": "user_query.keyword",
                        "size": limit
                    }
                }
            }
        }
        
        result = self.es.search(index=self.index_name, body=query)
        buckets = result['aggregations']['top_queries']['buckets']
        
        return [
            {"query": b['key'], "count": b['doc_count']}
            for b in buckets
        ]
    
    def get_low_rated_queries(self, limit: int = 10) -> List[Dict]:
        """Get queries with thumbs down or low ratings"""
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"feedback_type": "thumbs_down"}},
                        {"range": {"feedback_rating": {"lte": 2}}}
                    ]
                }
            },
            "size": limit,
            "sort": [{"timestamp": "desc"}],
            "_source": ["query_id", "user_query", "feedback_type", "feedback_rating", "feedback_comment"]
        }
        
        result = self.es.search(index=self.index_name, body=query)
        
        return [
            {
                "query_id": hit['_source']['query_id'],
                "query": hit['_source']['user_query'],
                "feedback_type": hit['_source'].get('feedback_type'),
                "rating": hit['_source'].get('feedback_rating'),
                "comment": hit['_source'].get('feedback_comment')
            }
            for hit in result['hits']['hits']
        ]


class RAGEvaluator:
    """Evaluates RAG system quality"""
    
    @staticmethod
    def evaluate_retrieval(
        query: str,
        retrieved_docs: List[Dict],
        ground_truth_docs: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate retrieval quality (requires ground truth)"""
        if ground_truth_docs is None:
            return {}
        
        retrieved_ids = {doc['doc_id'] for doc in retrieved_docs}
        ground_truth_set = set(ground_truth_docs)
        
        # Precision@k
        relevant_retrieved = retrieved_ids & ground_truth_set
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        
        # Recall@k
        recall = len(relevant_retrieved) / len(ground_truth_set) if ground_truth_set else 0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc['doc_id'] in ground_truth_set:
                mrr = 1.0 / i
                break
        
        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_score": f1,
            "mrr": mrr
        }
    
    @staticmethod
    def evaluate_answer_relevance(
        query: str,
        answer: str,
        min_length: int = 50
    ) -> Dict[str, bool]:
        """Basic answer quality checks"""
        return {
            "has_minimum_length": len(answer) >= min_length,
            "not_empty": bool(answer.strip()),
            "has_citations": "[Document" in answer or "[Source" in answer
        }


class TestSuite:
    """Test suite for regression testing"""
    
    def __init__(self, feedback_store: FeedbackStore):
        self.feedback_store = feedback_store
        self.test_cases: List[Dict] = []
    
    def add_test_case(
        self,
        query: str,
        expected_doc_ids: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
        min_score: float = 0.7
    ):
        """Add a test case"""
        self.test_cases.append({
            "query": query,
            "expected_doc_ids": expected_doc_ids,
            "expected_keywords": expected_keywords,
            "min_score": min_score
        })
    
    def run_tests(self, rag_system) -> Dict:
        """Run all test cases (requires RAG system instance)"""
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test_case in self.test_cases:
            # Run query through RAG system
            response = rag_system.query(test_case["query"])
            
            # Check expected docs
            passed = True
            reasons = []
            
            if test_case.get("expected_doc_ids"):
                retrieved_ids = {doc['doc_id'] for doc in response['retrieved_docs']}
                expected_ids = set(test_case["expected_doc_ids"])
                overlap = len(retrieved_ids & expected_ids) / len(expected_ids)
                
                if overlap < test_case["min_score"]:
                    passed = False
                    reasons.append(f"Doc overlap {overlap:.2f} < {test_case['min_score']}")
            
            if test_case.get("expected_keywords"):
                answer_lower = response['answer'].lower()
                missing_keywords = [
                    kw for kw in test_case["expected_keywords"]
                    if kw.lower() not in answer_lower
                ]
                if missing_keywords:
                    passed = False
                    reasons.append(f"Missing keywords: {missing_keywords}")
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "query": test_case["query"],
                "passed": passed,
                "reasons": reasons
            })
        
        return results
    
    def export_failing_queries_to_tests(self, days: int = 7):
        """Convert recent low-rated queries into test cases"""
        low_rated = self.feedback_store.get_low_rated_queries(limit=20)
        
        for query_info in low_rated:
            logger.info(f"Review this failed query: {query_info['query']}")
            logger.info(f"  Comment: {query_info.get('comment', 'None')}")
            # Manual review needed to create proper test case


# Example usage
if __name__ == "__main__":
    es_client = Elasticsearch(['http://localhost:9200'])
    feedback_store = FeedbackStore(es_client)
    
    # Simulate logging a query
    query_log = QueryLog(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        user_query="What is FastAPI?",
        retrieved_doc_ids=["doc1", "doc2"],
        llm_response="FastAPI is a modern web framework...",
        model_used="gemini-1.5-flash",
        total_time_ms=850,
        cost_estimate=0.0023
    )
    
    feedback_store.log_query(query_log)
    
    # Add feedback
    feedback_store.add_feedback(
        query_id=query_log.query_id,
        feedback_type=FeedbackType.THUMBS_UP
    )
    
    # Get metrics
    metrics = feedback_store.compute_metrics()
    print(f"Thumbs up rate: {metrics.thumbs_up_rate:.1%}")
    print(f"Avg response time: {metrics.avg_total_time_ms:.0f}ms")
    print(f"P95 latency: {metrics.p95_time_ms:.0f}ms")