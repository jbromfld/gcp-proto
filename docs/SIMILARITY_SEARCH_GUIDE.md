# Vector Similarity Search: A Comprehensive Guide

## Table of Contents
1. [Understanding Relevance Scores](#understanding-relevance-scores)
2. [Components That Affect Search Quality](#components-that-affect-search-quality)
3. [Testing & Evaluation Methodology](#testing--evaluation-methodology)
4. [Optimization Strategies](#optimization-strategies)
5. [Local Testing Workflows](#local-testing-workflows)
6. [Real-World Examples](#real-world-examples)

---

## Understanding Relevance Scores

### What is a Relevance Score?

A **relevance score** measures how similar a query is to a document in vector space. In our system:

```python
# Simplified version of what happens:
query_vector = embed("How do I install FastAPI?")      # [0.23, -0.45, 0.67, ...]
doc_vector = embed("Install FastAPI with pip install") # [0.25, -0.43, 0.65, ...]

similarity = cosine_similarity(query_vector, doc_vector)  # Returns 0.0 to 1.0
```

### Hybrid Search Scoring

Our system combines **vector search** + **keyword search**:

```python
# Vector similarity (semantic meaning)
vector_score = cosine_similarity(query_embedding, doc_embedding)  # 0.0 - 1.0

# Keyword match (exact words via BM25)
keyword_score = bm25_score(query_text, doc_text)  # varies, typically 0-20

# Combined score (Elasticsearch RRF - Reciprocal Rank Fusion)
final_score = combine_scores(vector_score, keyword_score)  # typically 1-15
```

**Typical score ranges in our system:**
- **1-3**: Weak match (maybe related topic)
- **3-6**: Moderate relevance (on topic, some keywords match)
- **6-10**: Strong relevance (direct answer, high semantic + keyword match)
- **10+**: Excellent match (exact or near-exact answer)

### Our Threshold Strategy

```python
RELEVANCE_THRESHOLD = 3.0  # Current setting

if max_score < 3.0:
    # Documents found but not confident they answer the question
    # Fall back to LLM general knowledge
    return llm_general_knowledge_answer(query)
else:
    # Documents are relevant enough to use as context
    return llm_answer_with_rag_context(query, documents)
```

**Why not always use retrieved docs?**
- If user asks "What is Java?" but only Python docs exist, score might be 2.5
- Without threshold: LLM sees irrelevant Python docs, gives confused answer
- With threshold: LLM uses general knowledge, gives correct Java answer

---

## Components That Affect Search Quality

### 1. Embedding Model

The embedding model converts text → vectors. Different models have different strengths:

#### Model Comparison

| Model | Dimensions | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | General docs, fast prototypes |
| **all-mpnet-base-v2** | 768 | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | General purpose, balanced |
| **text-embedding-ada-002** (OpenAI) | 1536 | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | Production, high quality needed |
| **text-embedding-004** (Vertex AI) | 768 | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Best | GCP deployments |
| **CodeBERT** | 768 | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | Source code, API docs |
| **e5-large-v2** | 1024 | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Multi-domain, research |

#### Code vs Documentation Models

**For Documentation:**
- Use general-purpose models (MiniLM, MPNet, Ada)
- Optimized for natural language
- Good at understanding questions

**For Source Code:**
- Use code-specific models:
  - **CodeBERT**: Trained on GitHub code
  - **GraphCodeBERT**: Understands code structure
  - **StarCoder embeddings**: Modern, multilingual
- Better at:
  - Function similarity
  - API usage patterns
  - Code-to-comment mapping

### 2. Vector Dimensions

More dimensions ≠ always better!

| Dimensions | Storage/RAM | Search Speed | Quality | Trade-off |
|-----------|-------------|--------------|---------|-----------|
| 384 | 1x baseline | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | Great for <100k docs |
| 768 | 2x baseline | ⚡⚡ Fast | ⭐⭐⭐⭐ Better | Sweet spot for most use cases |
| 1536 | 4x baseline | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | When quality > cost |

**Example storage:**
```
100,000 chunks × 768 dims × 4 bytes = ~300 MB (vectors only)
100,000 chunks × 1536 dims × 4 bytes = ~600 MB (vectors only)
```

**Diminishing returns:** 
- 384 → 768 dims: ~20% quality improvement
- 768 → 1536 dims: ~5-10% quality improvement
- Cost doubles each time!

### 3. Indexing Embeddings: How Vectors Are Stored & Searched

#### The Problem: Searching Millions of Vectors

**Naive approach (Exhaustive/Brute Force Search):**
```python
# Compare query against EVERY document
def brute_force_search(query_vector, all_doc_vectors):
    scores = []
    for doc_vector in all_doc_vectors:  # O(n)
        score = cosine_similarity(query_vector, doc_vector)
        scores.append(score)
    return top_k(scores, k=3)

# For 1 million vectors:
# - 1M comparisons per query
# - ~100-500ms latency
# - 100% accuracy (finds exact top-K)
```

**This doesn't scale!** For production systems with millions of vectors, we need approximate nearest neighbor (ANN) search.

#### Indexing Strategies

##### **1. HNSW (Hierarchical Navigable Small World)** - Current Choice ✅

**How it works:**
```
Imagine a multi-level highway system:

Level 3 (Highway):     A ←--------→ Z
                       ↓            ↓
Level 2 (Main Roads):  A → E → M → S → Z
                       ↓   ↓   ↓   ↓   ↓
Level 1 (Streets):     A→B→C→D→E→F→...→Z
                       ↓ ↓ ↓ ↓ ↓ ↓
Level 0 (All Nodes):   Every single document

Search process:
1. Start at top level (highway), jump to approximate region
2. Move down levels, refining search
3. At bottom level, find exact nearest neighbors
4. Total comparisons: ~log(n) instead of n
```

**Configuration:**
```python
"index_options": {
    "type": "hnsw",
    "m": 16,              # Connections per node at each level
    "ef_construction": 100 # How thoroughly to build index
}

# m (links per node):
# - m=4:  Fewer connections, faster build, less accurate search
# - m=16: Balanced (current) - good speed + accuracy ✅
# - m=32: More connections, slower build, very accurate
# - m=64: Diminishing returns, mostly for research

# ef_construction (build quality):
# - ef=50:  Fast indexing (~1000 docs/sec), ok search quality
# - ef=100: Balanced (current) - good quality ✅  
# - ef=200: Slow indexing (~300 docs/sec), excellent quality
# - ef=500: Very slow, marginal improvement
```

**Performance:**
```
Dataset: 1M vectors, 768 dimensions

m=8,  ef=50:   Build 15 min, Query  8ms,  Recall@10: 0.92
m=16, ef=100:  Build 25 min, Query 12ms,  Recall@10: 0.96 ✅
m=32, ef=200:  Build 60 min, Query 18ms,  Recall@10: 0.98
Brute force:   Build  0 min, Query 350ms, Recall@10: 1.00

Recall = what % of true top-10 results we actually found
```

**When to use:**
- ✅ Most production RAG systems (best speed/accuracy trade-off)
- ✅ When you need sub-50ms queries
- ✅ Static or slowly-changing datasets
- ❌ Extremely frequent updates (rebuild is expensive)

##### **2. IVFFlat (Inverted File with Flat Compression)**

**How it works:**
```
1. Cluster vectors into N groups (centroids)
2. At query time:
   - Find nearest centroids (fast)
   - Search only within those clusters (not all docs)
   - Return top-K

Example with 1M vectors, 100 clusters:
- Compare query to 100 centroids (fast)
- Search within 10 nearest clusters = 100k vectors
- Skip 900k vectors entirely!
```

**Configuration:**
```python
"index_options": {
    "type": "ivfflat",
    "n_lists": 100  # Number of clusters
}

# n_lists (cluster count):
# For N documents:
# - n_lists = sqrt(N):  Balanced (1M docs → 1000 clusters)
# - n_lists = N/1000:   Faster, less accurate
# - n_lists = N/100:    Slower, more accurate
```

**Performance:**
```
Dataset: 1M vectors, 768 dimensions, 1000 clusters

n_lists=100:   Build 10 min, Query 50ms,  Recall@10: 0.85
n_lists=1000:  Build 15 min, Query 120ms, Recall@10: 0.92
n_lists=5000:  Build 25 min, Query 280ms, Recall@10: 0.96
```

**When to use:**
- ✅ Frequently updated datasets (faster to rebuild than HNSW)
- ✅ When 50-200ms latency is acceptable
- ✅ Cost-sensitive (lower memory than HNSW)
- ❌ When you need <20ms queries

##### **3. Product Quantization (PQ)**

Compresses vectors to reduce memory:

```python
# Original vector: 768 floats × 4 bytes = 3KB
original_vector = [0.234, -0.456, 0.678, ...]  # 768 numbers

# Quantized: 768 floats → 96 bytes (96% smaller!)
quantized_vector = compress(original_vector, method="pq", m=8, nbits=8)

# Trade-off:
# - 32x less memory
# - Slightly lower accuracy (~95% of brute force)
# - Still fast search
```

**When to use:**
- ✅ Huge datasets (100M+ vectors)
- ✅ Limited RAM
- ✅ When slight accuracy loss is acceptable
- ❌ Small datasets (overhead not worth it)

##### **4. No Index (Brute Force)**

Sometimes simple is best:

**When to use brute force:**
```
Dataset size < 10,000 vectors:
  - Index overhead > search savings
  - Just do exhaustive search
  - 100% accuracy, <50ms queries

Dataset size > 10,000 vectors:
  - Use HNSW or IVFFlat
  - Approximate is faster
```

#### Index Comparison Table

| Method | Build Time | Query Speed | Accuracy | Memory | Best For |
|--------|-----------|-------------|----------|---------|----------|
| **None (Brute)** | 0s | 🐌🐌🐌 Slow | 100% | 1x | <10k docs |
| **IVFFlat** | ⚡⚡ Fast | 🏃 Medium | 92-95% | 1x | Frequently updated |
| **HNSW** | 🐌 Slow | ⚡⚡⚡ Fast | 95-98% | 1.5x | Production RAG ✅ |
| **PQ** | ⚡ Fast | ⚡⚡ Fast | 90-95% | 0.1x | Massive scale |
| **HNSW+PQ** | 🐌 Slow | ⚡⚡⚡ Fast | 93-96% | 0.2x | Best of both |

#### Real-World Indexing Examples

**Example 1: Small knowledge base (<10k docs)**
```yaml
Index: None (brute force)
Why: Index overhead not worth it
Query time: 30-50ms
Accuracy: 100%
```

**Example 2: Medium knowledge base (10k-1M docs)** - Current
```yaml
Index: HNSW
Config:
  m: 16
  ef_construction: 100
Query time: 10-20ms
Accuracy: 96%
Build time: 5 min for 100k docs
```

**Example 3: Large knowledge base (1M-10M docs)**
```yaml
Index: HNSW + Product Quantization
Config:
  m: 32
  ef_construction: 200
  pq_m: 8
Query time: 15-30ms
Accuracy: 94%
Memory: 80% reduction
Build time: 2 hours for 5M docs
```

**Example 4: Massive scale (10M+ docs)**
```yaml
Index: Distributed HNSW (across multiple nodes)
Or: Specialized vector DB (Pinecone, Weaviate, Qdrant)
Query time: 20-50ms
Accuracy: 95%
Cost: $$$
```

#### Understanding Index Build Time

```python
# How long does indexing take?

def estimate_build_time(num_docs, dims, index_type):
    if index_type == "hnsw":
        # Rough estimate: O(n * log(n) * m * ef_construction)
        base_time_per_doc = 0.01  # seconds (m=16, ef=100)
        return num_docs * base_time_per_doc * (dims / 384)
    
    elif index_type == "ivfflat":
        # Rough estimate: O(n * d) where d = dims
        base_time_per_doc = 0.001  # seconds
        return num_docs * base_time_per_doc * (dims / 384)

# Examples:
print(estimate_build_time(10_000, 384, "hnsw"))    # ~2 minutes
print(estimate_build_time(100_000, 768, "hnsw"))   # ~26 minutes
print(estimate_build_time(1_000_000, 768, "hnsw")) # ~4 hours

# For large datasets, consider:
# 1. Batch indexing (index in chunks)
# 2. Parallel indexing (multiple workers)
# 3. Pre-built indexes (index once, deploy everywhere)
```

#### Query-Time Parameters

Even with a built index, you can tune search behavior:

**HNSW query parameters:**
```python
# ef_search: How many candidates to explore at query time
# Higher = more accurate but slower

search(query, ef_search=10)   # Fast,  ~90% recall
search(query, ef_search=50)   # Medium, ~95% recall ✅ (default)
search(query, ef_search=100)  # Slow,  ~98% recall
search(query, ef_search=500)  # Very slow, ~99.5% recall

# Rule of thumb:
# ef_search >= k (top-k results you want)
# ef_search = 50-100 is sweet spot for most use cases
```

**IVFFlat query parameters:**
```python
# nprobe: How many clusters to search
# Higher = more accurate but slower

search(query, nprobe=1)   # Search 1 cluster,  very fast, ~70% recall
search(query, nprobe=10)  # Search 10 clusters, fast,     ~90% recall ✅
search(query, nprobe=50)  # Search 50 clusters, medium,   ~95% recall
search(query, nprobe=100) # Search 100 clusters, slow,    ~98% recall

# Trade-off:
# nprobe=1: Query 0.1% of data
# nprobe=10: Query 1% of data (100x more than nprobe=1)
# nprobe=100: Query 10% of data
```

#### Monitoring Index Health

**Check index statistics:**
```bash
# Elasticsearch
curl http://localhost:9200/knowledge_base/_stats | jq '{
  docs: .indices.knowledge_base.total.docs.count,
  size: .indices.knowledge_base.total.store.size,
  segments: .indices.knowledge_base.total.segments.count
}'

# Output:
# {
#   "docs": 30000,
#   "size": "450MB",
#   "segments": 12
# }
```

**Index fragmentation:**
```python
# Too many segments = slow search
# Elasticsearch automatically merges, but you can force:

# Force merge to 1 segment (do during off-peak hours)
POST /knowledge_base/_forcemerge?max_num_segments=1

# This can take hours for large indexes!
# But queries will be faster afterward
```

#### Incremental Indexing

When adding new documents to an existing index:

```python
# Option 1: Append (fast)
# - Just add new documents
# - HNSW builds connections on-the-fly
# - Slightly degrades search quality over time
es.index(index="knowledge_base", document=new_doc)

# Option 2: Rebuild periodically (better quality)
# - Every N documents, rebuild entire index
# - Maintains optimal graph structure
# - Requires downtime or dual indexes

if new_docs_count > 10000:
    rebuild_index()
```

**Dual index pattern (zero downtime):**
```python
# 1. Build new index
create_index("knowledge_base_v2")
index_all_documents("knowledge_base_v2")

# 2. Alias swap (atomic)
remove_alias("knowledge_base_current", "knowledge_base_v1")
add_alias("knowledge_base_current", "knowledge_base_v2")

# 3. Delete old index
delete_index("knowledge_base_v1")

# Your app always uses alias "knowledge_base_current"
# Users experience zero downtime!
```

#### Index Memory Requirements

**Elasticsearch HNSW:**
```python
def estimate_memory(num_docs, dims, m=16):
    # Each vector: dims × 4 bytes (float32)
    vector_storage = num_docs * dims * 4
    
    # HNSW graph: ~m × 2 × 8 bytes per document
    # (m connections, 2 layers average, 8 bytes per connection)
    graph_storage = num_docs * m * 2 * 8
    
    # Elasticsearch overhead: ~20%
    overhead = (vector_storage + graph_storage) * 0.2
    
    total_mb = (vector_storage + graph_storage + overhead) / (1024 * 1024)
    return total_mb

# Examples:
print(f"10k docs, 384 dims:   {estimate_memory(10_000, 384):.0f} MB")    # ~30 MB
print(f"100k docs, 768 dims:  {estimate_memory(100_000, 768):.0f} MB")   # ~600 MB
print(f"1M docs, 768 dims:    {estimate_memory(1_000_000, 768):.0f} MB") # ~6 GB
print(f"10M docs, 1536 dims:  {estimate_memory(10_000_000, 1536):.0f} MB") # ~125 GB

# This is RAM required for optimal search performance
# Elasticsearch can work with less but will use disk (slower)
```

**Our current system:**
```
30 chunks × 384 dims (local):
  - Vector storage: ~45 KB
  - HNSW graph: ~8 KB
  - Total: ~60 KB (tiny!)

100k chunks × 768 dims (production):
  - Vector storage: ~300 MB
  - HNSW graph: ~50 MB
  - Total: ~420 MB
  - ES recommendation: 4GB RAM (current: e2-medium ✅)
```

#### Index Update Strategies

**Strategy 1: Real-time Updates** (Simple)
```python
# As documents come in, add them immediately
def add_document(doc):
    chunk_and_embed(doc)
    es.index(index="knowledge_base", document=doc)

# Pros: Always up-to-date
# Cons: Index quality degrades, needs periodic rebuild
```

**Strategy 2: Batch Updates** (Efficient)
```python
# Accumulate documents, then bulk index
batch = []
for doc in new_documents:
    batch.append(doc)
    if len(batch) >= 1000:
        es.bulk(index="knowledge_base", operations=batch)
        batch = []

# Pros: 10-100x faster than individual inserts
# Cons: Slight delay before docs are searchable
```

**Strategy 3: Scheduled Rebuilds** (Current) ✅
```python
# Rebuild entire index daily/weekly
class ScheduledETL:
    def run_daily(self):
        # 1. Scrape all sources
        # 2. Chunk + embed
        # 3. Bulk index
        # 4. Force merge (optimize)

# Pros: Maintains optimal index quality
# Cons: Resource intensive, not real-time
# Works well when: Content changes slowly (docs, not chat)
```

**Strategy 4: Hybrid** (Best for Production)
```python
# Real-time updates + periodic optimization
class SmartIndexer:
    def add_document(self, doc):
        # Add immediately for searchability
        es.index(index="knowledge_base", document=doc)
        self.updates_since_rebuild += 1
        
        # Rebuild if too many updates
        if self.updates_since_rebuild > 10000:
            self.schedule_rebuild()
    
    def schedule_rebuild(self):
        # During off-peak hours
        if is_low_traffic_period():
            rebuild_index_optimized()
```

#### Monitoring Index Performance

**Key metrics to track:**

```python
# 1. Query latency
p50_latency = 15ms  # Median
p95_latency = 45ms  # 95th percentile  
p99_latency = 120ms # 99th percentile

# If p95 > 100ms, consider:
# - Reducing m or ef_construction
# - Adding more RAM
# - Reducing vector dimensions
# - Sharding across multiple nodes

# 2. Recall (accuracy)
# Compare to brute force on sample queries
recall_at_10 = 0.96  # Found 96% of true top-10

# If recall < 0.90, consider:
# - Increasing m or ef_construction
# - Using better embedding model
# - Switching from IVFFlat to HNSW

# 3. Index fragmentation
segments_count = 45  # Elasticsearch segments

# If segments > 30, force merge:
POST /knowledge_base/_forcemerge?max_num_segments=5

# 4. Memory pressure
heap_usage = 85%  # Elasticsearch JVM heap

# If > 85%, consider:
# - Adding more RAM
# - Reducing m parameter
# - Enabling compression
# - Sharding to multiple nodes
```

#### Advanced: Composite Indexes

For multi-field search with filters:

```python
# Scenario: Search docs + filter by date + filter by source

# Naive: Index only vectors, filter in application
results = vector_search(query, top_k=100)  # Get 100
filtered = [r for r in results if r.date > "2024-01-01"][:10]  # Filter to 10

# Problem: Wasted 90 searches, might miss relevant docs

# Better: Filtered vector search
results = vector_search(
    query,
    top_k=10,
    filter={
        "range": {"timestamp": {"gte": "2024-01-01"}},
        "term": {"source_type": "documentation"}
    }
)

# Index design:
{
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "index": true,
                "similarity": "cosine"
            },
            "timestamp": {
                "type": "date",
                "index": true  # ← Enable filtering
            },
            "source_type": {
                "type": "keyword",
                "index": true  # ← Enable filtering
            }
        }
    }
}
```

#### Index Optimization Checklist

**Before going to production:**

- [ ] **Load test**: Test with expected query volume (100 QPS? 1000 QPS?)
- [ ] **Stress test**: What happens with 10x documents?
- [ ] **Measure recall**: Compare to brute force on sample queries
- [ ] **Monitor memory**: Ensure headroom (don't use >80% heap)
- [ ] **Test updates**: How fast can you add new documents?
- [ ] **Disaster recovery**: Can you rebuild index from source?
- [ ] **Backup strategy**: Snapshot index regularly

**Optimization process:**

```bash
# 1. Baseline measurement
curl -X POST http://localhost:8000/api/query \
  -d '{"query": "test"}' \
  -w "Time: %{time_total}s\n"

# 2. Try different configs
# Edit index settings, rebuild, remeasure

# 3. Compare results
echo "Config A: 25ms, recall 0.94"
echo "Config B: 15ms, recall 0.92"  # ← Choose this if speed > accuracy
echo "Config C: 40ms, recall 0.97"
```

---

### 4. Vector Store: Elasticsearch vs pgvector

#### Elasticsearch (Current)

```yaml
Pros:
  - Hybrid search (vector + keyword) built-in
  - Excellent BM25 keyword matching
  - Horizontal scaling ready
  - Rich query DSL
  - Great for mixed content (docs + logs + metrics)

Cons:
  - Higher memory usage (~2-4GB minimum)
  - More complex setup
  - Overkill for pure vector search

Best for:
  - Hybrid RAG systems (vector + keyword)
  - Multi-tenant applications
  - When you also need logging/metrics
```

#### pgvector (Alternative)

```yaml
Pros:
  - Lightweight (~100MB+ RAM)
  - Native SQL queries
  - Simpler deployment
  - Lower cost at scale
  - Great PostgreSQL integration

Cons:
  - Vector-only (no built-in keyword search)
  - Need separate BM25 implementation
  - Single-node focused

Best for:
  - Pure vector search
  - Cost-sensitive deployments
  - When you already use PostgreSQL
```

#### Performance Comparison

**Query Speed (100k vectors, 768 dims):**
```
Elasticsearch (HNSW):     ~10-50ms
pgvector (HNSW):          ~10-40ms
pgvector (IVFFlat):       ~50-200ms (more accurate)
```

**Storage (1M vectors, 768 dims):**
```
Elasticsearch:  ~5GB (with keyword index)
pgvector:       ~3GB (vectors only)
```

### 4. Similarity Metrics

Different metrics for different use cases:

```python
# Cosine Similarity (Current - default)
# Range: -1 to 1 (we use 0 to 1 for embeddings)
# Good for: General text similarity
cosine_sim = dot(A, B) / (norm(A) * norm(B))

# Euclidean Distance
# Range: 0 to infinity (lower = more similar)
# Good for: Clustering, when magnitude matters
euclidean = sqrt(sum((A - B)^2))

# Dot Product
# Range: -infinity to infinity
# Good for: When embedding magnitude is meaningful
dot_product = sum(A * B)
```

**When to use which:**
- **Cosine**: General RAG, Q&A, documentation (current choice ✅)
- **Euclidean**: Clustering, deduplication
- **Dot Product**: Ranking, when embeddings are normalized

### 5. Chunking Strategy

How you chunk affects retrieval quality significantly:

```python
# Current: Fixed-size with overlap
chunk_size = 500 words
overlap = 50 words

# Alternative strategies:

# 1. Semantic chunking (sentence-based)
chunks = split_by_sentences(doc, max_sentences=10)

# 2. Paragraph-based
chunks = split_by_paragraphs(doc)

# 3. Hierarchical
chunks = {
    'parent': full_document,
    'children': [para1, para2, ...]
}

# 4. Code-specific
chunks = {
    'function': function_with_docstring,
    'class': class_with_methods,
    'file': file_with_imports
}
```

**Trade-offs:**

| Strategy | Precision | Recall | Context | Best For |
|----------|-----------|--------|---------|----------|
| Small chunks (200w) | High | Low | Specific | Q&A, facts |
| Medium chunks (500w) | Medium | Medium | Balanced | General RAG ✅ |
| Large chunks (1000w) | Low | High | Broad | Summarization |
| Semantic | High | Medium | Natural | Docs, articles |
| Code-based | Highest | Medium | Structured | Source code |

---

## Testing & Evaluation Methodology

### 1. Create a Test Dataset

Build a **golden dataset** of question-answer pairs:

```python
test_queries = [
    {
        "query": "How do I install FastAPI?",
        "expected_answer": "pip install fastapi",
        "expected_docs": ["installation.html", "quickstart.html"],
        "category": "installation"
    },
    {
        "query": "What is dependency injection in FastAPI?",
        "expected_answer": "...",
        "expected_docs": ["advanced/dependencies.html"],
        "category": "advanced"
    }
]
```

### 2. Metrics to Track

#### Retrieval Metrics

```python
# Hit Rate: Did we find ANY relevant doc?
hit_rate = relevant_docs_found / total_queries

# MRR (Mean Reciprocal Rank): Where was the first relevant doc?
mrr = average(1 / rank_of_first_relevant_doc)

# NDCG (Normalized Discounted Cumulative Gain): Quality of ranking
ndcg = score_ranking_quality(retrieved_docs, relevant_docs)

# Precision@K: What % of top-K are relevant?
precision_at_3 = relevant_in_top_3 / 3
```

#### End-to-End Metrics

```python
# Answer Quality (requires human eval or LLM-as-judge)
answer_quality = llm_judge(query, answer, expected_answer)

# Latency
p50_latency = median(response_times)
p95_latency = percentile_95(response_times)

# Cost
cost_per_query = embedding_cost + llm_cost + infrastructure_cost
```

### 3. Evaluation Framework

```python
# Create evaluation script
from rag_evaluation import evaluate_retrieval

results = evaluate_retrieval(
    test_queries=test_queries,
    rag_service=rag_service,
    metrics=['hit_rate', 'mrr', 'ndcg@3', 'precision@3']
)

# Output:
# Hit Rate: 0.85 (85% of queries found relevant docs)
# MRR: 0.72 (average first relevant doc at position 1.4)
# NDCG@3: 0.68
# Precision@3: 0.62
```

### 4. A/B Testing Different Configurations

```python
configs = [
    {
        "name": "Baseline",
        "embedding": "all-MiniLM-L6-v2",
        "dims": 384,
        "chunk_size": 500,
        "threshold": 3.0
    },
    {
        "name": "Higher Quality",
        "embedding": "all-mpnet-base-v2",
        "dims": 768,
        "chunk_size": 500,
        "threshold": 3.0
    },
    {
        "name": "Larger Chunks",
        "embedding": "all-MiniLM-L6-v2",
        "dims": 384,
        "chunk_size": 1000,
        "threshold": 3.0
    }
]

# Run each config and compare
for config in configs:
    results = run_evaluation(config)
    print(f"{config['name']}: Hit Rate={results['hit_rate']}")
```

---

## Optimization Strategies

### 1. Improving Retrieval Quality

#### Strategy A: Better Embeddings

**Test different models:**

```bash
# Install test models
pip install sentence-transformers

# Test script
python << 'EOF'
from sentence_transformers import SentenceTransformer

models = [
    "all-MiniLM-L6-v2",      # 384 dims, fast
    "all-mpnet-base-v2",     # 768 dims, balanced
    "multi-qa-mpnet-base",   # 768 dims, optimized for Q&A
    "e5-large-v2"            # 1024 dims, SOTA
]

query = "How do I install FastAPI?"

for model_name in models:
    model = SentenceTransformer(model_name)
    embedding = model.encode(query)
    print(f"{model_name}: {len(embedding)} dims")
EOF
```

**Run comparative tests:**

```python
# Test each model with your golden dataset
results = {}
for model in models:
    rag_service.embedder = create_embedder(model)
    results[model] = evaluate(test_queries)

# Compare results
print_comparison_table(results)
# Output:
#                        Hit Rate  MRR   Latency  Cost/1k
# all-MiniLM-L6-v2       0.82      0.70  50ms     $0.01
# all-mpnet-base-v2      0.87      0.75  80ms     $0.02
# multi-qa-mpnet-base    0.90      0.78  80ms     $0.02  ← Best for Q&A
# e5-large-v2            0.92      0.82  120ms    $0.04
```

#### Strategy B: Optimize Chunking

```python
# Test different chunk sizes
chunk_configs = [
    {"size": 200, "overlap": 20},   # Small, precise
    {"size": 500, "overlap": 50},   # Current default
    {"size": 1000, "overlap": 100}, # Large, more context
]

for config in chunk_configs:
    chunker = DocumentChunker(**config)
    # Re-index with new chunking
    # Run evaluation
    # Compare hit rates
```

**Results you might see:**
```
Chunk Size    Hit Rate    Precision@3    Avg Answer Quality
200 words     0.75        0.85           7.5/10 (too fragmented)
500 words     0.85        0.75           8.5/10 (balanced) ✅
1000 words    0.90        0.65           7.0/10 (too broad)
```

#### Strategy C: Tune Relevance Threshold

```python
# Test different thresholds
thresholds = [1.0, 2.0, 3.0, 5.0, 8.0]

for threshold in thresholds:
    rag_service.relevance_threshold = threshold
    results = evaluate(test_queries)
    
    print(f"Threshold {threshold}:")
    print(f"  RAG answers: {results['rag_answers']}")
    print(f"  Fallback answers: {results['fallback_answers']}")
    print(f"  Answer quality: {results['avg_quality']}")
```

**Expected patterns:**
```
Threshold 1.0:  90% RAG, 10% fallback, quality=7.5 (uses bad docs)
Threshold 3.0:  70% RAG, 30% fallback, quality=8.5 (balanced) ✅
Threshold 8.0:  30% RAG, 70% fallback, quality=7.0 (too strict)
```

### 2. Improving Search Speed

```python
# Current: HNSW (Hierarchical Navigable Small World)
# Best for: Most RAG systems (good speed + accuracy)

"index_options": {
    "type": "hnsw",
    "m": 16,              # Connections per layer (16 = balanced)
    "ef_construction": 100 # Build-time accuracy (higher = slower build, better search)
}

# Tuning HNSW:
# - m=8: Faster, less accurate
# - m=16: Balanced (current) ✅
# - m=32: Slower, more accurate

# - ef_construction=50: Fast build, ok accuracy
# - ef_construction=100: Balanced (current) ✅
# - ef_construction=200: Slow build, best accuracy
```

### 3. Hybrid Search Tuning

Adjust the balance between vector and keyword search:

```python
# In Elasticsearch query:
{
    "query": {
        "bool": {
            "should": [
                {
                    "knn": {...},           # Vector search
                    "boost": 1.0            # Weight for semantic similarity
                },
                {
                    "multi_match": {...},   # Keyword search
                    "boost": 0.5            # Weight for exact matches
                }
            ]
        }
    }
}

# Tuning boosts:
# - Higher vector boost (1.5): Better for conceptual queries ("explain authentication")
# - Higher keyword boost (1.5): Better for specific terms ("FastAPI @app decorator")
# - Equal (1.0 each): Balanced (current) ✅
```

---

## Local Testing Workflows

### Workflow 1: Quick Model Comparison

```bash
#!/bin/bash
# test_models.sh - Compare different embedding models

models=("all-MiniLM-L6-v2" "all-mpnet-base-v2" "multi-qa-mpnet-base")

for model in "${models[@]}"; do
    echo "Testing $model..."
    
    # Update config
    export EMBEDDING_MODEL=$model
    
    # Restart API
    pkill -f rag_api.py
    python rag_api.py &
    sleep 10
    
    # Run test queries
    python << EOF
import requests
queries = [
    "How do I install FastAPI?",
    "What is dependency injection?",
    "How to add middleware?"
]

for q in queries:
    resp = requests.post("http://localhost:8000/api/query", 
                        json={"query": q}).json()
    print(f"Score: {resp['retrieved_docs'][0]['score']:.2f}")
EOF
    
    echo "---"
done
```

### Workflow 2: Threshold Tuning

```python
#!/usr/bin/env python3
# tune_threshold.py

import requests
import json

# Your golden test set
test_cases = [
    {"query": "How to install FastAPI?", "has_docs": True},
    {"query": "What is Java?", "has_docs": False},  # No Java docs
    {"query": "FastAPI dependency injection", "has_docs": True},
]

thresholds = [1.0, 2.0, 3.0, 5.0, 8.0]

for threshold in thresholds:
    # Update threshold in code, restart API
    # ...
    
    correct_rag = 0
    correct_fallback = 0
    
    for test in test_cases:
        resp = requests.post("http://localhost:8000/api/query",
                           json={"query": test["query"]}).json()
        
        used_rag = len(resp['retrieved_docs']) > 0
        
        if test["has_docs"] and used_rag:
            correct_rag += 1
        elif not test["has_docs"] and not used_rag:
            correct_fallback += 1
    
    accuracy = (correct_rag + correct_fallback) / len(test_cases)
    print(f"Threshold {threshold}: {accuracy:.1%} accuracy")
```

### Workflow 3: End-to-End Quality Testing

```python
#!/usr/bin/env python3
# evaluate_system.py - Full system evaluation

from typing import List, Dict
import requests

class RAGEvaluator:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def evaluate(self, test_set: List[Dict]) -> Dict:
        """Run full evaluation on test set"""
        results = {
            "total": len(test_set),
            "rag_used": 0,
            "fallback_used": 0,
            "scores": [],
            "latencies": []
        }
        
        for test in test_set:
            resp = requests.post(
                f"{self.api_url}/api/query",
                json={"query": test["query"]}
            ).json()
            
            if resp.get('retrieved_docs'):
                results["rag_used"] += 1
                results["scores"].append(
                    resp['retrieved_docs'][0]['score']
                )
            else:
                results["fallback_used"] += 1
            
            results["latencies"].append(resp.get('latency_ms', 0))
        
        # Calculate metrics
        results["avg_score"] = sum(results["scores"]) / len(results["scores"]) if results["scores"] else 0
        results["p50_latency"] = sorted(results["latencies"])[len(results["latencies"])//2]
        
        return results

# Usage:
test_set = load_test_queries("test_queries.json")
evaluator = RAGEvaluator("http://localhost:8000")
results = evaluator.evaluate(test_set)

print(json.dumps(results, indent=2))
```

### Workflow 4: Vector Store Comparison

**Test Elasticsearch vs pgvector:**

```bash
# 1. Setup pgvector (Docker)
docker run -d --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ankane/pgvector

# 2. Create schema
psql -h localhost -U postgres << EOF
CREATE EXTENSION vector;

CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(768)
);

CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);
EOF

# 3. Index same data in both
python index_to_both.py

# 4. Run comparative tests
python benchmark.py --store elasticsearch --queries test_queries.json
python benchmark.py --store pgvector --queries test_queries.json

# Compare:
# - Query speed
# - Memory usage
# - Result quality
# - Cost
```

### Workflow 5: Benchmark Different Dimension Sizes

**Test how embedding dimensions affect your specific use case:**

This workflow helps you find the optimal balance between quality, speed, and cost for YOUR knowledge base.

#### Step 1: Prepare Your Test Environment

```bash
# 1. Create a test queries file with YOUR actual queries
cat > my_test_queries.json << 'EOF'
[
  {
    "query": "How do I install FastAPI?",
    "expected_source": "fastapi.tiangolo.com/installation",
    "category": "installation"
  },
  {
    "query": "What is dependency injection?",
    "expected_source": "fastapi.tiangolo.com/advanced",
    "category": "advanced"
  },
  {
    "query": "How to add CORS middleware?",
    "expected_source": "fastapi.tiangolo.com/tutorial",
    "category": "middleware"
  }
]
EOF

# 2. Make sure you have content indexed (use Admin UI or ETL)
# This workflow assumes you already have documents in your knowledge base
```

#### Step 2: Create the Benchmark Script

```python
#!/usr/bin/env python3
# benchmark_dimensions.py - Compare different embedding dimensions

import time
import json
import statistics
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from rag_embeddings import create_embeddings, EmbeddingConfig

# Test configurations
DIMENSION_CONFIGS = [
    {
        "name": "Small (384 dims)",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "index": "knowledge_base_384"
    },
    {
        "name": "Medium (768 dims)",
        "model": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "index": "knowledge_base_768"
    },
    {
        "name": "Large (1536 dims)",
        "model": "sentence-transformers/all-mpnet-base-v2",  # Will pad/truncate
        "dimensions": 1536,
        "index": "knowledge_base_1536"
    }
]

class DimensionBenchmark:
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.es = Elasticsearch(es_url)
        self.results = []
    
    def create_index(self, index_name: str, dimensions: int):
        """Create an Elasticsearch index with specified dimensions"""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "title": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100
                        }
                    },
                    "source_url": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        # Delete if exists
        if self.es.indices.exists(index=index_name):
            print(f"  Deleting existing index: {index_name}")
            self.es.indices.delete(index=index_name)
        
        # Create new
        self.es.indices.create(index=index_name, mappings=mapping["mappings"])
        print(f"  Created index: {index_name} with {dimensions} dimensions")
    
    def reindex_with_dimensions(self, source_index: str, target_index: str, 
                               model_name: str, target_dims: int):
        """Re-index documents with different embedding dimensions"""
        print(f"\n  Re-indexing from {source_index} to {target_index}...")
        
        # Load the embedding model
        model = SentenceTransformer(model_name)
        
        # Get all documents from source
        query = {"query": {"match_all": {}}, "size": 10000}
        response = self.es.search(index=source_index, **query)
        docs = response['hits']['hits']
        
        print(f"  Found {len(docs)} documents to re-embed")
        
        # Re-embed and index
        for i, doc in enumerate(docs):
            source = doc['_source']
            
            # Generate new embedding with target model
            embedding = model.encode(source['content']).tolist()
            
            # Pad or truncate to target dimensions
            if len(embedding) < target_dims:
                embedding = embedding + [0.0] * (target_dims - len(embedding))
            elif len(embedding) > target_dims:
                embedding = embedding[:target_dims]
            
            # Index document with new embedding
            self.es.index(
                index=target_index,
                id=doc['_id'],
                document={
                    **source,
                    "embedding": embedding
                }
            )
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(docs)} documents")
        
        # Refresh index
        self.es.indices.refresh(index=target_index)
        print(f"  ✅ Re-indexing complete: {len(docs)} documents")
    
    def benchmark_config(self, config: Dict, test_queries: List[Dict]) -> Dict:
        """Benchmark a specific dimension configuration"""
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        model = SentenceTransformer(config['model'])
        index = config['index']
        dims = config['dimensions']
        
        # Metrics to track
        query_times = []
        relevance_scores = []
        hits = 0
        embedding_times = []
        
        for i, test in enumerate(test_queries, 1):
            query = test['query']
            
            # Time embedding generation
            embed_start = time.time()
            query_embedding = model.encode(query).tolist()
            
            # Pad or truncate
            if len(query_embedding) < dims:
                query_embedding = query_embedding + [0.0] * (dims - len(query_embedding))
            elif len(query_embedding) > dims:
                query_embedding = query_embedding[:dims]
            
            embedding_time = (time.time() - embed_start) * 1000
            embedding_times.append(embedding_time)
            
            # Time search query
            search_start = time.time()
            
            search_query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": 3
            }
            
            try:
                response = self.es.search(index=index, **search_query)
                search_time = (time.time() - search_start) * 1000
                query_times.append(search_time)
                
                # Get top result
                if response['hits']['hits']:
                    top_hit = response['hits']['hits'][0]
                    score = top_hit['_score']
                    relevance_scores.append(score)
                    
                    # Check if it's a hit (contains expected source)
                    if 'expected_source' in test:
                        if test['expected_source'] in top_hit['_source'].get('source_url', ''):
                            hits += 1
                    
                    print(f"  Query {i}: {query[:50]:50} | "
                          f"Score: {score:5.2f} | "
                          f"Embed: {embedding_time:4.0f}ms | "
                          f"Search: {search_time:4.0f}ms")
                else:
                    print(f"  Query {i}: No results")
            
            except Exception as e:
                print(f"  Query {i}: Error - {e}")
        
        # Calculate statistics
        results = {
            "config": config['name'],
            "dimensions": dims,
            "model": config['model'],
            "total_queries": len(test_queries),
            "hit_rate": hits / len(test_queries) if test_queries else 0,
            "avg_relevance_score": statistics.mean(relevance_scores) if relevance_scores else 0,
            "median_relevance_score": statistics.median(relevance_scores) if relevance_scores else 0,
            "avg_embedding_time_ms": statistics.mean(embedding_times) if embedding_times else 0,
            "avg_search_time_ms": statistics.mean(query_times) if query_times else 0,
            "total_time_ms": statistics.mean(embedding_times) + statistics.mean(query_times) if embedding_times and query_times else 0,
            "p95_total_time_ms": sorted([e+s for e,s in zip(embedding_times, query_times)])[int(len(query_times)*0.95)] if query_times else 0,
            "index_size_estimate_mb": self._estimate_index_size(dims, len(test_queries))
        }
        
        return results
    
    def _estimate_index_size(self, dims: int, doc_count: int) -> float:
        """Estimate index size in MB"""
        # Vector storage: dims × 4 bytes × doc_count
        vector_mb = (dims * 4 * doc_count) / (1024 * 1024)
        
        # HNSW graph: ~m × 2 × 8 bytes × doc_count (m=16)
        graph_mb = (16 * 2 * 8 * doc_count) / (1024 * 1024)
        
        # Elasticsearch overhead: ~20%
        total_mb = (vector_mb + graph_mb) * 1.2
        
        return total_mb
    
    def print_comparison(self, results: List[Dict]):
        """Print comparison table"""
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Config':<20} {'Dims':>6} {'Hit%':>6} {'AvgScore':>9} "
              f"{'Embed(ms)':>10} {'Search(ms)':>11} {'Total(ms)':>10} {'Size(MB)':>10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['config']:<20} "
                  f"{r['dimensions']:>6} "
                  f"{r['hit_rate']*100:>5.1f}% "
                  f"{r['avg_relevance_score']:>9.2f} "
                  f"{r['avg_embedding_time_ms']:>10.1f} "
                  f"{r['avg_search_time_ms']:>11.1f} "
                  f"{r['total_time_ms']:>10.1f} "
                  f"{r['index_size_estimate_mb']:>10.1f}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80 + "\n")
        
        # Find best configurations
        best_speed = min(results, key=lambda x: x['total_time_ms'])
        best_quality = max(results, key=lambda x: x['hit_rate'])
        best_size = min(results, key=lambda x: x['index_size_estimate_mb'])
        
        print(f"🚀 Fastest:        {best_speed['config']} "
              f"({best_speed['total_time_ms']:.1f}ms avg latency)")
        print(f"🎯 Most Accurate:  {best_quality['config']} "
              f"({best_quality['hit_rate']*100:.1f}% hit rate)")
        print(f"💾 Smallest:       {best_size['config']} "
              f"({best_size['index_size_estimate_mb']:.1f}MB estimated)")
        
        print("\n💡 Choosing the right configuration:")
        print("   - For <100ms queries: Choose Fastest")
        print("   - For best answers: Choose Most Accurate")
        print("   - For cost savings: Choose Smallest")
        print("   - For production: Balance all three\n")

def main():
    # Load test queries
    with open('my_test_queries.json') as f:
        test_queries = json.load(f)
    
    print(f"Loaded {len(test_queries)} test queries")
    
    benchmark = DimensionBenchmark()
    all_results = []
    
    # Assume you have a source index with current embeddings
    source_index = "knowledge_base"
    
    for config in DIMENSION_CONFIGS:
        # Create index with target dimensions
        benchmark.create_index(config['index'], config['dimensions'])
        
        # Re-index documents with new embeddings
        benchmark.reindex_with_dimensions(
            source_index=source_index,
            target_index=config['index'],
            model_name=config['model'],
            target_dims=config['dimensions']
        )
        
        # Run benchmark
        results = benchmark.benchmark_config(config, test_queries)
        all_results.append(results)
        
        # Wait a bit between tests
        time.sleep(2)
    
    # Print comparison
    benchmark.print_comparison(all_results)
    
    # Save results to file
    with open('dimension_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: dimension_benchmark_results.json")
    print(f"📊 Visualize with: python plot_benchmark.py dimension_benchmark_results.json\n")

if __name__ == "__main__":
    main()
```

#### Step 3: Run the Benchmark

```bash
# Make sure Elasticsearch is running
curl http://localhost:9200/_cluster/health

# Make sure you have documents indexed
curl http://localhost:9200/knowledge_base/_count

# Run the benchmark
python benchmark_dimensions.py

# Expected output:
# ============================================================
# Testing: Small (384 dims)
# ============================================================
#   Query 1: How do I install FastAPI?            | Score:  6.45 | Embed:   25ms | Search:   45ms
#   Query 2: What is dependency injection?        | Score:  5.20 | Embed:   23ms | Search:   42ms
#   ...
#
# ============================================================
# Testing: Medium (768 dims)
# ============================================================
#   Query 1: How do I install FastAPI?            | Score:  6.89 | Embed:   45ms | Search:   65ms
#   ...
```

#### Step 4: Analyze Results

The benchmark will output a comparison table:

```
================================================================================
BENCHMARK RESULTS COMPARISON
================================================================================

Config                Dims   Hit%  AvgScore  Embed(ms)  Search(ms)  Total(ms)  Size(MB)
--------------------------------------------------------------------------------
Small (384 dims)       384   82.0%      6.24       24.3        43.2       67.5      120.5
Medium (768 dims)      768   87.0%      6.58       44.8        58.7      103.5      230.8
Large (1536 dims)     1536   89.0%      6.72       89.2        95.3      184.5      450.2

================================================================================
RECOMMENDATIONS:
================================================================================

🚀 Fastest:        Small (384 dims) (67.5ms avg latency)
🎯 Most Accurate:  Large (1536 dims) (89.0% hit rate)
💾 Smallest:       Small (384 dims) (120.5MB estimated)

💡 Choosing the right configuration:
   - For <100ms queries: Choose Fastest
   - For best answers: Choose Most Accurate
   - For cost savings: Choose Smallest
   - For production: Balance all three
```

#### Step 5: Interpret Your Results

**Key questions to answer:**

1. **Is the quality improvement worth the cost?**
   ```
   384 dims: 82% hit rate, 67ms, 120MB
   768 dims: 87% hit rate, 103ms, 230MB  ← +5% accuracy for +54% latency
   
   Decision: If 5% accuracy is worth 36ms extra latency, use 768
   ```

2. **What's your bottleneck?**
   ```
   If Embed(ms) > Search(ms):
     → Bottleneck is embedding generation (CPU/GPU)
     → Consider smaller model or batch processing
   
   If Search(ms) > Embed(ms):
     → Bottleneck is Elasticsearch (RAM/disk)
     → Consider more RAM or optimize HNSW params
   ```

3. **How does it scale?**
   ```
   Your test: 100 docs, 384 dims = 120MB
   Production: 100,000 docs, 384 dims = 120GB (1000x)
   
   Can your infrastructure handle this?
   ```

#### Step 6: Optional - Visualize Results

```python
#!/usr/bin/env python3
# plot_benchmark.py - Visualize benchmark results

import json
import sys
import matplotlib.pyplot as plt

def plot_results(results_file):
    with open(results_file) as f:
        results = json.load(f)
    
    configs = [r['config'] for r in results]
    dimensions = [r['dimensions'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Hit Rate vs Dimensions
    hit_rates = [r['hit_rate'] * 100 for r in results]
    ax1.bar(configs, hit_rates, color='green', alpha=0.7)
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title('Search Accuracy by Dimensions')
    ax1.axhline(y=80, color='r', linestyle='--', label='80% target')
    ax1.legend()
    
    # Plot 2: Latency vs Dimensions
    latencies = [r['total_time_ms'] for r in results]
    ax2.bar(configs, latencies, color='blue', alpha=0.7)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Query Latency by Dimensions')
    ax2.axhline(y=100, color='r', linestyle='--', label='100ms target')
    ax2.legend()
    
    # Plot 3: Index Size vs Dimensions
    sizes = [r['index_size_estimate_mb'] for r in results]
    ax3.bar(configs, sizes, color='orange', alpha=0.7)
    ax3.set_ylabel('Size (MB)')
    ax3.set_title('Index Size by Dimensions')
    
    # Plot 4: Speed vs Accuracy Trade-off
    ax4.scatter(latencies, hit_rates, s=200, alpha=0.6)
    for i, config in enumerate(configs):
        ax4.annotate(config, (latencies[i], hit_rates[i]))
    ax4.set_xlabel('Latency (ms)')
    ax4.set_ylabel('Hit Rate (%)')
    ax4.set_title('Speed vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimension_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: dimension_benchmark.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_benchmark.py dimension_benchmark_results.json")
        sys.exit(1)
    
    plot_results(sys.argv[1])
```

#### Step 7: Make Your Decision

**Decision Matrix:**

| Priority | Dimensions | Reason |
|----------|-----------|---------|
| **Speed First** (real-time apps) | 384 | Sub-100ms queries, lower compute |
| **Quality First** (critical answers) | 768-1536 | Best accuracy, worth the cost |
| **Cost First** (budget constrained) | 384 | Smallest storage, fastest indexing |
| **Balanced** (most production systems) | 768 | Sweet spot for quality/speed/cost ✅ |

**Update your configuration:**

```python
# rag_embeddings.py
EMBEDDING_CONFIGS = {
    'local_minilm': EmbeddingConfig(
        provider='local',
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        dimensions=384  # ← Change this based on your benchmark
    )
}
```

#### Tips for Accurate Benchmarking

1. **Use realistic queries**: Don't test with toy examples
2. **Test with production data**: Index your actual documents
3. **Measure end-to-end**: Include embedding + search time
4. **Consider your workload**: Batch queries? Real-time? Concurrent users?
5. **Test at scale**: 100 docs ≠ 100,000 docs performance
6. **Monitor in production**: Benchmarks ≠ real-world usage

---

### Workflow 6: Benchmark Matryoshka Embeddings (OpenAI)

**What are Matryoshka embeddings?**

Traditional embeddings: Each dimension size needs a separate model
```
384-dim model → 384 dimensions only
768-dim model → 768 dimensions only (cannot use first 384)
```

Matryoshka embeddings: One model, multiple dimension sizes!
```
text-embedding-3-large (3072 dims) → Use ANY prefix:
  - First 256 dims  (fastest, lowest quality)
  - First 512 dims  (balanced)
  - First 1024 dims (good quality)
  - First 1536 dims (high quality)
  - All 3072 dims   (best quality, slowest)
```

**Key advantage:** Truncating maintains quality (unlike padding/truncating regular embeddings).

#### Why Use Matryoshka Embeddings?

```yaml
Traditional Approach:
  - Need 3 separate models for 384, 768, 1536 dims
  - Can't change dimensions after indexing
  - Storage: 3 separate indexes

Matryoshka Approach:
  - ONE model (text-embedding-3-large)
  - Test 5+ dimension sizes from same embeddings
  - Can switch dimensions without re-embedding
  - Storage: Just slice existing vectors
```

#### Supported Models

| Model | Max Dims | Recommended Truncations | Cost | Quality |
|-------|----------|------------------------|------|---------|
| **text-embedding-3-small** | 1536 | 256, 512, 1024, 1536 | $ | ⭐⭐⭐⭐ |
| **text-embedding-3-large** | 3072 | 256, 512, 1024, 1536, 3072 | $$ | ⭐⭐⭐⭐⭐ |

#### Benchmark Script for Matryoshka Embeddings

```python
#!/usr/bin/env python3
# benchmark_matryoshka.py - Test OpenAI Matryoshka embeddings

import os
import time
import json
import statistics
from typing import List, Dict
from openai import OpenAI
from elasticsearch import Elasticsearch

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Matryoshka dimension configurations to test
MATRYOSHKA_CONFIGS = [
    {"name": "Tiny (256 dims)", "dimensions": 256},
    {"name": "Small (512 dims)", "dimensions": 512},
    {"name": "Medium (1024 dims)", "dimensions": 1024},
    {"name": "Large (1536 dims)", "dimensions": 1536},
    {"name": "XL (3072 dims)", "dimensions": 3072},  # Only for text-embedding-3-large
]

class MatryoshkaBenchmark:
    def __init__(self, 
                 model: str = "text-embedding-3-large",
                 es_url: str = "http://localhost:9200"):
        self.model = model
        self.es = Elasticsearch(es_url)
        self.full_embeddings_cache = {}  # Cache full embeddings
    
    def get_embedding(self, text: str, dimensions: int = None) -> List[float]:
        """Get OpenAI embedding with optional dimension truncation"""
        cache_key = text
        
        # Get full embedding (cache for reuse)
        if cache_key not in self.full_embeddings_cache:
            response = client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            self.full_embeddings_cache[cache_key] = response.data[0].embedding
        
        full_embedding = self.full_embeddings_cache[cache_key]
        
        # Truncate to desired dimensions
        if dimensions and dimensions < len(full_embedding):
            return full_embedding[:dimensions]
        return full_embedding
    
    def create_index(self, index_name: str, dimensions: int):
        """Create Elasticsearch index for specific dimensions"""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "title": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100
                        }
                    },
                    "source_url": {"type": "keyword"},
                    "full_embedding_cached": {"type": "binary"}  # Store full for later
                }
            }
        }
        
        if self.es.indices.exists(index=index_name):
            print(f"  Deleting existing index: {index_name}")
            self.es.indices.delete(index=index_name)
        
        self.es.indices.create(index=index_name, mappings=mapping["mappings"])
        print(f"  Created index: {index_name} with {dimensions} dimensions")
    
    def index_with_matryoshka(self, source_index: str, target_index: str, 
                              target_dims: int):
        """
        Index documents using Matryoshka embeddings
        Key: Generate FULL embedding once, then truncate for different sizes
        """
        print(f"\n  Indexing to {target_index} with {target_dims} dimensions...")
        
        # Get documents from source
        query = {"query": {"match_all": {}}, "size": 10000}
        response = self.es.search(index=source_index, **query)
        docs = response['hits']['hits']
        
        print(f"  Found {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            source = doc['_source']
            
            # Get full embedding (cached), then truncate
            full_embedding = self.get_embedding(source['content'])
            truncated_embedding = full_embedding[:target_dims]
            
            # Index with truncated embedding
            self.es.index(
                index=target_index,
                id=doc['_id'],
                document={
                    **source,
                    "embedding": truncated_embedding
                }
            )
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(docs)} documents")
        
        self.es.indices.refresh(index=target_index)
        print(f"  ✅ Indexed {len(docs)} documents with {target_dims} dims")
    
    def benchmark_dimension(self, config: Dict, test_queries: List[Dict],
                           index_name: str) -> Dict:
        """Benchmark specific dimension size"""
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        dims = config['dimensions']
        
        query_times = []
        relevance_scores = []
        hits = 0
        embedding_times = []
        embedding_costs = []
        
        for i, test in enumerate(test_queries, 1):
            query = test['query']
            
            # Time embedding generation
            embed_start = time.time()
            query_embedding = self.get_embedding(query, dimensions=dims)
            embedding_time = (time.time() - embed_start) * 1000
            embedding_times.append(embedding_time)
            
            # Cost estimate (OpenAI pricing)
            # text-embedding-3-small: $0.02 per 1M tokens
            # text-embedding-3-large: $0.13 per 1M tokens
            cost_per_1m_tokens = 0.13 if "large" in self.model else 0.02
            # Rough estimate: ~1 token per 4 chars
            tokens = len(query) / 4
            cost = (tokens / 1_000_000) * cost_per_1m_tokens
            embedding_costs.append(cost)
            
            # Time search
            search_start = time.time()
            
            search_query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": 3
            }
            
            try:
                response = self.es.search(index=index_name, **search_query)
                search_time = (time.time() - search_start) * 1000
                query_times.append(search_time)
                
                if response['hits']['hits']:
                    top_hit = response['hits']['hits'][0]
                    score = top_hit['_score']
                    relevance_scores.append(score)
                    
                    if 'expected_source' in test:
                        if test['expected_source'] in top_hit['_source'].get('source_url', ''):
                            hits += 1
                    
                    print(f"  Q{i}: {query[:45]:45} | "
                          f"Score: {score:5.2f} | "
                          f"Embed: {embedding_time:4.0f}ms | "
                          f"Search: {search_time:4.0f}ms | "
                          f"Cost: ${cost*1000:.4f}")
                else:
                    print(f"  Q{i}: No results")
            
            except Exception as e:
                print(f"  Q{i}: Error - {e}")
        
        results = {
            "config": config['name'],
            "dimensions": dims,
            "model": self.model,
            "total_queries": len(test_queries),
            "hit_rate": hits / len(test_queries) if test_queries else 0,
            "avg_relevance_score": statistics.mean(relevance_scores) if relevance_scores else 0,
            "avg_embedding_time_ms": statistics.mean(embedding_times) if embedding_times else 0,
            "avg_search_time_ms": statistics.mean(query_times) if query_times else 0,
            "total_time_ms": statistics.mean(embedding_times) + statistics.mean(query_times) if embedding_times and query_times else 0,
            "cost_per_query": statistics.mean(embedding_costs) if embedding_costs else 0,
            "cost_per_1k_queries": statistics.mean(embedding_costs) * 1000 if embedding_costs else 0,
            "index_size_mb": self._estimate_size(dims, len(test_queries))
        }
        
        return results
    
    def _estimate_size(self, dims: int, doc_count: int) -> float:
        """Estimate index size"""
        vector_mb = (dims * 4 * doc_count) / (1024 * 1024)
        graph_mb = (16 * 2 * 8 * doc_count) / (1024 * 1024)
        return (vector_mb + graph_mb) * 1.2
    
    def print_comparison(self, results: List[Dict]):
        """Print comparison table"""
        print(f"\n{'='*90}")
        print("MATRYOSHKA EMBEDDING BENCHMARK RESULTS")
        print(f"{'='*90}\n")
        
        print(f"Model: {self.model}")
        print(f"{'Config':<20} {'Dims':>6} {'Hit%':>6} {'Score':>7} "
              f"{'Total(ms)':>10} {'$/1k':>8} {'Size(MB)':>10}")
        print("-" * 90)
        
        for r in results:
            print(f"{r['config']:<20} "
                  f"{r['dimensions']:>6} "
                  f"{r['hit_rate']*100:>5.1f}% "
                  f"{r['avg_relevance_score']:>7.2f} "
                  f"{r['total_time_ms']:>10.1f} "
                  f"${r['cost_per_1k_queries']:>7.2f} "
                  f"{r['index_size_mb']:>10.1f}")
        
        print("\n" + "="*90)
        print("RECOMMENDATIONS:")
        print("="*90 + "\n")
        
        best_speed = min(results, key=lambda x: x['total_time_ms'])
        best_quality = max(results, key=lambda x: x['hit_rate'])
        best_cost = min(results, key=lambda x: x['cost_per_1k_queries'])
        best_value = max(results, key=lambda x: x['hit_rate'] / (r['total_time_ms'] / 100))
        
        print(f"⚡ Fastest:       {best_speed['config']} ({best_speed['total_time_ms']:.0f}ms)")
        print(f"🎯 Most Accurate: {best_quality['config']} ({best_quality['hit_rate']*100:.1f}% hit rate)")
        print(f"💰 Cheapest:      {best_cost['config']} (${best_cost['cost_per_1k_queries']:.2f}/1k queries)")
        print(f"⭐ Best Value:    {best_value['config']} (accuracy/speed ratio)")
        
        print("\n🔍 Matryoshka Insight:")
        print("   Notice how hit rate changes with dimensions - find your sweet spot!")
        
        # Calculate efficiency
        if len(results) > 1:
            smallest = results[0]
            largest = results[-1]
            
            quality_gain = (largest['hit_rate'] - smallest['hit_rate']) * 100
            speed_loss = ((largest['total_time_ms'] - smallest['total_time_ms']) / 
                         smallest['total_time_ms']) * 100
            
            print(f"\n📊 Scaling Analysis:")
            print(f"   {smallest['dimensions']} → {largest['dimensions']} dims:")
            print(f"   • Quality gain: +{quality_gain:.1f} percentage points")
            print(f"   • Speed cost: +{speed_loss:.1f}%")
            print(f"   • Worth it? {quality_gain / (speed_loss + 0.01):.2f}x quality/speed ratio")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Matryoshka embeddings')
    parser.add_argument('--model', default='text-embedding-3-large',
                       choices=['text-embedding-3-small', 'text-embedding-3-large'],
                       help='OpenAI embedding model')
    parser.add_argument('--queries', default='my_test_queries.json',
                       help='Test queries JSON file')
    parser.add_argument('--source-index', default='knowledge_base',
                       help='Source Elasticsearch index')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   export OPENAI_API_KEY='sk-...'")
        return
    
    # Load test queries
    with open(args.queries) as f:
        test_queries = json.load(f)
    
    print(f"📚 Loaded {len(test_queries)} test queries")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Testing {len(MATRYOSHKA_CONFIGS)} dimension configurations\n")
    
    benchmark = MatryoshkaBenchmark(model=args.model)
    all_results = []
    
    # Filter configs based on model max dimensions
    max_dims = 3072 if "large" in args.model else 1536
    configs = [c for c in MATRYOSHKA_CONFIGS if c['dimensions'] <= max_dims]
    
    for config in configs:
        dims = config['dimensions']
        index_name = f"knowledge_base_matryoshka_{dims}"
        
        # Create index
        benchmark.create_index(index_name, dims)
        
        # Index with truncated embeddings
        benchmark.index_with_matryoshka(args.source_index, index_name, dims)
        
        # Benchmark
        results = benchmark.benchmark_dimension(config, test_queries, index_name)
        all_results.append(results)
        
        time.sleep(1)
    
    # Print comparison
    benchmark.print_comparison(all_results)
    
    # Save results
    output_file = f'matryoshka_benchmark_{args.model}.json'
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "test_queries": len(test_queries),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
```

#### How to Run

```bash
# 1. Set up OpenAI API key
export OPENAI_API_KEY='sk-proj-...'

# 2. Make sure you have documents indexed
curl http://localhost:9200/knowledge_base/_count

# 3. Run benchmark with text-embedding-3-large (best quality)
python benchmark_matryoshka.py --model text-embedding-3-large --queries my_test_queries.json

# Or test text-embedding-3-small (cheaper)
python benchmark_matryoshka.py --model text-embedding-3-small --queries my_test_queries.json
```

#### Expected Output

```
================================================================================
MATRYOSHKA EMBEDDING BENCHMARK RESULTS
================================================================================

Model: text-embedding-3-large
Config                Dims   Hit%   Score  Total(ms)     $/1k  Size(MB)
----------------------------------------------------------------------------------
Tiny (256 dims)        256   78.0%    5.92       45.2   $0.15      85.3
Small (512 dims)       512   84.0%    6.28       52.8   $0.15     160.4
Medium (1024 dims)    1024   88.0%    6.45       68.5   $0.15     310.6
Large (1536 dims)     1536   90.0%    6.58       89.3   $0.15     455.9
XL (3072 dims)        3072   91.0%    6.65      142.7   $0.15     901.8

================================================================================
RECOMMENDATIONS:
================================================================================

⚡ Fastest:       Tiny (256 dims) (45ms)
🎯 Most Accurate: XL (3072 dims) (91.0% hit rate)
💰 Cheapest:      Tiny (256 dims) ($0.15/1k queries)
⭐ Best Value:    Medium (1024 dims) (accuracy/speed ratio)

🔍 Matryoshka Insight:
   Notice how hit rate changes with dimensions - find your sweet spot!

📊 Scaling Analysis:
   256 → 3072 dims:
   • Quality gain: +13.0 percentage points
   • Speed cost: +215.7%
   • Worth it? 4.12x quality/speed ratio
```

#### Key Insights

**1. Diminishing Returns:**
```
256 → 512 dims:  +6% accuracy for +17% latency  ✅ Good trade-off
512 → 1024 dims: +4% accuracy for +30% latency  ✅ Still worth it
1024 → 1536 dims: +2% accuracy for +30% latency ⚠️ Questionable
1536 → 3072 dims: +1% accuracy for +60% latency ❌ Not worth it
```

**2. Cost is Constant!**
```
All dimensions cost the same ($0.15/1k queries)
You only pay for the full embedding generation
Truncation is free!
```

**3. Sweet Spot for Most Use Cases:**
```
text-embedding-3-large with 1024 dims:
  • 88% hit rate (good enough)
  • 68ms latency (acceptable)
  • $0.15/1k queries (reasonable)
  • 311MB index size (manageable)
```

#### Comparison with Traditional Embeddings

| Approach | Model Switching | Re-indexing | Dimension Flexibility | Quality |
|----------|----------------|-------------|---------------------|---------|
| **Traditional** | Need new model | Full re-embed | Fixed after indexing | ⭐⭐⭐⭐ |
| **Matryoshka** | Same model | Just re-slice | Change anytime | ⭐⭐⭐⭐⭐ |

#### Production Recommendation

```python
# rag_embeddings.py - Add OpenAI Matryoshka config

EMBEDDING_CONFIGS = {
    'openai_matryoshka': EmbeddingConfig(
        provider='openai',
        model_name='text-embedding-3-large',
        dimensions=1024,  # Start here, adjust based on benchmark
        api_key=os.environ.get('OPENAI_API_KEY')
    )
}
```

**When to use Matryoshka:**
- ✅ You want flexibility to tune dimensions without re-embedding
- ✅ You're using OpenAI embeddings anyway
- ✅ You want best-in-class quality with cost control
- ✅ You might scale from 10k → 1M documents (can reduce dims later)

**When to stick with traditional:**
- ✅ You're using local models (sentence-transformers)
- ✅ You're using Vertex AI embeddings
- ✅ You want to avoid API dependency
- ✅ Cost is not a concern

---

#### Tips for Accurate Benchmarking

1. **Use realistic queries**: Don't test with toy examples
2. **Test with production data**: Index your actual documents
3. **Measure end-to-end**: Include embedding + search time
4. **Consider your workload**: Batch queries? Real-time? Concurrent users?
5. **Test at scale**: 100 docs ≠ 100,000 docs performance
6. **Monitor in production**: Benchmarks ≠ real-world usage

---

## Real-World Examples

### Example 1: Documentation RAG System (Current Use Case)

**Optimal Configuration:**
```yaml
Embedding Model: all-mpnet-base-v2 or text-embedding-004
Dimensions: 768
Vector Store: Elasticsearch (for hybrid search)
Chunk Size: 500 words, 50 overlap
Similarity: Cosine
Threshold: 3.0
Hybrid Search: 60% vector, 40% keyword
```

**Why:**
- Docs have natural language → general embeddings work well
- Users ask questions → need good Q&A performance
- Mix of exact terms ("@app.get") and concepts ("dependency injection")
- Hybrid search handles both

**Testing checklist:**
```python
test_queries = [
    # Exact matches (keyword-heavy)
    "What is @app.get decorator?",
    "How to use Pydantic models?",
    
    # Conceptual (vector-heavy)
    "How does FastAPI handle async?",
    "Best practices for API design",
    
    # Mixed
    "How to add authentication middleware?",
    
    # Out-of-domain (should fallback)
    "What is React?",
    "Explain neural networks"
]
```

### Example 2: Source Code RAG System

**Optimal Configuration:**
```yaml
Embedding Model: CodeBERT or graphcodebert
Dimensions: 768
Vector Store: Elasticsearch or pgvector
Chunk Size: Function-level (not word-based)
Similarity: Cosine
Threshold: 4.0 (stricter for code)
Search Features:
  - Syntax-aware chunking
  - Import/dependency tracking
  - Language-specific parsing
```

**Chunking for code:**
```python
# Bad: Word-based chunking
chunk = "def handle_auth(request):\n    token = request.headers.get('Authorization')"

# Good: Function-level with context
chunk = """
# File: auth/handlers.py
# Imports: jwt, fastapi.Request

def handle_auth(request: Request) -> User:
    \"\"\"Validates JWT token and returns user\"\"\"
    token = request.headers.get('Authorization')
    if not token:
        raise HTTPException(401)
    return verify_token(token)
"""
```

**Testing checklist:**
```python
test_queries = [
    # Find similar implementations
    "Show me examples of JWT authentication",
    "How do we handle database connections?",
    
    # API usage
    "Examples using the User model",
    "How to call the payment service",
    
    # Patterns
    "Error handling patterns in our codebase",
    "Examples of async background tasks",
    
    # Debugging
    "Where do we log errors?",
    "How is rate limiting implemented?"
]
```

### Example 3: Multi-Domain RAG (Docs + Code + Tickets)

**Optimal Configuration:**
```yaml
Embedding Model: e5-large-v2 (multi-domain)
Dimensions: 1024
Vector Store: Elasticsearch (filtering needed)
Chunk Size: Varies by type (500w docs, function-level code)
Similarity: Cosine
Threshold: 3.5
Index Structure: Content-type filtering
```

**Index schema:**
```python
{
    "content": "...",
    "embedding": [...],
    "content_type": "documentation" | "code" | "ticket" | "chat",
    "language": "python" | "typescript" | null,
    "repo": "frontend" | "backend" | null
}
```

**Query-time filtering:**
```python
# User asks code question
if is_code_query(query):
    filter = {"content_type": ["code"]}
    boost_vector = 1.5  # Prioritize semantic

# User asks "how do we..."
else:
    filter = {"content_type": ["documentation", "ticket"]}
    boost_keyword = 1.5  # Prioritize exact processes
```

---

## Component Selection Decision Tree

### Start Here: What are you embedding?

```
┌─ Natural Language Docs ────────────────────────────┐
│  → Use: MPNet, text-embedding-004                  │
│  → Dims: 768                                       │
│  → Chunk: 500 words                                │
│  → Store: Elasticsearch (for hybrid)               │
└────────────────────────────────────────────────────┘

┌─ Source Code ──────────────────────────────────────┐
│  → Use: CodeBERT, StarCoder embeddings             │
│  → Dims: 768                                       │
│  → Chunk: Function/class level                     │
│  → Store: pgvector (simpler, cheaper)              │
└────────────────────────────────────────────────────┘

┌─ Mixed (Docs + Code) ──────────────────────────────┐
│  → Use: e5-large-v2 (multi-domain)                 │
│  → Dims: 1024                                      │
│  → Chunk: Adaptive by content type                 │
│  → Store: Elasticsearch (filtering + hybrid)       │
└────────────────────────────────────────────────────┘

┌─ Multilingual ─────────────────────────────────────┐
│  → Use: multilingual-e5-large                      │
│  → Dims: 1024                                      │
│  → Chunk: 500 words                                │
│  → Store: Either (no keyword search in non-English)│
└────────────────────────────────────────────────────┘
```

### Budget Constraints?

```
┌─ Free Tier / Low Cost ────────────────────────────┐
│  → Model: all-MiniLM-L6-v2 (local)                │
│  → Store: pgvector on free PostgreSQL              │
│  → Cost: ~$0/month                                 │
│  → Trade-off: Lower quality (80% vs 90%)           │
└────────────────────────────────────────────────────┘

┌─ Mid-Budget (~$50/month) ─────────────────────────┐
│  → Model: Vertex AI text-embedding-004            │
│  → Store: Elasticsearch on e2-medium GCE           │
│  → Cost: ~$25 ES + ~$15 embeddings                 │
│  → Quality: 90%+ accuracy ✅ (current setup)       │
└────────────────────────────────────────────────────┘

┌─ Production / High Scale ─────────────────────────┐
│  → Model: OpenAI text-embedding-3-large           │
│  → Store: Elasticsearch cluster or Pinecone        │
│  → Cost: ~$200-500/month                           │
│  → Quality: 95%+ accuracy                          │
└────────────────────────────────────────────────────┘
```

---

## Practical Testing Scenarios

### Scenario 1: Testing Your Current System

```bash
# 1. Create test queries file
cat > test_queries.json << EOF
[
  {"query": "How do I install FastAPI?", "category": "getting_started"},
  {"query": "What is dependency injection in FastAPI?", "category": "advanced"},
  {"query": "How to add CORS middleware?", "category": "middleware"},
  {"query": "Async database connections", "category": "database"},
  {"query": "What is React?", "category": "out_of_domain"}
]
EOF

# 2. Run queries and collect scores
python << 'EOF'
import requests
import json

with open('test_queries.json') as f:
    queries = json.load(f)

for q in queries:
    resp = requests.post("http://localhost:8000/api/query",
                        json={"query": q["query"]}).json()
    
    docs = resp.get('retrieved_docs', [])
    score = docs[0]['score'] if docs else 0
    used_rag = score > 3.0
    
    print(f"{q['category']:20} Score: {score:5.2f}  RAG: {used_rag}")
EOF

# Expected output:
# getting_started      Score:  6.50  RAG: True  ✅
# advanced             Score:  5.20  RAG: True  ✅
# middleware           Score:  4.80  RAG: True  ✅
# database             Score:  3.50  RAG: True  ✅
# out_of_domain        Score:  1.20  RAG: False ✅
```

### Scenario 2: Testing Different Chunk Sizes

```python
# chunk_size_test.py

from rag_etl_pipeline import DocumentChunker, Document

# Sample document
doc = Document(
    doc_id="test",
    title="FastAPI Tutorial",
    content=open('sample_fastapi_doc.txt').read(),
    source_url="test",
    timestamp="now",
    content_hash="hash"
)

chunk_sizes = [200, 500, 1000, 2000]

for size in chunk_sizes:
    chunker = DocumentChunker(chunk_size=size, overlap=size//10)
    chunks = chunker.chunk_document(doc)
    
    print(f"\nChunk size {size}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c.split()) for c in chunks) / len(chunks):.0f} words")
    print(f"  Sample chunk preview: {chunks[0][:100]}...")
    
    # Test retrieval quality
    # (Would need to re-index and run queries for each)
```

### Scenario 3: Measuring Search Accuracy

```python
# search_accuracy.py - Calculate retrieval metrics

import requests
from typing import List, Dict

def calculate_metrics(test_queries: List[Dict]) -> Dict:
    """
    test_queries format:
    [
        {
            "query": "How to install FastAPI?",
            "relevant_urls": ["https://fastapi.../installation"]
        }
    ]
    """
    hits = 0
    reciprocal_ranks = []
    precisions_at_3 = []
    
    for test in test_queries:
        resp = requests.post("http://localhost:8000/api/query",
                           json={"query": test["query"], "top_k": 3}).json()
        
        retrieved = resp.get('retrieved_docs', [])
        retrieved_urls = [d['source_url'] for d in retrieved]
        
        # Hit rate: Did we get ANY relevant doc?
        if any(url in test["relevant_urls"] for url in retrieved_urls):
            hits += 1
        
        # MRR: Position of first relevant doc
        for i, url in enumerate(retrieved_urls, 1):
            if url in test["relevant_urls"]:
                reciprocal_ranks.append(1/i)
                break
        else:
            reciprocal_ranks.append(0)
        
        # Precision@3: How many of top-3 are relevant?
        relevant_in_top3 = sum(1 for url in retrieved_urls[:3] 
                              if url in test["relevant_urls"])
        precisions_at_3.append(relevant_in_top3 / 3)
    
    return {
        "hit_rate": hits / len(test_queries),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "precision_at_3": sum(precisions_at_3) / len(precisions_at_3)
    }

# Run it:
test_data = load_test_queries()
metrics = calculate_metrics(test_data)

print(f"Hit Rate: {metrics['hit_rate']:.2%}")      # Target: >80%
print(f"MRR: {metrics['mrr']:.3f}")                # Target: >0.7
print(f"Precision@3: {metrics['precision_at_3']:.2%}")  # Target: >60%
```

---

## Advanced Topics

### 1. Re-ranking

Add a re-ranker for better accuracy:

```python
from sentence_transformers import CrossEncoder

# After initial retrieval (fast, lower quality)
initial_results = hybrid_search(query, top_k=20)

# Re-rank with cross-encoder (slow, higher quality)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc.content] for doc in initial_results]
scores = reranker.predict(pairs)

# Return top-3 after re-ranking
final_results = sorted(zip(initial_results, scores), 
                      key=lambda x: x[1], reverse=True)[:3]
```

**Impact:**
- +10-15% accuracy improvement
- +50-100ms latency
- Worth it for production systems

### 2. Query Expansion

Improve recall by expanding queries:

```python
def expand_query(query: str, llm) -> List[str]:
    """Generate query variations"""
    prompt = f"Generate 3 variations of this query: '{query}'"
    variations = llm.generate(prompt).split('\n')
    return [query] + variations

# Search with all variations
results = []
for q_variant in expand_query("install FastAPI", llm):
    results.extend(search(q_variant, top_k=5))

# Deduplicate and re-rank
final = dedupe_and_rank(results, top_k=3)
```

### 3. Metadata Filtering

Filter by source, date, type:

```python
# Query: "Recent FastAPI authentication examples"
search_with_filters(
    query="FastAPI authentication",
    filters={
        "source_url": "*fastapi.tiangolo.com*",
        "timestamp": {"gte": "2024-01-01"},
        "content_type": "documentation"
    }
)
```

---

## Quick Start: Running Your Own Tests

### 1. Set up test environment:

```bash
# Source local env
source env.local.template

# Start services
./start_local_dev.sh

# Verify
curl http://localhost:8000/
curl http://localhost:9200/_cluster/health
```

### 2. Add test data via Admin UI:

1. Go to http://localhost:8501 → Admin
2. Add sources:
   - FastAPI docs: `https://fastapi.tiangolo.com/*` (depth 2, 20 pages)
   - Python docs: `https://docs.python.org/3/tutorial/*` (depth 2, 30 pages)
3. Trigger ingestion
4. Wait for completion

### 3. Run test queries:

```bash
# Create test script
cat > test_rag.sh << 'EOF'
#!/bin/bash
queries=(
    "How to install FastAPI?"
    "What is dependency injection?"
    "How to use async in Python?"
    "What is Java?"  # Out of domain
)

for q in "${queries[@]}"; do
    echo "Q: $q"
    curl -s -X POST http://localhost:8000/api/query \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"$q\"}" | \
      jq '{score: .retrieved_docs[0].score, answer: .answer[:100]}'
    echo "---"
done
EOF

chmod +x test_rag.sh
./test_rag.sh
```

### 4. Experiment with configurations:

```bash
# Test 1: Try different embedding model
# Edit rag_embeddings.py → change local_minilm to all-mpnet-base-v2
# Rebuild, re-index, re-test

# Test 2: Try different chunk size
# Edit rag_api.py startup_event → change chunk_size=500 to 1000
# Re-index, re-test

# Test 3: Try different threshold
# Edit rag_service.py → change RELEVANCE_THRESHOLD = 3.0 to 5.0
# Restart API, re-test
```

---

## Summary: Recommended Approach

### Phase 1: Baseline (Current)
- ✅ Model: all-MiniLM-L6-v2 (local) or text-embedding-004 (GCP)
- ✅ Dims: 768 (or 384 for local)
- ✅ Store: Elasticsearch
- ✅ Chunks: 500 words, 50 overlap
- ✅ Threshold: 3.0

### Phase 2: Measure
1. Create 20-30 test queries with expected results
2. Run evaluation, collect metrics
3. Identify failure patterns (what queries fail?)

### Phase 3: Optimize
Based on failures:
- **Low scores on specific queries** → Try better embedding model
- **Wrong docs retrieved** → Adjust chunking or add metadata filtering
- **Slow searches** → Optimize HNSW params or use fewer dims
- **High false positives** → Increase threshold
- **High false negatives** → Decrease threshold or improve chunking

### Phase 4: Production
- Monitor query logs
- Track user feedback (thumbs up/down)
- Continuously update test set
- Re-evaluate monthly

---

## Additional Resources

### Papers & Research
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Standard retrieval evaluation
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding model rankings
- "Precise Zero-Shot Dense Retrieval" (Hypothetical Document Embeddings)
- "Lost in the Middle" - How context position affects LLM accuracy

### Tools
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [ranx](https://github.com/AmenRa/ranx) - Ranking evaluation
- [ragas](https://github.com/explodinggradients/ragas) - RAG evaluation framework

### Our Evaluation Module
```python
from rag_evaluation import FeedbackStore, EvaluationMetrics

# Track user feedback
feedback_store.log_feedback(query_id, FeedbackType.THUMBS_UP)

# Get metrics
metrics = feedback_store.get_metrics(days=7)
print(f"Avg score: {metrics.avg_rating}")
print(f"Thumbs up rate: {metrics.positive_feedback_rate:.1%}")
```

---

## Questions to Guide Your Testing

1. **What are users really asking?**
   - Installation questions? → Need good intro doc coverage
   - Debugging questions? → Need error message indexing
   - "How do we..." questions? → Need code examples

2. **What's more important: Precision or Recall?**
   - High precision: Better to say "I don't know" than give wrong answer
   - High recall: Better to return more results, let LLM filter

3. **What's your latency budget?**
   - <100ms: Use small models (384 dims), HNSW, limit re-ranking
   - <500ms: Current setup works well
   - <2s: Can use large models, re-ranking, query expansion

4. **How often does content change?**
   - Daily: Need fast re-indexing, incremental updates
   - Weekly: Current setup is fine
   - Never: Can use heavier preprocessing

---

## Next Steps

1. **Collect real usage data** (already tracking via feedback)
2. **Build test dataset** from actual user queries
3. **Run systematic comparisons** using frameworks above
4. **Iterate on configuration** based on data, not hunches

The key is **measure, don't guess!** 📊

