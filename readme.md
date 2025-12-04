# Technical Documentation: Hybrid Information Retrieval System

## Table of Contents
1. [System Overview](#system-overview)
2. [BM25 Algorithm](#bm25-algorithm)
3. [Neural Reranking with Transformers](#neural-reranking-with-transformers)
4. [Hybrid Scoring](#hybrid-scoring)
5. [Caching Strategy](#caching-strategy)
6. [Architecture and Pipeline](#architecture-and-pipeline)

---

## System Overview

This hybrid information retrieval system combines classical probabilistic ranking (BM25) with modern neural semantic search (Sentence Transformers). The architecture follows a two-stage retrieval-reranking paradigm:

1. **First Stage (Retrieval)**: BM25 rapidly retrieves top K candidate documents from the corpus
2. **Second Stage (Reranking)**: Neural model reranks candidates using semantic similarity
3. **Caching Layer**: LRU cache stores results for frequently repeated queries

This design balances speed, accuracy, and computational efficiency.

---

## BM25 Algorithm

### Mathematical Foundation

BM25 (Best Matching 25) is a probabilistic ranking function based on the bag-of-words model. It estimates the relevance of a document D to a query Q.

#### Core Formula

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
              qi∈Q
```

Where:
- `qi`: Each query term
- `f(qi, D)`: Term frequency of qi in document D
- `|D|`: Length of document D (in words)
- `avgdl`: Average document length in the corpus
- `k1`: Term frequency saturation parameter (default: 1.5)
- `b`: Length normalization parameter (default: 0.75)
- `IDF(qi)`: Inverse document frequency of term qi

#### Inverse Document Frequency (IDF)

```
IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
```

Where:
- `N`: Total number of documents in corpus
- `n(qi)`: Number of documents containing term qi

#### Term Frequency Component

The term frequency component has a saturation effect controlled by k1:

```
TF_component = (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

**Key Properties:**
- As f(qi, D) increases, the score increases but with diminishing returns (saturation)
- Parameter k1 controls how quickly saturation occurs
- Longer documents are penalized through length normalization (b parameter)

#### Parameter Tuning

**k1 (Term Frequency Saturation)**
- Range: [1.2, 2.0]
- Lower values: Faster saturation, less weight on term frequency
- Higher values: Slower saturation, more weight on repeated terms
- Default: 1.5

**b (Length Normalization)**
- Range: [0, 1]
- b = 0: No length normalization
- b = 1: Full length normalization
- Default: 0.75
- Higher values penalize longer documents more heavily

#### Implementation Details

Our implementation uses the rank-bm25 library with the Okapi BM25 variant. The corpus is preprocessed through:

1. **Tokenization**: Breaking text into individual words
2. **Lowercasing**: Normalizing case
3. **Stopword Removal**: Removing common words (optional)
4. **Lemmatization**: Converting words to base form using spaCy (optional)

The index structure stores:
- Tokenized corpus (list of tokenized documents)
- Document statistics (lengths, term frequencies)
- Inverted index (term -> document mappings)

---

## Neural Reranking with Transformers

### Transformer Architecture

We use Sentence Transformers, specifically the `all-MiniLM-L6-v2` model, which is a distilled version of BERT optimized for semantic similarity tasks.

#### Model Architecture

**Base Model: MiniLM-L6**
- 6 transformer layers (vs 12 in BERT-base)
- 384-dimensional embeddings (vs 768 in BERT-base)
- 22.7M parameters (vs 110M in BERT-base)
- Mean pooling over token embeddings

#### Embedding Generation

For a text sequence T = [t1, t2, ..., tn]:

1. **Tokenization**: Convert text to WordPiece tokens
2. **Token Embeddings**: Map tokens to embedding vectors
3. **Position Embeddings**: Add positional information
4. **Transformer Layers**: Apply self-attention and feed-forward layers
5. **Pooling**: Mean-pool token embeddings to get sentence embedding

```
embedding(T) = mean([h1, h2, ..., hn])
```

Where hi is the hidden state for token i from the final transformer layer.

#### Self-Attention Mechanism

The core of transformers is the scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √dk) V
```

Where:
- Q (Query): Current token representation
- K (Key): All token representations
- V (Value): All token representations
- dk: Dimension of key vectors (for scaling)

**Multi-Head Attention** runs multiple attention operations in parallel:

```
MultiHead(Q, K, V) = Concat(head1, ..., headh)W^O

where headi = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

This allows the model to attend to different aspects of the input simultaneously.

#### Semantic Similarity Computation

After generating embeddings for query Q and document D:

```
similarity(Q, D) = cosine_similarity(emb(Q), emb(D))
                 = (emb(Q) · emb(D)) / (||emb(Q)|| × ||emb(D)||)
```

Cosine similarity measures the angle between embedding vectors, ranging from -1 to 1, where:
- 1: Identical semantic meaning
- 0: Orthogonal (unrelated)
- -1: Opposite meaning

#### Why This Model?

**all-MiniLM-L6-v2 Advantages:**
1. **Speed**: 5x faster than BERT-base due to fewer layers and smaller embeddings
2. **Quality**: Trained on 1B+ sentence pairs for semantic similarity
3. **Efficiency**: 384-dim embeddings balance expressiveness and computational cost
4. **Pre-trained**: Fine-tuned on diverse text for general-purpose semantic search

---

## Hybrid Scoring

### Motivation

BM25 and neural models have complementary strengths:

**BM25 Strengths:**
- Fast (no neural inference needed)
- Exact keyword matching
- Works well for technical/specific terms
- Transparent and interpretable

**Neural Strengths:**
- Captures semantic similarity
- Handles paraphrasing and synonyms
- Context-aware understanding
- Better for conceptual queries

**BM25 Weaknesses:**
- Vocabulary mismatch problem
- No understanding of synonyms
- Bag-of-words ignores word order

**Neural Weaknesses:**
- Computationally expensive
- May miss exact keyword matches
- Can be overconfident on irrelevant but semantically similar text

### Hybrid Combination Strategy

#### Two-Stage Pipeline

**Stage 1: BM25 Retrieval**
```
candidates = top_k(BM25_scores, k=100)
```

Rapidly retrieves top 100 candidates from entire corpus. This reduces the search space for the expensive neural model from thousands to hundreds.

**Stage 2: Neural Reranking**
```
For each candidate c in candidates:
    neural_score(c) = cosine_similarity(query_embedding, doc_embedding(c))
```

Only the top 100 candidates undergo neural scoring, making it computationally feasible.

#### Score Normalization

Before combining scores, we normalize to [0, 1]:

```
BM25_normalized = BM25_score / max(BM25_scores)
Neural_normalized = (neural_score + 1) / 2  # cosine ∈ [-1, 1] → [0, 1]
```

#### Hybrid Score Formula

```
hybrid_score = α × BM25_normalized + (1 - α) × Neural_normalized
```

Where α ∈ [0, 1] is the interpolation weight:
- α = 0: Pure neural ranking
- α = 0.5: Equal weighting (default)
- α = 1: Pure BM25 ranking

#### Choosing Alpha

The optimal α depends on the dataset and query types:

**Higher α (favor BM25)** when:
- Queries contain specific technical terms
- Exact keyword matching is important
- Dataset has controlled vocabulary

**Lower α (favor Neural)** when:
- Queries are conceptual or vague
- Synonym and paraphrase handling matters
- Dataset has natural language text

Our default α = 0.5 balances both approaches.

#### Final Ranking

```
results = top_k(hybrid_scores, k=10)
```

Return top 10 documents by hybrid score as final results.

---

## Caching Strategy

### LRU Cache Implementation

We implement a Least Recently Used (LRU) cache to store query results.

#### Data Structure

The cache uses Python's OrderedDict:
```
cache = OrderedDict()
{
    "query_text_rerank_true_alpha_0.5": {
        "results": [...],
        "query_time": 0.123,
        ...
    }
}
```

**Cache Key Format:**
```
key = f"{query}_{use_reranking}_{hybrid_alpha}"
```

This ensures different configurations are cached separately.

#### Cache Operations

**Get Operation - O(1)**
```python
def get(key):
    if key in cache:
        cache.move_to_end(key)  # Mark as recently used
        hits += 1
        return cache[key]
    else:
        misses += 1
        return None
```

**Put Operation - O(1)**
```python
def put(key, value):
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    if len(cache) > capacity:
        cache.popitem(last=False)  # Remove oldest
```

#### Why LRU?

**Access Patterns in Search:**
- Temporal locality: Recently searched queries likely to be searched again
- Popular queries: Some queries are frequently repeated (trends, news)
- Session behavior: Users often refine queries iteratively

**LRU Advantages:**
1. Simple and efficient: O(1) operations
2. Automatically evicts stale entries
3. Keeps frequently accessed items
4. Bounded memory usage

### Performance Impact

#### Without Cache

For a typical query:
1. BM25 retrieval: ~50ms
2. Neural encoding: ~100ms (query)
3. Neural encoding: ~150ms (100 candidates)
4. Similarity computation: ~10ms
5. **Total: ~310ms**

#### With Cache (Hit)

1. Cache lookup: <1ms
2. **Total: <1ms**

**Speedup: ~300x for cached queries**

#### Cache Hit Rate Analysis

Empirically, in production systems:
- First-time queries: 0% hit rate
- Within session: 30-50% hit rate
- Popular queries: 70-90% hit rate
- Overall: 20-40% hit rate typical

**Expected Savings:**
- 30% hit rate: ~93ms average query time (70% × 310ms + 30% × 1ms)
- 40% hit rate: ~186ms average query time

#### Memory Considerations

**Storage per entry:**
- Query string: ~100 bytes
- Results (10 docs): ~5KB
- Metadata: ~100 bytes
- **Total: ~5.2KB per entry**

**Cache capacity: 100 entries**
- Total memory: ~520KB
- Negligible compared to model size (~90MB for MiniLM)

#### Cache Invalidation

The cache is **never invalidated** in our system because:
1. The corpus is static (no real-time updates)
2. Query semantics don't change over time
3. Model and parameters are fixed post-deployment

For dynamic systems, cache invalidation strategies would include:
- Time-based expiration (TTL)
- Event-based invalidation (on corpus update)
- Manual cache clearing API

---

## Architecture and Pipeline

### System Components

```

                 User Query                        

                     │
                     ▼

                  LRU Cache Check                    
  - Check if query exists in cache                   
  - If hit: return cached results (1ms)

                    Cache Miss   

                     │ 
                     ▼

            Text Preprocessing                     
            Tokenization (NLTK)                              
            Lowercasing                                      
            Stopword removal (optional)                      
            Lemmatization with spaCy (optional)

                     │
                     ▼

           BM25 Retrieval                       
           Compute BM25 scores for all documents            
           Select top-k candidates (k=100)                  
            Time: ~50ms                                      

                     │
                     ▼

              Neural Reranking                        
      Encode query with Sentence Transformer          
      Encode top-k candidates                         
      Compute cosine similarities                     
      Normalize BM25 and neural scores                
      Compute hybrid scores                           
      Sort by hybrid score                            
      Time: ~250ms                                     

                     │
                     ▼

              Result Preparation                      
  - Select top-10 documents                          
  - Add metadata (scores, doc_id, timestamps)        
  - Store in cache                                   

                     │
                     ▼

               Return Results                        

```

### Index Structure

**Stored on Disk:**
```
data/processed/
├── bm25_index.pkl              # BM25 index (rank-bm25 object)
├── tokenized_corpus.pkl        # Preprocessed documents
├── documents.csv               # Original documents + metadata
└── preprocessor_config.json    # Preprocessing parameters
```

**Loaded in Memory:**
- BM25 index: ~50MB (for 2,692 documents)
- Neural model: ~90MB (MiniLM)
- Document DataFrame: ~20MB
- Total RAM: ~160MB

### Query Processing Flow

1. **Query arrives** (via API or CLI)
2. **Cache lookup**: O(1) dictionary lookup
3. **If cache miss**:
   - **Preprocess query**: O(n) where n = query length
   - **BM25 scoring**: O(N × m) where N = corpus size, m = query terms
   - **Sort and select**: O(N log k) where k = 100
   - **Neural encoding**: O(1) per document (fixed model cost)
   - **Similarity**: O(k × d) where k = 100, d = 384
   - **Hybrid combination**: O(k)
   - **Final sort**: O(k log 10)
4. **Store in cache**: O(1)
5. **Return results**

### Computational Complexity

**Space Complexity:**
- BM25 index: O(V × N) where V = vocabulary size, N = corpus size
- Neural embeddings: Not stored (computed on-demand)
- Cache: O(C × k) where C = cache capacity, k = results per query

**Time Complexity per Query:**
- BM25 retrieval: O(N × m) where m = query terms
- Neural encoding: O(L) where L = sequence length (constant)
- Similarity computation: O(k × d) where k = candidates, d = embedding dim
- Overall: O(N × m + k × d) ≈ O(N) dominated by BM25

### Optimization Strategies

1. **Two-stage pipeline**: Only run expensive neural model on top-k
2. **Caching**: Avoid recomputation for repeated queries
3. **Batch processing**: Encode multiple documents simultaneously
4. **Model distillation**: Use smaller model (MiniLM vs BERT)
5. **Index precomputation**: Build BM25 index once, reuse for all queries

---

## Performance Benchmarks

### Query Latency

**Measured on 2,692 document corpus:**

| Configuration | First Query | Cached Query | Improvement |
|--------------|-------------|--------------|-------------|
| BM25 only | 50ms | <1ms | 50x |
| Neural only | 300ms | <1ms | 300x |
| Hybrid | 310ms | <1ms | 310x |

### Accuracy Metrics

**Evaluation on test set (8 queries):**

| Method | MAP | MRR | P@10 |
|--------|-----|-----|------|
| BM25 | 0.8431 | 1.0000 | 0.4375 |
| Neural | 0.8253 | 1.0000 | 0.4375 |
| Hybrid | 0.8448 | 1.0000 | 0.4375 |

Where:
- MAP: Mean Average Precision
- MRR: Mean Reciprocal Rank
- P@10: Precision at 10 results

### Resource Usage

**Memory:**
- Idle: ~100MB
- Active (processing query): ~200MB
- Peak: ~250MB

**CPU:**
- BM25: Single-threaded, ~10% utilization
- Neural: Multi-threaded (PyTorch), ~60% utilization
- Average: ~20% utilization

**Disk:**
- Index files: ~70MB
- Model cache: ~90MB
- Logs: <1MB
- Total: ~160MB

---

## Conclusion

This hybrid system combines the speed of classical BM25 with the semantic understanding of modern transformers. The two-stage retrieval-reranking pipeline ensures efficiency, while the LRU cache provides excellent performance for repeated queries. The modular architecture allows easy tuning of parameters (k1, b, α) to optimize for specific use cases and datasets.

**Key Takeaways:**
1. BM25 provides fast, keyword-based retrieval
2. Transformers add semantic understanding for reranking
3. Hybrid scoring balances precision and recall
4. Caching dramatically reduces latency for common queries
5. The system scales well to moderate-sized corpora (thousands of documents)

For larger corpora (millions of documents), consider adding:
- Approximate nearest neighbor search (FAISS, Annoy)
- Distributed computing (Elasticsearch, Apache Lucene)
- GPU acceleration for neural components
- More sophisticated caching (Redis, Memcached)
