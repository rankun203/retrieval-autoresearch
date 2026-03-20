# exp10-backbone-sweep: Dense Retrieval Backbone Sweep

## Goal

Evaluate remaining Priority 2 dense retrieval backbones in zero-shot mode with the proven hybrid fusion pipeline (BM25+Bo1 + dense linear fusion at alpha=0.3). Identify whether any smaller/different backbone can match or approach the Qwen3-Embedding-8B result (MAP@100=0.2929) at lower compute cost, or whether a different architecture offers complementary signal.

## Hypothesis

Different embedding models have different strengths on news-domain text (Robust04). Models trained on diverse retrieval data (BGE, e5-large) may close the gap with Qwen3-8B when combined with BM25+Bo1 fusion. ModernBERT's longer context and modern architecture may also help on longer news articles. The fusion pipeline is the key equalizer -- even weaker dense retrievers benefit substantially from BM25 combination.

## Method

For each backbone model:
1. Load model with correct prefixes, pooling, and normalization per model card
2. Encode all 528K Robust04 docs (cache embeddings to disk via `utils/build_cache_key.py`)
3. Encode 249 test queries
4. FAISS IndexFlatIP search, retrieve top-1000
5. Evaluate dense-only retrieval
6. Linear fusion with BM25+Bo1 (alpha=0.3 dense, 0.7 sparse)
7. Evaluate fused retrieval

No reranking -- exp07 showed reranking can hurt with strong embeddings.

### Model-specific details

**intfloat/e5-large-v2** (335M params, 1024-dim)
- Prefixes: "query: " for queries, "passage: " for documents
- Pooling: mean pooling over last hidden state (average_pool)
- Normalization: L2 normalize
- Max length: 512
- Encode batch: 256
- MTEB retrieval: ~49.5 nDCG@10

**BAAI/bge-base-en-v1.5** (110M params, 768-dim)
- Prefixes: "Represent this sentence for searching relevant passages: " for queries, none for documents
- Pooling: CLS token
- Normalization: L2 normalize
- Max length: 512
- Encode batch: 512
- MTEB retrieval: ~53.3 nDCG@10

**nomic-ai/modernbert-embed-base** (~150M params, 768-dim)
- Prefixes: "search_query: " for queries, "search_document: " for documents
- Pooling: mean pooling
- Normalization: L2 normalize
- Max length: 512 (supports up to 8192)
- Encode batch: 512
- Uses `trust_remote_code=True`

## Key Parameters

| Parameter         | e5-large-v2 | bge-base-en-v1.5 | modernbert-embed-base |
|-------------------|-------------|-------------------|-----------------------|
| Model size        | 335M        | 110M              | ~150M                 |
| Embedding dim     | 1024        | 768               | 768                   |
| Encode batch      | 256         | 512               | 512                   |
| Doc max length    | 512         | 512               | 512                   |
| Query max length  | 512         | 512               | 512                   |
| Query prefix      | "query: "   | long instruction  | "search_query: "      |
| Doc prefix        | "passage: " | none              | "search_document: "   |
| Pooling           | mean        | CLS               | mean                  |
| Normalize         | L2          | L2                | L2                    |

Shared parameters:
- BM25: k1=0.9, b=0.4, Bo1 fb_docs=5, fb_terms=30
- Fusion: linear alpha=0.3 (dense=0.3, sparse=0.7)
- Dense top-K: 1000
- BM25 top-K: 1000
- No reranking
- No training (zero-shot only)

## Runs

| Run name                       | Description                                              | Expected output               |
|--------------------------------|----------------------------------------------------------|-------------------------------|
| `dense-e5-large-v2`           | e5-large-v2 dense-only retrieval                         | `dense-e5-large-v2.run`      |
| `fusion-e5-large-v2`          | e5-large-v2 + BM25+Bo1 linear fusion alpha=0.3          | `fusion-e5-large-v2.run`     |
| `dense-bge-base-en-v1.5`      | bge-base-en-v1.5 dense-only retrieval                   | `dense-bge-base-en-v1.5.run` |
| `fusion-bge-base-en-v1.5`     | bge-base-en-v1.5 + BM25+Bo1 linear fusion alpha=0.3     | `fusion-bge-base-en-v1.5.run`|
| `dense-modernbert-embed-base`  | modernbert-embed-base dense-only retrieval              | `dense-modernbert-embed-base.run`|
| `fusion-modernbert-embed-base` | modernbert-embed-base + BM25+Bo1 linear fusion alpha=0.3| `fusion-modernbert-embed-base.run`|

6 runs total (3 models x 2 variants each). Jina-v4 skipped due to complexity (multimodal, 3.8B params, unusual API).

## Expected Outcome

| Model               | Dense MAP@100 (est) | Fusion MAP@100 (est) | Rationale                                     |
|----------------------|--------------------:|---------------------:|-----------------------------------------------|
| e5-large-v2          | 0.19-0.22          | 0.27-0.28            | Larger than e5-base (0.17), but still limited |
| bge-base-en-v1.5     | 0.17-0.20          | 0.26-0.28            | Similar tier to e5-base                       |
| modernbert-embed-base| 0.17-0.20          | 0.26-0.28            | Modern arch but small model                   |

None are expected to beat Qwen3-Embedding-8B fusion (0.2929) but the data will show how much backbone quality matters vs fusion doing the heavy lifting. If any model approaches 0.28+ in fusion, it would be a much cheaper alternative.

## Baseline Comparison

- exp02 dense baseline: e5-base-v2 zero-shot MAP@100 = 0.1697
- exp05 Qwen3-0.6B fusion: MAP@100 = 0.2762 (linear alpha=0.3)
- exp07 Qwen3-8B fusion: MAP@100 = 0.2929 (current best, linear alpha=0.3)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries
- [x] Training does NOT use Robust04 qrels (relevance judgments)
- [x] Hard negative mining uses MS-MARCO or documented train split only
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights

(This is a zero-shot experiment -- no training occurs. Models are used as-is.)
