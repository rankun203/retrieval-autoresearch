# Experiment: exp05-hybrid-rrf

## Goal

Combine BM25+Bo1 sparse retrieval with Qwen3-Embedding-0.6B dense retrieval via Reciprocal Rank Fusion (RRF) and linear interpolation, then optionally rerank with Qwen3-Reranker-0.6B. This tests whether hybrid retrieval can surpass either individual system and whether fusion + reranking achieves a new best.

## Hypothesis

BM25 and dense retrieval capture complementary relevance signals: BM25 excels at exact term matching while dense models capture semantic similarity. RRF (Cormack et al., 2009) is a simple, parameter-light fusion method that combines ranked lists without requiring score normalization. Qwen3-Embedding-0.6B is a strong zero-shot dense encoder (MTEB-en retrieval 61.83), and combining it with BM25+Bo1 should yield higher recall than either alone, which in turn provides a better candidate pool for Qwen3-Reranker-0.6B.

### Literature Context

**RRF (Cormack et al., 2009)**: Reciprocal Rank Fusion assigns each document a fused score = sum(1 / (k + rank_i)) across systems. The k parameter (typically 60) dampens the influence of high ranks. RRF is robust, requires no score normalization, and consistently performs well across diverse system combinations.

**Hybrid dense+sparse retrieval**: The survey (Section 6) notes that "hybrid methods consistently outperform either dense or sparse retrieval alone on standard benchmarks." BGE-M3 (chen2024bgem3) demonstrates dense+sparse+multi-vector fusion within a single model, achieving strong results via self-knowledge distillation.

**Qwen3-Embedding-0.6B**: 0.6B parameter decoder-only embedding model with instruction support. Uses last-token pooling with EOS token. Supports task-specific instructions for queries. 32K context window, 1024-dim embeddings.

**RRF k parameter**: k=60 is the standard from the original paper. Lower k (e.g., 10) gives more weight to top-ranked documents; higher k (e.g., 100) makes the fusion more uniform across ranks. The optimal k depends on the quality/diversity of the component systems.

## Method

1. **BM25+Bo1 retrieval**: Standard BM25 with Bo1 PRF, retrieve top-1000 per query (existing baseline)
2. **Qwen3-Embedding-0.6B dense retrieval**: Zero-shot encoding of all 528K corpus documents and 249 queries, FAISS flat inner product search, retrieve top-1000
3. **RRF fusion**: Combine BM25+Bo1 and dense ranked lists using RRF with k=60
4. **Linear interpolation**: Alternative fusion via score normalization (min-max per query) and alpha-weighted combination
5. **Reranking**: Best fusion run reranked with Qwen3-Reranker-0.6B (ml768, top-1000)

### Qwen3-Embedding-0.6B Usage

Based on the HuggingFace model card:
- Query prefix: `Instruct: {task_instruction}\nQuery: {query_text}`
- Document prefix: none (just raw text)
- Pooling: last token (EOS)
- Normalize embeddings to unit length
- Task instruction for retrieval: `Given a web search query, retrieve relevant passages that answer the query`

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embedding model | Qwen/Qwen3-Embedding-0.6B | 0.6B params, 1024-dim |
| Reranker model | Qwen/Qwen3-Reranker-0.6B | Same as exp03b |
| Doc max length | 512 tokens | Balance quality/speed for 528K docs |
| Query max length | 512 tokens | Generous for Robust04 title queries |
| Encode batch size | 256 | ~5GB VRAM with 0.6B model |
| BM25 k1/b | 0.9/0.4 | From exp01 |
| Bo1 fb_docs/fb_terms | 5/30 | From exp01 |
| BM25 top-K | 1000 | Standard depth |
| Dense top-K | 1000 | Standard depth |
| RRF k values | 10, 60, 100 | Sweep around standard k=60 |
| Linear alpha values | 0.3, 0.5, 0.7 | alpha * dense + (1-alpha) * sparse |
| Reranker max_length | 768 | Best from exp03b |
| Reranker batch_size | 64 | From exp03b |
| Reranker depth | 1000 | Best from exp03b |

## Runs

### Run 1: `dense-qwen3-zeroshot`
- **Description**: Zero-shot Qwen3-Embedding-0.6B dense retrieval, top-1000
- **Parameter overrides**: None (uses defaults above)
- **Expected output**: `runs/dense-qwen3-zeroshot.run`
- **Purpose**: Establishes dense retrieval baseline with strong backbone

### Run 2: `rrf-k10`, `rrf-k60`, `rrf-k100`
- **Description**: RRF fusion of BM25+Bo1 and dense retrieval with different k values
- **Parameter overrides**: k=10, k=60, k=100 respectively
- **Expected output**: `runs/rrf-k{k}.run`
- **Purpose**: Find optimal RRF k parameter

### Run 3: `linear-a03`, `linear-a05`, `linear-a07`
- **Description**: Linear interpolation with alpha sweep
- **Parameter overrides**: alpha=0.3, 0.5, 0.7
- **Expected output**: `runs/linear-a{alpha}.run`
- **Purpose**: Compare linear interpolation against RRF

### Run 4: `best-fusion-reranked`
- **Description**: Best fusion run reranked with Qwen3-Reranker-0.6B (ml768, top-1000)
- **Parameter overrides**: Uses best fusion from runs 2-3
- **Expected output**: `runs/best-fusion-reranked.run`
- **Purpose**: Test if better first-stage recall translates to better reranking

## Expected Outcome

- Dense-only (Qwen3-Embedding-0.6B): MAP@100 ~0.20-0.24 (competitive with BM25+Bo1's 0.2504)
- RRF fusion: MAP@100 ~0.27-0.30 (hybrid typically improves 10-20% over best individual)
- Best fusion + reranker: MAP@100 ~0.29-0.33 (better first-stage recall should help reranker)

## Baseline Comparison

- BM25+Bo1: MAP@100 = 0.2504, nDCG@10 = 0.4662
- BM25+Bo1 >> Qwen3-Reranker ml768 top-1000: MAP@100 = 0.2668, nDCG@10 = 0.5288 (current best)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, zero-shot only)
- [x] Training does NOT use Robust04 qrels (no training)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A, no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (all models used zero-shot)
