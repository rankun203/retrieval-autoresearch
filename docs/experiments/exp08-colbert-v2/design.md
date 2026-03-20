# Experiment: exp08-colbert-v2

## Goal

Retry ColBERT late interaction retrieval on Robust04 using the PyLate library (Chaffin & Sourty, CIKM 2025) instead of the manual implementation that failed in exp06. The exp06 failure was caused by the projection head not being loaded correctly when using raw `AutoModel` -- PyLate handles ColBERT model loading, query augmentation ([MASK] padding), marker tokens, and MaxSim scoring correctly out of the box.

## Hypothesis

ColBERT's token-level MaxSim matching is more expressive than single-vector bi-encoders and should provide a strong retrieval signal for Robust04's newswire documents. The exp06 review confirmed this was "an implementation failure, not a methodology failure" and recommended retrying with a proper library. With correct ColBERT embeddings:

1. ColBERT standalone retrieval should achieve MAP@100 in the 0.20-0.26 range (competitive with BM25+Bo1 at 0.2504)
2. ColBERT + BM25 linear fusion should outperform either system alone (complementary signals)
3. The full pipeline (ColBERT+BM25 fusion + Qwen3-Reranker) should beat the current best (MAP@100=0.2827) by providing a higher-quality candidate pool than Qwen3-Embedding used in exp05

### Literature Context

**ColBERT (Khattab & Zaharia, SIGIR 2020)**: Introduces late interaction with MaxSim scoring. Each query token finds its maximum cosine similarity with any document token; scores are summed across query tokens. Uses 128-dim token embeddings projected from BERT via a learned linear layer. Query tokens are augmented with [MASK] padding to a fixed length (32 by default) to enable query-independent document encoding.

**ColBERTv2 (Santhanam et al., NAACL 2022)**: Improves ColBERT with denoised supervision (cross-encoder distillation on MS-MARCO), residual compression for storage efficiency, and the PLAID retrieval engine. The `colbert-ir/colbertv2.0` checkpoint achieves MRR@10=39.7 on MS-MARCO dev, with strong zero-shot transfer to out-of-domain collections.

**PyLate (Chaffin & Sourty, CIKM 2025)**: Library built on Sentence Transformers for training and retrieval with ColBERT models. Handles model loading (including projection head), query/document tokenization with proper marker tokens, [MASK] query augmentation, and integrates with FastPLAID for efficient PLAID indexing. Supports `colbert-ir/colbertv2.0` and other ColBERT-compatible models.

**PLAID (Santhanam et al., 2022)**: Engine for efficient ColBERT retrieval using centroid-based candidate generation and deferred interaction with residual-compressed token embeddings. Reduces storage by ~6-10x compared to full token embeddings while maintaining retrieval quality. PyLate uses FastPLAID (4-bit residual quantization by default).

**Token pruning (Zong & Piwowarski, SIGIR 2025)**: Shows ColBERT maintains near-lossless performance with only 30% of document tokens, confirming redundancy in token representations. We use doc_length=180 tokens to balance storage and quality.

**SPLATE (Formal et al., SIGIR 2024)**: Maps ColBERTv2 token embeddings to sparse vocabulary space, matching ColBERTv2's MRR@10=40.0 on MS-MARCO dev. Confirms the quality of ColBERTv2's token-level representations.

**exp06 post-mortem**: The manual implementation used `AutoModel.from_pretrained("colbert-ir/colbertv2.0")` which loaded `BertModel` and silently discarded the `linear.weight` projection head. Token embeddings were truncated to 128 dims from the raw 768-dim BERT hidden state, producing meaningless similarity scores. ColBERT reranking was actively harmful (MAP@100=0.1194 vs BM25 alone at 0.2504). PyLate avoids this by wrapping the model with a proper `ColBERT` class that includes the linear projection.

## Method

### Architecture

Use PyLate's `models.ColBERT` class to load `colbert-ir/colbertv2.0` with correct projection head, marker tokens, and query augmentation. Use PLAID indexing via `indexes.PLAID` and `retrieve.ColBERT` for efficient retrieval over 528K documents.

### Pipeline

1. **ColBERT encoding**: Encode all 528K corpus documents using PyLate's `model.encode(is_query=False)` with 180-token max length. Build PLAID index with 4-bit residual quantization.
2. **ColBERT retrieval**: Encode 249 queries with `model.encode(is_query=True)` (32-token query length with [MASK] padding). Retrieve top-1000 per query via PLAID index.
3. **BM25+Bo1 retrieval**: Standard first stage via PyTerrier (cached index from prior experiments).
4. **Linear fusion**: Min-max normalize both score sets per query, combine with alpha=0.3 (ColBERT weight).
5. **Qwen3-Reranker**: Free ColBERT model from GPU, load Qwen3-Reranker-0.6B, rerank top-1000 from best fusion with news-short instruction, ml768, batch64.

### Score Fusion

For each query, min-max normalize ColBERT and BM25 scores to [0, 1], then:

    fused_score = alpha * colbert_norm + (1 - alpha) * bm25_norm

With alpha=0.3 (30% ColBERT, 70% BM25), matching the best fusion weight from exp05.

### Memory Management

- ColBERTv2: ~440MB model (BERT-base + 128-dim projection)
- PLAID index: ~4-6GB with 4-bit quantization for 528K docs x 180 tokens x 128 dims
- After ColBERT retrieval: delete model and index, free GPU
- Qwen3-Reranker: ~1.2GB model, ~20GB peak with ml768 batch64
- Total peak VRAM: ~25-30GB (well within L40S 46GB)

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| ColBERT model | colbert-ir/colbertv2.0 | BERT-base + 128-dim linear projection |
| Document max tokens | 180 | Balance storage/quality for 528K docs |
| Query max tokens | 32 | Standard ColBERT query length with [MASK] padding |
| Token embedding dim | 128 | ColBERTv2 projection dimension |
| Encode batch size | 256 | Conservative for GPU memory |
| PLAID nbits | 4 | Residual quantization bits (default) |
| PLAID kmeans_niters | 4 | Default clustering iterations |
| Retrieval top-K | 1000 | Standard depth |
| BM25 k1/b | 0.9/0.4 | From exp01 |
| Bo1 fb_docs/fb_terms | 5/30 | From exp01 |
| BM25 top-K | 1000 | Standard depth |
| Linear fusion alpha | 0.3 | Best from exp05 (ColBERT weight) |
| Reranker model | Qwen/Qwen3-Reranker-0.6B | Proven from exp03b/exp05 |
| Reranker instruction | news-short | Best from exp04 |
| Reranker max_length | 768 | Best from exp03b |
| Reranker batch_size | 64 | From exp03b |
| Reranker depth | 1000 | Full reranking of fusion candidates |

## Runs

### Run 1: `colbert-retrieval`
- **Description**: ColBERTv2 retrieval via PLAID index over 528K docs, top-1000 per query
- **Parameter overrides**: None (uses defaults above)
- **Expected output**: `runs/colbert-retrieval.run`
- **Purpose**: Standalone ColBERT retrieval quality; compare against BM25+Bo1 and Qwen3-Embedding

### Run 2: `colbert-bm25-fusion`
- **Description**: Linear interpolation of ColBERT retrieval + BM25+Bo1 (alpha=0.3)
- **Parameter overrides**: alpha=0.3 (ColBERT weight), 0.7 (BM25 weight)
- **Expected output**: `runs/colbert-bm25-fusion.run`
- **Purpose**: Test complementarity of ColBERT and BM25 signals

### Run 3: `colbert-fusion-reranked`
- **Description**: Best fusion run reranked with Qwen3-Reranker-0.6B (news-short, ml768, depth 1000)
- **Parameter overrides**: Uses fusion from run 2 as candidates
- **Expected output**: `runs/colbert-fusion-reranked.run`
- **Purpose**: Full pipeline; target to beat exp05 best (MAP@100=0.2827)

## Expected Outcome

- **ColBERT retrieval standalone**: MAP@100 ~0.20-0.26. ColBERTv2 is strong zero-shot but Robust04 is out-of-domain newswire. Should be competitive with BM25+Bo1 (0.2504).
- **ColBERT+BM25 fusion**: MAP@100 ~0.27-0.30. Late interaction and bag-of-words are complementary signals; fusion should improve recall and precision over either alone.
- **Full pipeline with Qwen3-Reranker**: MAP@100 ~0.28-0.32. If ColBERT provides different relevant documents than Qwen3-Embedding (likely due to different matching mechanisms), the candidate pool may be better for reranking, beating exp05's 0.2827.

## Baseline Comparison

| System | MAP@100 | Source |
|--------|---------|--------|
| BM25+Bo1 | 0.2504 | exp01 |
| Qwen3-Embedding-0.6B dense | 0.2105 | exp05 |
| Linear-a03 fusion (BM25+Qwen3-Embed) | 0.2762 | exp05 |
| Linear-a03 fusion + Qwen3-Reranker | 0.2827 | exp05 (current best) |
| ColBERTv2 rerank BM25 (BROKEN) | 0.1194 | exp06 (discarded) |
| ColBERTv2 full retrieval (BROKEN) | 0.0809 | exp06 (discarded) |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, zero-shot ColBERTv2 only)
- [x] Training does NOT use Robust04 qrels (no training)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A, no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (all models used zero-shot)
