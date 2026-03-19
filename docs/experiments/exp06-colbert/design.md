# Experiment: exp06-colbert

## Goal

Evaluate ColBERT late interaction retrieval on Robust04. ColBERT uses per-token embeddings with MaxSim scoring, which is more expressive than single-vector bi-encoders while remaining more efficient than full cross-encoders. Test both ColBERT as a reranker (over BM25 candidates) and as a standalone retriever (full corpus encoding with brute-force MaxSim), then combine with BM25 fusion and Qwen3-Reranker.

## Hypothesis

ColBERT's token-level matching captures fine-grained interactions that single-vector models miss. On Robust04's newswire documents with short title queries, exact term matching (captured by MaxSim over individual tokens) should complement BM25's bag-of-words approach. ColBERTv2 was trained on MS-MARCO and has strong zero-shot generalization. We expect:

1. ColBERT reranking of BM25 top-1000 to outperform single-vector dense retrieval (Qwen3-Embedding MAP@100=0.2105)
2. ColBERT + BM25 linear fusion to improve recall over either system alone
3. The full pipeline (ColBERT+BM25 fusion + Qwen3-Reranker) to potentially beat the current best (MAP@100=0.2827)

### Literature Context

**ColBERT (Khattab & Zaharia, 2020)**: Introduces late interaction with MaxSim -- each query token finds its maximum cosine similarity with any document token, scores are summed. Trained on MS-MARCO with contrastive loss. Uses 128-dim token embeddings projected from BERT.

**ColBERTv2 (Santhanam et al., 2022)**: Improves ColBERT with residual compression, denoised supervision (cross-encoder distillation), and the PLAID engine for efficient retrieval. The `colbert-ir/colbertv2.0` checkpoint is the standard pre-trained model.

**SPLATE (formal2024splate)**: Maps ColBERTv2 token embeddings to sparse vocabulary space, matching ColBERTv2's MRR@10=40.0 on MS-MARCO dev. Confirms ColBERTv2 embeddings contain rich token-level information.

**Token pruning (sigir2025tokenpruning)**: Shows ColBERT can maintain near-lossless performance with only 30% of document tokens, suggesting redundancy in token representations. We use doc_maxlen=180 to keep memory manageable.

**Survey Section 3**: Late interaction methods "preserve fine-grained token-level matching signals" and "sit between bi-encoders and cross-encoders in the expressiveness-efficiency tradeoff."

## Method

### Architecture

Manual ColBERT implementation using transformers + torch, avoiding heavy library dependencies. The ColBERTv2 model (`colbert-ir/colbertv2.0`) is a BERT-base model with a 128-dim linear projection head.

### Pipeline

1. **BM25+Bo1 retrieval**: Standard first stage, top-1000 per query (from prior experiments)
2. **ColBERT encoding**: Encode all 528K corpus documents with per-token embeddings (128-dim), stored as memory-mapped numpy arrays on disk
3. **ColBERT reranking**: Rerank BM25 top-1000 using ColBERT MaxSim scores (fast, only 249K pairs)
4. **ColBERT full retrieval**: For each query, compute MaxSim against all document token embeddings using batched GPU operations
5. **Fusion**: Linear interpolation of ColBERT retrieval scores with BM25+Bo1 scores (alpha=0.3)
6. **Qwen3-Reranker**: Rerank best fusion output with Qwen3-Reranker-0.6B (news-short instruction, ml768, depth 1000)

### ColBERT MaxSim Scoring

For a query Q with token embeddings {q_1, ..., q_m} and document D with token embeddings {d_1, ..., d_n}:

    score(Q, D) = sum_i max_j cos_sim(q_i, d_j)

All embeddings are L2-normalized, so cosine similarity = dot product.

### Memory Management

- ColBERTv2 model: ~440MB (BERT-base + projection)
- Per-doc token embeddings: 528K docs x 180 tokens x 128 dims x 2 bytes (fp16) = ~24GB on disk
- We use memory-mapped arrays and process documents in chunks during retrieval
- For full retrieval: load chunks of doc embeddings into GPU, compute MaxSim batch-wise
- After ColBERT encoding/retrieval, free GPU memory before loading Qwen3-Reranker

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| ColBERT model | colbert-ir/colbertv2.0 | BERT-base + 128-dim projection |
| Doc max tokens | 180 | Balance storage/quality for 528K docs |
| Query max tokens | 32 | Robust04 title queries are short |
| Token embedding dim | 128 | ColBERTv2 projection dimension |
| Encode batch size | 256 | Fits easily on L40S |
| Retrieval chunk size | 50000 | Process 50K docs at a time for MaxSim |
| BM25 k1/b | 0.9/0.4 | From exp01 |
| Bo1 fb_docs/fb_terms | 5/30 | From exp01 |
| BM25 top-K | 1000 | Standard depth |
| Linear fusion alpha | 0.3 | Best from exp05 (ColBERT weight) |
| Reranker model | Qwen/Qwen3-Reranker-0.6B | From exp03b/exp05 |
| Reranker instruction | news-short | Best from exp04 |
| Reranker max_length | 768 | Best from exp03b |
| Reranker batch_size | 64 | From exp03b |
| Reranker depth | 1000 | Full reranking of candidates |

## Runs

### Run 1: `colbert-rerank-bm25`
- **Description**: ColBERT MaxSim reranking of BM25+Bo1 top-1000 candidates
- **Parameter overrides**: Only scores BM25 candidates (no full retrieval needed)
- **Expected output**: `runs/colbert-rerank-bm25.run`
- **Purpose**: Fast test of ColBERT scoring quality; compare against Qwen3-Reranker

### Run 2: `colbert-fullretrieval`
- **Description**: Full ColBERT retrieval over 528K docs, brute-force MaxSim
- **Parameter overrides**: Scores all docs for all queries
- **Expected output**: `runs/colbert-fullretrieval.run`
- **Purpose**: ColBERT as standalone first-stage retriever

### Run 3: `colbert-bm25-fusion`
- **Description**: Linear interpolation of ColBERT full retrieval + BM25+Bo1 (alpha=0.3)
- **Parameter overrides**: alpha=0.3 (ColBERT weight), 0.7 (BM25 weight)
- **Expected output**: `runs/colbert-bm25-fusion.run`
- **Purpose**: Test complementarity of ColBERT and BM25

### Run 4: `colbert-fusion-reranked`
- **Description**: Best fusion run reranked with Qwen3-Reranker-0.6B (news-short, ml768, depth 1000)
- **Parameter overrides**: Uses best fusion from run 3
- **Expected output**: `runs/colbert-fusion-reranked.run`
- **Purpose**: Full pipeline, compare against exp05 best (0.2827)

## Expected Outcome

- ColBERT reranking of BM25: MAP@100 ~0.24-0.27 (ColBERT is expressive but not as strong as a 0.6B cross-encoder)
- ColBERT full retrieval: MAP@100 ~0.22-0.26 (strong zero-shot, between BM25 and Qwen3-Embedding)
- ColBERT+BM25 fusion: MAP@100 ~0.27-0.30 (complementary signals)
- Full pipeline with Qwen3-Reranker: MAP@100 ~0.28-0.32 (may beat exp05's 0.2827 if ColBERT provides better recall than Qwen3-Embedding)

## Baseline Comparison

- BM25+Bo1: MAP@100 = 0.2504
- Qwen3-Embedding-0.6B dense: MAP@100 = 0.2105
- Linear-a03 fusion (BM25+Qwen3-Embed): MAP@100 = 0.2762
- Linear-a03 fusion + Qwen3-Reranker: MAP@100 = 0.2827 (current best)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, zero-shot ColBERTv2 only)
- [x] Training does NOT use Robust04 qrels (no training)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A, no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (all models used zero-shot)
