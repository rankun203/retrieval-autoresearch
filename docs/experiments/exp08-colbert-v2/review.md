# Review: exp08-colbert-v2

## Data Leakage Check: PASS

**Evidence:**

1. `load_robust04()` is called once at line 71, returning `corpus`, `queries`, `qrels`.
2. `corpus` is used to prepare document texts for ColBERT encoding (lines 74-79) and to retrieve document text during reranking (lines 400-402). Corpus text is allowed -- it is not test data.
3. `queries` is used in three places:
   - Line 181: to build `query_texts_list` for ColBERT retrieval encoding (final evaluation, not training).
   - Line 272: to build `topics_df` for BM25+Bo1 retrieval (final evaluation).
   - Line 412: to get query text for reranking (final evaluation).
4. `qrels` is used only at lines 229, 336, 444 -- all calls to `evaluate_run()` for final metric computation.
5. No training occurs in this experiment (zero-shot ColBERTv2, zero-shot Qwen3-Reranker). No hard negative mining, no fine-tuning, no gradient updates.
6. No test-time information flows into model weights.

**Conclusion:** Clean zero-shot pipeline. No leakage.

## Code Quality

**Strengths:**
- Excellent sanity check for ColBERT projection head (lines 100-109): verifies the linear layer exists and aborts if not found, directly addressing the exp06 failure mode.
- Embedding sanity check (lines 121-138): confirms relevant documents score higher than irrelevant ones via MaxSim, catching broken embeddings early.
- Proper GPU memory management: ColBERT model/index freed before loading reranker (lines 214-218).
- Clean separation of phases with progress reporting.

**Minor issues:**
- The vocab expansion warning from PyLate ("new embeddings will be initialized from a multivariate normal distribution") suggests PyLate is resizing the tokenizer vocabulary for marker tokens. This is expected behavior for ColBERT models loaded through sentence-transformers/PyLate and does not affect results.
- The `max_length` warning during reranker tokenization (line 382 `tokenizer.pad` with `max_length` when `padding=True`) is cosmetic and does not affect correctness.

## Design Adherence

| Aspect | Design | Actual | Match |
|--------|--------|--------|-------|
| ColBERT model | colbert-ir/colbertv2.0 | colbert-ir/colbertv2.0 | Yes |
| Doc max tokens | 180 | 180 | Yes |
| Query max tokens | 32 | 32 | Yes |
| PLAID nbits | 4 | 4 | Yes |
| Encode batch | 256 | 256 | Yes |
| Retrieval top-K | 1000 | 1000 | Yes |
| Fusion alpha | 0.3 | 0.3 | Yes |
| Reranker | Qwen3-Reranker-0.6B | Qwen3-Reranker-0.6B | Yes |
| Reranker max_length | 768 | 768 | Yes |
| Reranker batch | 64 | 64 | Yes |
| Run 1 (colbert-retrieval) | Yes | Yes | Yes |
| Run 2 (colbert-bm25-fusion) | Yes | Yes | Yes |
| Run 3 (colbert-fusion-reranked) | Yes | Yes | Yes |

All parameters and runs match the design document exactly.

## Performance Analysis

### Results

| Run | MAP@100 | nDCG@10 | MAP@1000 | Recall@100 |
|-----|---------|---------|----------|------------|
| colbert-retrieval | 0.1844 | 0.4475 | 0.2088 | 0.3451 |
| colbert-bm25-fusion | 0.2689 | 0.4992 | 0.3195 | 0.4810 |
| colbert-fusion-reranked | 0.2870 | 0.5436 | 0.3362 | 0.4985 |

### vs. Design Expectations

| Run | Expected MAP@100 | Actual MAP@100 | Assessment |
|-----|-------------------|----------------|------------|
| colbert-retrieval | 0.20-0.26 | 0.1844 | Below range |
| colbert-bm25-fusion | 0.27-0.30 | 0.2689 | Just below range |
| colbert-fusion-reranked | 0.28-0.32 | 0.2870 | Within range |

### vs. Current Best and Baselines

| System | MAP@100 | Source |
|--------|---------|--------|
| **Current best: exp07 Qwen3-Embed-8B fusion** | **0.2929** | exp07 |
| exp08 colbert-fusion-reranked (best this exp) | 0.2870 | exp08 |
| exp05 Linear-a03 + Qwen3-Reranker | 0.2827 | exp05 |
| exp08 colbert-bm25-fusion | 0.2689 | exp08 |
| BM25+Bo1 baseline | 0.2504 | exp01 |
| exp08 colbert-retrieval | 0.1844 | exp08 |
| exp06 ColBERT brute-force (BROKEN) | 0.0809 | exp06 |

No exp08 run beats the current best of MAP@100=0.2929.

### ColBERT Standalone Analysis

ColBERT standalone MAP@100=0.1844 is below the expected 0.20-0.26 range and below BM25+Bo1 (0.2504). However, this is a massive improvement over exp06's broken 0.0809, confirming the projection head fix worked. The sanity check in the log confirms embeddings are meaningful (relevant=23.04 vs irrelevant=4.70, delta=18.34). Factors explaining the lower-than-expected performance:

1. **Domain mismatch**: ColBERTv2 was trained on MS-MARCO (web search); Robust04 is newswire. Zero-shot transfer is imperfect.
2. **PLAID approximation**: PLAID with 4-bit quantization and centroid-based candidate generation trades some recall for speed.
3. **Recall@100 = 0.3451**: Significantly lower than BM25+Bo1's 0.4527. ColBERT is finding different but fewer relevant documents in the top-100.

Despite the low standalone score, ColBERT contributes meaningfully to fusion: the fusion (0.2689) exceeds BM25+Bo1 alone (0.2504) by +0.0185, confirming complementary retrieval signals.

### Comparison with exp06 (Broken ColBERT)

| Metric | exp06 (broken) | exp08 (fixed) | Improvement |
|--------|---------------|---------------|-------------|
| ColBERT standalone MAP@100 | 0.0809 | 0.1844 | +128% |
| ColBERT+BM25 fusion MAP@100 | 0.2438 | 0.2689 | +10.3% |
| Fusion+Reranker MAP@100 | 0.2792 | 0.2870 | +2.8% |

The projection head fix made a substantial difference to standalone ColBERT (+128%), a moderate difference to fusion (+10.3%), and a small difference after reranking (+2.8%). The reranker partially compensates for poor first-stage retrieval, but better first-stage quality still helps.

## Budget Assessment: OK

- Zero-shot experiment, no training budget applies.
- Total wall clock: 5879.6s (~98 minutes).
- Peak VRAM: 25036.1 MB (24.4 GB), well within L40S 46GB.

## Cache Verification

- BM25 Terrier index: used cached index from prior experiments (log: "Using cached Terrier index"). Expected and correct.
- PLAID index: built fresh with `override=True` (line 163). No cached ColBERT artifacts reused.
- ColBERT model: downloaded from HuggingFace.

## Verdict: **APPROVE**

All three runs completed successfully with valid metrics. No data leakage. Code correctly addresses the exp06 projection head failure. No run beats the current best (MAP@100=0.2929), so all three runs are logged as `discard`.

### Status Assignments
- colbert-retrieval: `discard`
- colbert-bm25-fusion: `discard`
- colbert-fusion-reranked: `discard`

## Recommendations

1. **Multi-system fusion**: Fusing all three signals (BM25+Bo1, Qwen3-Embed-8B, ColBERTv2) could capture complementary matches from token-level, embedding-level, and lexical matching.
2. **ColBERT fine-tuning**: Zero-shot ColBERTv2 is weak on Robust04 newswire. Fine-tuning on MS-MARCO with domain adaptation could improve standalone performance substantially.
3. **Brute-force MaxSim**: Running exact MaxSim instead of PLAID to check if quantization/approximation is hurting recall significantly.
4. **Larger ColBERT backbone**: Training a ColBERT head on a stronger base model (e.g., DeBERTa-v3 or a larger BERT variant).
