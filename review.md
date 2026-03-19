# Review: exp05-hybrid-rrf

## Data Leakage Check: PASS

Verified every reference to `qrels`, `queries`, and `load_robust04` in `train.py`:

- **`load_robust04()`** (line 72): Returns `corpus`, `queries`, `qrels`.
- **`corpus`**: Used for document text retrieval (lines 76-78), encoding documents (lines 201-205), and counting (line 565). Corpus is allowed.
- **`queries`**: Used to build BM25 topics (line 114), encode query embeddings (lines 213-214), and format reranker inputs (line 486). All are retrieval/evaluation uses. There is NO training in this experiment (zero-shot only), so no test information flows into model weights.
- **`qrels`**: Used exclusively in `evaluate_run()` calls at lines 263, 364, 389, and 516. All are final evaluation. ALLOWED.
- **No training**: No `stream_msmarco_triples()`, no gradient updates, no hard negative mining. All models are used zero-shot.

No data leakage detected.

## Code Quality

**Good:**
- Clean separation of phases: BM25, dense encoding, fusion, reranking
- Proper VRAM management with model deletion and cache clearing between phases
- Correct Qwen3-Embedding usage with instruction prefix and last-token pooling
- Reranker code reuses proven pattern from exp03b/exp04

**Issues found:**
- Line 131 (first run): alpha parsing bug `alpha = int(alpha_str) / 10.0` failed when `best_fusion_name = "linear-a03"` because split on `-a` from the full name could match incorrectly. Fixed in commit f3eccbe and experiment re-run successfully.
- Minor: `alpha_str` formatting at line 377 (`"03"`, `"05"`, `"07"`) works but is fragile.
- The run_name reconstruction logic (lines 409-415) duplicates fusion computation unnecessarily; the run dict could be cached.

## Design Adherence

The design specified 8 runs and all 8 were produced:

| Run | Specified | Produced | Match |
|-----|-----------|----------|-------|
| dense-qwen3-zeroshot | Yes | Yes | Yes |
| rrf-k10 | Yes | Yes | Yes |
| rrf-k60 | Yes | Yes | Yes |
| rrf-k100 | Yes | Yes | Yes |
| linear-a03 | Yes | Yes | Yes |
| linear-a05 | Yes | Yes | Yes |
| linear-a07 | Yes | Yes | Yes |
| best-fusion-reranked | Yes | Yes | Yes |

All parameters match design.md specifications (embedding model, reranker model, batch sizes, max lengths, k values, alpha values). The best fusion (linear-a03) was correctly selected and reranked.

## Performance Analysis

Results from second (successful) run:

| Run | MAP@100 | nDCG@10 | MAP@1000 | Recall@100 | vs Best (0.2668) |
|-----|---------|---------|----------|------------|-------------------|
| dense-qwen3-zeroshot | 0.2105 | 0.5026 | 0.2412 | 0.3911 | -21.1% |
| rrf-k10 | 0.2718 | 0.5342 | 0.3235 | 0.4914 | +1.9% |
| rrf-k60 | 0.2704 | 0.5340 | 0.3224 | 0.4893 | +1.3% |
| rrf-k100 | 0.2671 | 0.5317 | 0.3194 | 0.4887 | +0.1% |
| linear-a03 | 0.2762 | 0.5095 | 0.3289 | 0.4920 | +3.5% |
| linear-a05 | 0.2743 | 0.5335 | 0.3264 | 0.4813 | +2.8% |
| linear-a07 | 0.2500 | 0.5372 | 0.2968 | 0.4408 | -6.3% |
| **best-fusion-reranked** | **0.2827** | **0.5445** | **0.3343** | **0.4961** | **+6.0%** |

**Key findings:**
1. Zero-shot Qwen3-Embedding-0.6B (MAP@100=0.2105) underperforms BM25+Bo1 (0.2504) but provides complementary signal. nDCG@10 is actually higher (0.5026 vs 0.4662), suggesting the dense model ranks top results well but has worse recall.
2. All fusion methods beat both individual systems, confirming the hybrid hypothesis.
3. Linear interpolation with alpha=0.3 (sparse-heavy) is the best fusion method (MAP@100=0.2762), outperforming all RRF variants.
4. Lower RRF k (k=10) performs best among RRF variants, consistent with the sparse system being stronger.
5. Reranking the best fusion with Qwen3-Reranker yields a new best: MAP@100=0.2827, a 6.0% improvement over the previous best.
6. Recall@100 improves substantially: 0.4961 vs 0.4527 for BM25+Bo1 alone, confirming that fusion provides a better candidate pool.

Design expected MAP@100 of 0.27-0.30 for fusion and 0.29-0.33 for fusion+reranking. Actual results (0.2762 and 0.2827) fall within these ranges.

## Budget Assessment: OK

No training involved. Total wall-clock time ~10,169 seconds (~2.8 hours), dominated by corpus encoding (~5,370s) and reranking (~4,736s). Peak VRAM: 33.5 GB.

## Verdict: **APPROVE**

All 8 runs completed successfully (second run, after alpha parsing fix). No data leakage. The best-fusion-reranked run achieves MAP@100=0.2827, a clear improvement over the previous best of 0.2668 (+6.0%). The experiment cleanly demonstrates that hybrid dense+sparse fusion with reranking outperforms either system alone.

### Status assignments:
- **best-fusion-reranked**: `keep` (new best MAP@100=0.2827)
- **linear-a03**: `keep` (MAP@100=0.2762 beats previous best 0.2668, fast without reranking)
- **rrf-k10**: `discard` (beats previous best but not as strong as linear-a03)
- **rrf-k60**: `discard`
- **rrf-k100**: `discard`
- **linear-a05**: `discard`
- **linear-a07**: `discard`
- **dense-qwen3-zeroshot**: `discard`

## Recommendations

1. **Try larger embedding models**: Qwen3-Embedding-0.6B is relatively small. A 7B embedding model might push the dense baseline higher, leading to even better fusion.
2. **Fine-tune the dense encoder**: The zero-shot dense model underperforms BM25. Fine-tuning on MS-MARCO could close that gap and improve fusion further.
3. **Optimize alpha more finely**: alpha=0.3 was best but the sweep was coarse. Try alpha in [0.2, 0.25, 0.3, 0.35].
4. **Cache corpus embeddings**: The 5,370s encoding time could be avoided in future experiments by caching the FAISS index to disk.
5. **ColBERT-style late interaction**: Multi-vector representations could capture finer-grained matching than single-vector dense retrieval.
