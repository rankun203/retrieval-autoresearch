# Review: exp11-llm-rerank

## Data Leakage Check: PASS

Both `train.py` and `rerank_only.py` were reviewed line by line.

**train.py:**
- Line 80: `corpus, queries, qrels = load_robust04()`
- `qrels` referenced only at lines 396 and 633, both in `evaluate_run()` calls -- final evaluation only.
- `queries` used at line 122 (BM25 topics), lines 267-268/306-307 (encode queries for dense retrieval), line 575 (query text for reranking prompt). All evaluation-time usage. No training occurs in this experiment (zero-shot).
- `corpus` used for encoding/indexing -- allowed.

**rerank_only.py:**
- Line 31: `corpus, queries, qrels = load_robust04()`
- `qrels` referenced only at line 206 in `evaluate_run()` -- final evaluation only.
- `queries` used at line 149 to get query text for reranking prompts -- evaluation-time usage only.

No training data is needed (RankZephyr is a pre-trained model used zero-shot). No qrels or queries flow into model weights or training data selection.

## Code Quality

**Strengths:**
- Clean separation into two scripts (train.py for embedding+fusion, rerank_only.py for vLLM reranking) to avoid CUDA initialization conflicts.
- Proper caching of BM25 runs and document embeddings with correct cache keys.
- RankGPT sliding window implementation is faithful to the paper: slides from end to front, window_size=20, step_size=10.
- Robust parsing fallback (lines 88-98 in rerank_only.py): missing indices are appended in original order.
- Error handling with try/except around vLLM calls (rerank_only.py lines 166-177).

**Issues:**
- `train.py` lines 528-645 load vLLM and attempt reranking, but this path crashed because the embedding model already initialized CUDA (conflicting with vLLM's subprocess spawning). The split into rerank_only.py was the correct fix, but the dead code in train.py remains.
- `rerank_only.py` line 243 hardcodes `peak_vram_mb: 14000.0` rather than measuring actual VRAM. vLLM manages its own CUDA context in a subprocess, so torch.cuda metrics are unavailable. Acceptable for reporting.
- `MAX_PASSAGE_WORDS` was reduced from 300 (design.md) to 100 (rerank_only.py) to fit within the 4096 context window. This is a necessary fix but limits document text available to the LLM.

## Cache Verification: PASS

From the phase 1 log:
- BM25+Bo1 cache: `.cache/bm25_run_BM25-Bo1_b-0.4_dataset-robust04_fb_docs-5_fb_terms-30_k1-0.9_top_k-1000` -- correct parameters (k1=0.9, b=0.4, fb_docs=5, fb_terms=30, top_k=1000).
- Embeddings cache: `.cache/embeddings_Qwen_Qwen3-Embedding-0.6B_dataset-robust04_max_length-512_pooling-last_token` -- correct model (Qwen3-Embedding-0.6B), correct dataset (robust04).

## Design Adherence

Partially adhered:

| Design spec | Actual | Match? |
|---|---|---|
| Qwen3-Embedding-0.6B | 0.6B used | Yes |
| Fusion alpha=0.3 | 0.3 used | Yes |
| RankZephyr 7B | castorini/rank_zephyr_7b_v1_full | Yes |
| Window=20, step=10 | 20, 10 | Yes |
| Depths 20, 50, 100 | All three tested | Yes |
| MAX_PASSAGE_WORDS=300 | Reduced to 100 | No (necessary to fit 4096 context) |
| fusion-baseline reproduces exp07 MAP@100=0.2929 | MAP@100=0.2781 | No (expected: uses 0.6B not 8B) |

The fusion-baseline MAP@100=0.2781 is correct for 0.6B embeddings (exp05 reported 0.2762 for the same fusion without reranking). The design incorrectly expected 0.2929 (which was the 8B result from exp07).

## Performance Analysis

### Fusion Baseline (from train.py phase 1)
| Metric | Value | exp05 (0.6B fusion) | exp07 (8B fusion) |
|---|---|---|---|
| MAP@100 | 0.2781 | 0.2762 | 0.2929 |
| nDCG@10 | 0.5138 | 0.5095 | 0.5258 |
| Recall@100 | 0.4914 | 0.4920 | 0.5103 |

### Reranking Runs (from rerank_only.py)
| Run | MAP@100 | nDCG@10 | MAP@1000 | R@100 | LLM calls | Duration | Parse failures |
|---|---|---|---|---|---|---|---|
| rerank-top20 | 0.2761 | 0.5430 | 0.3296 | 0.4914 | 249 | 565s | 1 |
| rerank-top50 | 0.2822 | 0.5533 | 0.3357 | 0.4914 | 996 | 3264s | 10 |
| rerank-top100 | 0.2820 | 0.5573 | 0.3356 | 0.4914 | 2241 | 7817s | 35 |

### Key Observations

1. **nDCG@10 improved significantly**: +5.7% for top-20 (0.5138 to 0.5430), +7.7% for top-50 (to 0.5533), +8.5% for top-100 (to 0.5573). Confirms RankZephyr's strength at precision-oriented reranking.
2. **MAP@100 barely changed**: top-20 slightly hurt (-0.0020), top-50 improved modestly (+0.0041), top-100 similar (+0.0039). Reranking cannot add new relevant documents, only reorder.
3. **Recall@100 unchanged** at 0.4914 for all runs, as expected.
4. **Diminishing returns past top-50**: top-100 took 2.4x longer than top-50 but yielded nearly identical MAP@100 (0.2820 vs 0.2822) and only marginal nDCG@10 improvement (0.5573 vs 0.5533).
5. **Parse failures increase with depth**: 1 (top-20), 10 (top-50), 35 (top-100). More windows means more chances for malformed LLM output. The fallback to original order is appropriate.

### Comparison to Current Best
- Current best MAP@100: 0.2929 (exp07, Qwen3-Embedding-8B fusion)
- Best this experiment: 0.2822 (rerank-top50), 3.65% below current best
- Best 0.6B-based MAP@100: 0.2827 (exp05, fusion+Qwen3-Reranker-0.6B top-1000)
- This experiment's best: 0.2822, 0.18% below exp05

The RankZephyr 7B listwise reranker achieves comparable MAP@100 to the Qwen3-Reranker-0.6B pointwise reranker (0.2822 vs 0.2827), but with much higher nDCG@10 (0.5533 vs 0.5445). However, the 7B model is significantly slower (3264s vs ~336s for the 0.6B reranker) and neither approach exceeds the 8B embedding fusion baseline.

## Budget Assessment: OK

Zero-shot inference, no training. Reranking durations are reasonable for a 7B model: 565s (top-20), 3264s (top-50), 7817s (top-100). Total reranking wall time: ~11,687s (~3.2 hours).

## Verdict: **APPROVE** (discard all runs)

The experiment was conducted correctly with no data leakage. The RankGPT-style listwise reranking implementation is faithful to the literature and produces credible results. However:

- No run exceeds the current best MAP@100 of 0.2929 (exp07).
- No run exceeds the best 0.6B-based MAP@100 of 0.2827 (exp05).
- Status: **discard** for all four runs.

The nDCG@10 improvements (up to 0.5573) are noteworthy and confirm the value of listwise LLM reranking for top-of-list precision. If nDCG@10 were the primary metric, these results would be competitive.

## Recommendations

1. **LLM reranking on top of 8B embeddings**: The 0.6B first-stage limits recall@100 (0.49 vs 0.51 for 8B). Running RankZephyr on the 8B fusion pipeline could push past 0.2929, but 8B encoding is ~8 hours.
2. **Longer passage context**: MAX_PASSAGE_WORDS=100 severely truncates documents. A model supporting longer contexts (8K-16K) with 300+ words could improve ranking quality.
3. **FIRST method**: Single-token logit extraction (Reddy et al. 2024) could give comparable reranking quality at 50% of the latency.
4. **Focus on first-stage recall**: The fundamental bottleneck is recall@100=0.49. Domain-adapted embeddings or better query expansion would have more impact on MAP@100 than reranking.
