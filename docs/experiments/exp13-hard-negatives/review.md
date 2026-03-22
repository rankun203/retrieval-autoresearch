# Review: exp13-hard-negatives

## Data Leakage Check: PASS

Verified every reference to `qrels`, `queries`, and `load_robust04` in both `train.py` and `llm_cleanup.py`:

- **`load_robust04()`** (train.py line 132): Returns `corpus`, `queries`, `qrels`.
- **`corpus`**: Used for document text retrieval and encoding (allowed).
- **`queries`**: Used only at lines 977-978 (encoding test queries for evaluation) and line 1058 (building BM25 topics for evaluation). Never used during training or mining phases.
- **`qrels`**: Used exclusively in `evaluate_run()` calls at lines 1011, 1109, 1184. All final evaluation. ALLOWED.
- **`llm_cleanup.py`**: Does not import or reference `qrels` or `queries` at all. Loads only cached mining data containing MS-MARCO queries.
- **Training queries**: All sourced from `stream_msmarco_triples()` (train.py line 186), filtered by news-domain keywords. The `news_queries` variable contains only MS-MARCO queries throughout Phases 0-3.

No data leakage detected.

## Code Quality

**Strengths:**
- Clean 3-phase pipeline with proper separation of concerns (train.py for mining+training, llm_cleanup.py for LLM scoring in a separate process to avoid CUDA fork conflicts)
- Proper VRAM management: models freed between phases, fp32 for training / fp16 for inference
- Caching at every stage (news queries, mining data, LLM scores, embeddings) for reproducibility
- MarginMSE loss implementation is correct: `MSE(bi_encoder_margin, teacher_margin)`
- Gradient checkpointing enabled for memory efficiency
- Numpy dot product instead of faiss-gpu to avoid CUDA conflicts
- vLLM logprobs extraction for continuous P(yes) scores is well-implemented

**Issues:**
- The CE positive threshold of 0.8 was too strict, yielding only 139/500 valid training queries. This is the root cause of underperformance.
- Lines 562-563: Falls back to MS-MARCO positive text when no doc exceeds CE threshold, but this positive may not exist in the Robust04 corpus. This fallback is bypassed for the 139 valid examples (line 582: `if len(hard_negs) < 2: continue`) but is a design gap.
- Loss converging to near-zero (0.012 at 100%) with only 862 examples over 4 epochs indicates memorization rather than generalization.

## Cache Verification

- **News queries cache**: `news_queries_msmarco_count-500` -- correct, 500 MS-MARCO queries.
- **Mining cache**: `hard_negatives_Qwen_Qwen3-Embedding-0.6B_ce_model-BAAI_bge-reranker-v2-m3_num_queries-500_retrieval_depth-500_round-0` -- correct model names and parameters.
- **LLM scores cache**: `llm_scores_Qwen_Qwen3-8B_condition-think_num_pairs-723` -- correct model, 723 pairs (139 positives + 584 negatives).
- **Base embeddings cache**: Uses model=Qwen3-Embedding-0.6B, max_length=512, dataset=robust04 -- correct.

All caches match expected models and parameters.

## Design Adherence

| Design spec | Actual | Match? |
|---|---|---|
| 500 MS-MARCO news queries | 500 queries selected | Yes |
| BM25 mining round 0 | BM25 used via PyTerrier | Yes |
| BGE-reranker-v2-m3 CE scoring | BGE-reranker-v2-m3 loaded | Yes |
| Qwen3-8B LLM judge with thinking | Qwen3-8B via vLLM, think+no-think | Yes |
| 2 rounds of remining | 2 rounds executed | Yes |
| MarginMSE loss | Implemented correctly | Yes |
| fp32 training | fp32 confirmed | Yes |
| Run 1: dense-only | Completed | Yes |
| Run 2: fusion alpha=0.3 | Completed | Yes |
| Run 3: expanded+fusion | Skipped (expansion data unavailable) | Partial |

Run 3 was skipped because expanded corpus data from exp12 was not available in the worktree cache. This is acceptable -- the first two runs already demonstrate the core result.

## Performance Analysis

### Results vs Expectations

| Run | Expected MAP@100 | Actual MAP@100 | Delta |
|---|---|---|---|
| finetuned-dense-only | 0.22-0.24 | 0.1888 | Below expected |
| finetuned-fusion-a03 | 0.29-0.31 | 0.2724 | Below expected |

Both runs **underperformed** expectations and zero-shot baselines:
- Dense-only: 0.1888 vs 0.2105 zero-shot (-0.0217, a 10.3% degradation)
- Fusion: 0.2724 vs 0.2762 zero-shot fusion (-0.0038, a 1.4% degradation)

### Root Cause: Insufficient Training Data

Only 139 out of 500 MS-MARCO queries produced valid training examples (CE > 0.8 threshold). This yielded:
- Round 1: 862 training examples (139 queries x ~6.2 negatives each)
- Round 2: Similar count from bi-encoder remining

With only ~860 examples and 202 steps per round (4 epochs), the model memorized the small training set (loss dropped to 0.012) rather than learning generalizable domain representations.

### LLM Scoring Analysis

The LLM scoring infrastructure worked well:
- **Think mode**: 723/723 valid scores, mean P(yes)=0.527, pos/neg separation=0.234
- **No-think mode**: 723/723 valid scores, mean P(yes)=0.812, pos/neg separation=0.158
- Think mode showed 48% better pos/neg separation than no-think, confirming the hypothesis that CoT reasoning produces better-calibrated scores
- No-think mode exhibited strong positivity bias (mean 0.812 vs 0.527 for think)
- Total LLM scoring time: ~1106s (think) + ~33s (no-think) = ~19 minutes for 723 pairs

### Comparison to Current Best

| Method | MAP@100 | Status |
|---|---|---|
| Qwen3-8B fusion (exp07, best) | 0.2929 | keep |
| Qwen3-0.6B + doc expansion fusion (exp12) | 0.2903 | keep |
| Qwen3-0.6B fusion zero-shot (exp05) | 0.2762 | keep |
| **This exp: finetuned fusion** | **0.2724** | **discard** |
| **This exp: finetuned dense-only** | **0.1888** | **discard** |

## Budget Assessment

- Round 1: 202 steps, 601.8s, budget_assessment=OK
- Round 2: ~205 steps (total 407), budget_assessment=OK
- Loss curves show rapid convergence to near-zero, consistent with overfitting a small dataset
- The "OK" budget assessment is technically correct (loss was still decreasing) but misleading -- the model converged quickly because the dataset was too small, not because training was well-paced
- Peak VRAM: 37.7 GB
- Total time: ~19,289s (~5.4 hours), dominated by corpus re-encoding for round 2 (~5,540s)

## Verdict: **APPROVE**

Despite negative results, this experiment is methodologically sound:
- No data leakage
- Correct implementation of MarginMSE, hard negative mining, and LLM scoring
- The failure mode is well-understood: insufficient training data from overly strict CE threshold (only 139/500 queries yielded valid examples)
- The LLM think-mode scoring infrastructure is validated and reusable
- Both runs are DISCARD (below zero-shot baselines)

### Status assignments:
- **finetuned-dense-only**: `discard` (MAP@100=0.1888, below zero-shot 0.2105)
- **finetuned-fusion-a03**: `discard` (MAP@100=0.2724, below zero-shot fusion 0.2762)

## Recommendations

1. **Lower CE positive threshold**: From 0.8 to 0.5-0.6. This should yield 300-400+ valid training queries instead of 139, providing 5-10x more training data.

2. **Scale up MS-MARCO query pool**: Use 2000-5000 queries instead of 500 to increase the chances of finding good query-document matches in the Robust04 corpus.

3. **Use LLM think scores to re-label positives**: Instead of requiring CE > 0.8 for positives, use the LLM P(yes) scores to identify relevant documents. The think-mode P(yes) showed good calibration.

4. **Consider GPL-style training**: Generate synthetic queries for Robust04 documents using an LLM, then use CE/LLM scores as teacher. This avoids the bottleneck of finding MS-MARCO queries that match Robust04 documents.

5. **Preserve the LLM scoring infrastructure**: The vLLM + think-mode + logprobs pipeline is correct and efficient (1106s for 723 think judgments). Scale it up in future experiments.
