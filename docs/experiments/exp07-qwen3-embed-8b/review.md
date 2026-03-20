# Review: exp07-qwen3-embed-8b

## Data Leakage Check: PASS

- **Line 74**: `corpus, queries, qrels = load_robust04()` -- all three loaded.
- **`corpus`**: Used for doc text encoding (line 204) and BM25 indexing (lines 96-100). ALLOWED -- corpus is not test data.
- **`queries`**: Used for BM25 topic submission (line 114), query embedding (lines 213-214), and reranker input (line 394). All at evaluation time in a zero-shot pipeline. No model weights are updated. ALLOWED.
- **`qrels`**: Used only in `evaluate_run()` calls at lines 268, 318, 425. ALLOWED.
- **No training loop**: The entire pipeline is zero-shot inference. No model parameters are modified.
- **No hard negative mining**: No training data sources used at all.

Verdict: **No data leakage detected.**

## Code Quality

- Clean, well-structured pipeline with clear section separators.
- Proper GPU memory management: embedding model freed before loading reranker (line 249).
- VRAM tracking throughout.
- Minor note: The first run attempt (lines 1-32 of log) crashed due to `flash_attention_2` not being installed; fixed to `sdpa` in commit 78e3b87. Good recovery.
- The `normalize_scores` function (lines 284-296) correctly handles edge cases (empty docs, zero range).
- `last_token_pool` (lines 138-149) follows the Qwen3-Embedding model card exactly.
- Reranker implementation follows the official prefix/suffix token format from the model card.

## Design Adherence

| Design Spec | Actual | Match? |
|-------------|--------|--------|
| Qwen3-Embedding-8B, 4096-dim | Yes, shape (528155, 4096) | Yes |
| BM25+Bo1 first stage | Yes, 249 queries x 1000 docs | Yes |
| Linear fusion alpha=0.3 | Yes | Yes |
| Qwen3-Reranker-0.6B, ml768, depth 1000 | Yes | Yes |
| 3 runs planned | 3 runs produced | Yes |

All runs match the design specification.

## Performance Analysis

| Run | MAP@100 | nDCG@10 | MAP@1000 | R@100 |
|-----|---------|---------|----------|-------|
| dense-qwen3-8b-zeroshot | 0.2644 | 0.5581 | 0.3047 | 0.4596 |
| linear-a03 | **0.2929** | 0.5258 | 0.3495 | 0.5103 |
| linear-a03-reranked | 0.2797 | 0.5384 | 0.3345 | 0.4989 |

### vs. Baselines and Current Best

- **Current best**: MAP@100=0.2827 (exp05-hybrid-rrf, Linear-a03 + Qwen3-Reranker)
- **linear-a03 (no rerank)**: 0.2929 -- **+3.6% above current best** -- new best!
- **dense-8b zeroshot**: 0.2644 -- nearly matches exp03b reranked pipeline (0.2668), demonstrating the 8B model's strong zero-shot performance. Massive improvement over 0.6B dense-only (0.2105, +25.6%).
- **Reranking hurt**: linear-a03-reranked (0.2797) < linear-a03 (0.2929). The Qwen3-Reranker-0.6B degrades rankings when applied to the already high-quality 8B fusion candidates. This is a meaningful finding.

### Key Insight

The 8B embedding model produces such high-quality dense representations that the 0.6B reranker adds noise rather than signal. Future experiments should either skip reranking entirely with this encoder, or use a larger/better reranker (e.g., Qwen3-Reranker-8B if available).

## Budget Assessment: OK

- Zero-shot pipeline, no training. No overfitting risk.
- Total runtime: 34,521s (~9.6 hours), dominated by 528K document encoding (~8.3 hours).
- Peak VRAM: 22,394 MB (~21.9 GB) during encoding phase.

## Verdict: **APPROVE**

The experiment is clean, well-executed, and produces a new best MAP@100 of 0.2929 (+3.6% over previous best). No data leakage, no training involved. The finding that reranking hurts with the 8B encoder is valuable for guiding future experiments.

### Results to Log

| Run | Status |
|-----|--------|
| dense-qwen3-8b-zeroshot | discard (0.2644 < 0.2827 best) |
| linear-a03 | **keep** (0.2929 > 0.2827 best, new best) |
| linear-a03-reranked | discard (0.2797 < 0.2929 new best) |

## Recommendations

1. **Skip reranking with Qwen3-Embedding-8B** -- the 0.6B reranker is a bottleneck, not a boost.
2. **Try alpha tuning** -- alpha=0.3 was inherited from exp05 with 0.6B. The 8B model may benefit from higher dense weight (alpha=0.4 or 0.5).
3. **Try Qwen3-Embedding-8B with Matryoshka dimensions** -- lower dims (e.g., 2048, 1024) could speed encoding 2-4x while retaining most quality.
4. **Consider fine-tuning** -- the 8B model is strong zero-shot; even light MS-MARCO fine-tuning could push MAP@100 past 0.30.
5. **Try a stronger reranker** -- if reranking is desired, a larger reranker (8B class) may complement rather than degrade the 8B embeddings.
