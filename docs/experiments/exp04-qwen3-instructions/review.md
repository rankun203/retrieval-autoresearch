# Review: exp04-qwen3-instructions

## Data Leakage Check: PASS

- `load_robust04()` called at line 81, returns `corpus`, `queries`, `qrels`.
- `corpus`: Used only for document text lookup in `get_doc_text()` (line 86-87). ALLOWED.
- `queries`: Used at line 121 to build BM25 topics dataframe, and at line 213 to get query text for reranking input pairs. Both are inference-time uses. ALLOWED.
- `qrels`: Used only at line 249 in `evaluate_run(reranked_run, qrels)` for final evaluation. ALLOWED.
- No training occurs (zero-shot instruction sweep). No model weights are updated.
- No hard negative mining or training data derived from Robust04 queries/qrels.

**Verdict: PASS** -- no data leakage.

## Code Quality

- Clean, well-structured code. Reranker implementation matches exp03b (correct prefix/suffix tokens, log_softmax scoring).
- BATCH_SIZE=64, consistent throughout.
- Proper GPU memory cleanup between runs (lines 228-229: `del inputs; torch.cuda.empty_cache()`).
- All 5 instructions clearly defined in the INSTRUCTIONS dict (lines 41-64).
- Comparison table and summary block properly formatted for log parsing.

Minor note: `MAX_LENGTH=768` in code (line 38) vs design.md says "Max input length: 4096 tokens" (line 57 of design.md). The effective max length is 768 minus prefix/suffix tokens. The design.md value is incorrect but this does not affect results -- the code is what actually ran, and 768 matches the exp03b ml768 configuration.

## Design Adherence

- Design specified 5 runs (default + 4 experimental instructions): all 5 completed.
- Design specified rerank depth 100: confirmed in code and logs.
- Design specified BM25+Bo1 first stage with matching parameters: confirmed.
- Instructions in code match design.md exactly.
- Design expected MAP@100 range of 0.265-0.285: actual range was 0.2597-0.2667, slightly below expectations.
- Design expected default to match exp03b at ~0.2675: actual was 0.2597. This is because the design compared against the exp03b top-1000 result (0.2668), but this experiment only reranks top-100. The exp03b top-100 result at ml768 was 0.2597, which matches perfectly. The design baseline comparison was slightly misleading but the code behavior is correct.

## Performance Analysis

| Run | MAP@100 | nDCG@10 | Recall@100 |
|-----|---------|---------|------------|
| qwen3-default | 0.2597 | 0.5326 | 0.4527 |
| qwen3-general-short | 0.2639 | 0.5400 | 0.4527 |
| qwen3-general-long | 0.2618 | 0.5345 | 0.4527 |
| qwen3-news-short | 0.2667 | 0.5429 | 0.4527 |
| qwen3-news-long | 0.2639 | 0.5364 | 0.4527 |

Key findings:
- **Best instruction**: "news-short" (MAP@100=0.2667, +2.7% relative over default at same depth)
- All custom instructions outperform the default web-search instruction.
- Domain-specific "news" framing helps more than generic "determine relevance" framing.
- Shorter instructions outperform longer ones in both categories (general-short > general-long, news-short > news-long).
- Recall@100 is identical across all runs (0.4527) as expected -- reranking does not change the candidate set.
- No run beats the current best MAP@100=0.2668 (exp03b, Qwen3 ml768 top-1000). The best run (news-short at 0.2667) is 0.0001 below, and that comparison is between top-100 vs top-1000 reranking depth.

### vs Baselines
- exp01 BM25+Bo1: 0.2504 -- all runs beat this significantly.
- exp03b Qwen3 ml768 top-100: 0.2597 -- news-short improves by +0.0070 (same-depth comparison).
- exp03b Qwen3 ml768 top-1000: 0.2668 -- no run beats this (current best).

## Budget Assessment: OK

Zero-shot experiment, no training. Total runtime ~2420s (5 reranking passes at ~475s each + BM25). Peak VRAM: 20.3 GB.

## Verdict: APPROVE

All 5 runs completed successfully. No data leakage. Code is correct and matches exp03b implementation. The experiment provides valuable insight: the "news-short" instruction ("Given a topic query, retrieve relevant news articles that discuss the topic") is the best instruction for Qwen3-Reranker on Robust04, improving MAP@100 by 2.7% over the default at the same rerank depth. However, no run sets a new overall best because all runs use rerank depth 100 only, while the current best uses depth 1000.

All 5 runs logged as **discard** since none exceeds MAP@100=0.2668.

## Recommendations

1. **Re-run news-short at depth 1000**: The news-short instruction should be tested at top-1000 rerank depth. If the 2.7% relative improvement over default holds at depth 1000, the expected MAP@100 would be ~0.274, which would be a new best.
2. **Use news-short as the default instruction** for all future Qwen3-Reranker experiments on Robust04.
3. **Consider instruction-document length interaction**: The shorter instructions may work better with the 768-token limit because they leave more room for document content. At higher max_length, longer instructions might close the gap.
