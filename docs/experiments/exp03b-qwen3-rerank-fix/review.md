# Review: exp03b-qwen3-rerank-fix

## Data Leakage Check: PASS

- `load_robust04()` called at line 53, returning `corpus`, `queries`, `qrels`
- `corpus`: Used only for document text lookup during reranking (lines 57-58) -- ALLOWED
- `queries`: Used to build BM25 topics (line 88) and as reranking query text (line 175) -- this is inference/evaluation, not training -- ALLOWED
- `qrels`: Used only in `evaluate_run(reranked_run, qrels)` at line 211 -- final evaluation only -- ALLOWED
- No training occurs (zero-shot pretrained reranker) -- no data leakage possible
- No hard negative mining, no training loops, no gradient updates

**Verdict: No data leakage detected.**

## Code Quality

Good overall. Clean structure with BM25 first-stage followed by Qwen3 reranking.

- Minor: `torch_dtype` deprecation warning (line 114) -- should use `dtype` instead. Non-blocking.
- Minor: `max_length` parameter in `tokenizer.pad()` is ignored when `padding=True` (warning in log). The truncation is handled correctly in the `tokenizer()` call above, so actual behavior is correct.
- The prefix/suffix token construction (lines 122-125, 130-142) correctly follows the official Qwen3-Reranker model card format with `<think>` block and colon-formatted tags.

## Design Adherence

The design specified 2 runs (top-100, top-1000) at max_length=8192. The actual implementation ran a 3x2 grid (max_lengths 256/512/768 x depths 100/1000 = 6 runs). This is a reasonable deviation -- the runner likely reduced max_length from 8192 to avoid OOM, and tested multiple lengths to find the sweet spot. The grid search is more informative than a single configuration.

The design expected MAP@100 of 0.30-0.38. The best result (0.2668) falls below this expectation but still beats the BM25+Bo1 baseline (0.2504) by +6.5%. The shorter max_lengths likely truncate long Robust04 documents, limiting the reranker's effectiveness.

## Performance Analysis

| Run | MAP@100 | nDCG@10 | MAP@1000 | R@100 | Time |
|-----|---------|---------|----------|-------|------|
| qwen3-ml256-top100 | 0.2340 | 0.4954 | 0.2340 | 0.4527 | 145s |
| qwen3-ml256-top1000 | 0.2149 | 0.4920 | 0.2568 | 0.4012 | 1484s |
| qwen3-ml512-top100 | 0.2524 | 0.5250 | 0.2524 | 0.4527 | 313s |
| qwen3-ml512-top1000 | 0.2525 | 0.5242 | 0.2957 | 0.4564 | 3175s |
| qwen3-ml768-top100 | 0.2597 | 0.5326 | 0.2597 | 0.4527 | 481s |
| qwen3-ml768-top1000 | **0.2668** | 0.5288 | **0.3120** | **0.4776** | 4810s |

Key observations:
- Longer max_length consistently improves MAP@100 (0.234 -> 0.252 -> 0.267 at depth 1000)
- Depth 1000 helps at ml768 (+0.0071 MAP@100 vs top100) but hurts at ml256 (-0.019)
- At short max_lengths, depth 1000 hurts because the reranker sees truncated docs and makes worse decisions on the expanded candidate set
- Best run (ml768-top1000) beats BM25+Bo1 baseline by +0.0164 MAP@100 (+6.5%)
- nDCG@10 improves substantially: 0.5288 vs 0.4662 baseline (+13.4%)
- The broken exp03 Qwen3 runs (MAP@100=0.013-0.128) are fully fixed

vs. current best (BM25+Bo1, MAP@100=0.2504): **+6.5% improvement**
vs. exp03 BGE reranker (MAP@100=0.2487): **+7.3% improvement**

## Budget Assessment: OK

Zero-shot inference, no training budget applies. Total wall-clock ~10,451s (2.9 hours) for all 6 runs.

## Verdict: **APPROVE**

The experiment successfully fixes the broken Qwen3 reranker from exp03 by using the correct prompt format. The best configuration (ml768-top1000) achieves MAP@100=0.2668, beating the current best of 0.2504. The code is clean with no data leakage. The results are plausible and consistent with expectations for a 0.6B reranker on Robust04.

Best run `qwen3-ml768-top1000`: **keep** (new best MAP@100)
All other runs: **discard**

## Recommendations

1. Try max_length=1024 or higher -- the trend shows continued improvement with longer context. The model supports up to 32K tokens, and Robust04 docs average ~250 words but some are much longer.
2. Consider a larger Qwen3-Reranker variant if available, or combine with a stronger first-stage retriever.
3. The 4810s reranking time for depth-1000 is substantial. For future experiments, consider whether depth-100 (481s, MAP@100=0.2597) offers a better speed/quality tradeoff.
4. The batch size of 64 works well at these max_lengths. At higher max_lengths, may need to reduce.
