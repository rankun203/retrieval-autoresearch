# Review: exp03-cross-encoder-rerank

## Data Leakage Check: PASS

- `load_robust04()` called at line 82, returning `corpus`, `queries`, `qrels`.
- `queries` used at line 119-121 to build `topics_df` for BM25 retrieval, and at lines 173/246 to get query text for cross-encoder scoring. Since this is a zero-shot experiment (no training whatsoever), using test queries at inference time is the standard evaluation protocol. No test-time information flows into model weights.
- `qrels` used only at lines 341 and 376, both inside `evaluate_run()` -- final evaluation only.
- `corpus` used for Terrier indexing (lines 97-99) and document text retrieval (lines 142-149). Corpus is not test data.
- No calls to `stream_msmarco_triples()` or any training loop. All three models are loaded from HuggingFace and used as-is.
- No hard negative mining, no fine-tuning, no gradient updates anywhere in the code.

**Verdict: No data leakage detected.**

## Code Quality

**Good:**
- Clean separation of reranking logic per model type (`rerank_with_cross_encoder` vs `rerank_with_qwen3`)
- Proper GPU memory cleanup between runs (`gc.collect()`, `torch.cuda.empty_cache()`)
- Configurable model selection via `RUN_MODELS` env var
- VRAM tracking throughout
- Progress reporting every 50 queries

**Issues:**
1. The Qwen3 reranker implementation has a fundamental scoring problem. The `yes`/`no` token IDs obtained via `tokenizer.convert_tokens_to_ids()` at lines 229-230 may not match the actual token IDs the model uses for "yes"/"no" responses. The official Qwen3-Reranker uses `tokenizer.apply_chat_template()` rather than manual prompt construction. The resulting MAP@100 of 0.1285 (top-100) and 0.0138 (top-1000) confirms the scoring is broken -- worse than random for the top-1000 case.
2. The Qwen3 tokenizer uses `padding_side="left"` (line 216) which is correct for causal LMs, but the `max_length=512` truncation (line 263) may be too aggressive given the verbose prompt template, leaving little room for document text.
3. `eval_duration` in the summary block (line 412) reports BM25 time (`t_bm25`) rather than total evaluation time, which is misleading.
4. MiniLM reranking hurts MAP@100 vs BM25 baseline (0.2351 vs 0.2504), suggesting the model is not strong enough for Robust04's query/document style.

## Design Adherence

The design specified 6 runs (3 models x 2 depths) and all 6 were executed:

| Run | Specified | Completed |
|-----|-----------|-----------|
| minilm-rerank-top100 | Yes | Yes |
| minilm-rerank-top1000 | Yes | Yes |
| bge-rerank-top100 | Yes | Yes |
| bge-rerank-top1000 | Yes | Yes |
| qwen3-rerank-top100 | Yes | Yes |
| qwen3-rerank-top1000 | Yes | Yes |

All run files present in `runs/` directory. BM25+Bo1 baseline parameters match design (k1=0.9, b=0.4, fb_docs=5, fb_terms=30).

Batch sizes differ slightly from design (128 vs 64 for MiniLM, 64 vs 32 for BGE/Qwen3) but these are reasonable adjustments that do not affect correctness.

**Adherence: GOOD**

## Performance Analysis

| Run | MAP@100 | nDCG@10 | MAP@1000 | Recall@100 | Time(s) |
|-----|---------|---------|----------|------------|---------|
| BM25+Bo1 baseline | 0.2504 | 0.4662 | 0.2968 | 0.4527 | 22.7 |
| minilm-rerank-top100 | 0.2351 | 0.4697 | 0.2351 | 0.4527 | 46.3 |
| minilm-rerank-top1000 | 0.2260 | 0.4648 | 0.2676 | 0.4242 | 241.5 |
| bge-rerank-top100 | 0.2487 | 0.5059 | 0.2487 | 0.4527 | 344.5 |
| bge-rerank-top1000 | 0.2431 | 0.5046 | 0.2864 | 0.4376 | 3298.7 |
| qwen3-rerank-top100 | 0.1285 | 0.2258 | 0.1285 | 0.4527 | 314.9 |
| qwen3-rerank-top1000 | 0.0138 | 0.0686 | 0.0505 | 0.1068 | 3176.8 |

**Key findings:**

1. **No run beats the BM25+Bo1 baseline on MAP@100.** The best cross-encoder run (bge-rerank-top100, MAP@100=0.2487) is slightly below baseline (0.2504).

2. **BGE improves nDCG@10 significantly** (0.5059 vs 0.4662, +8.5%), indicating it does improve top-10 precision. The MAP@100 deficit comes from cross-encoder scores not preserving ranking quality for documents outside the top-10.

3. **Reranking top-1000 consistently hurts MAP@100** compared to top-100 for all models. The cross-encoders are miscalibrated for Robust04 and push irrelevant documents into the top-100 when given a larger candidate pool.

4. **Qwen3-Reranker is broken.** The catastrophic performance (MAP@100=0.0138 for top-1000, recall@100=0.1068) confirms the yes/no logit scoring implementation does not work correctly with this model.

5. **Expectations vs reality:** Design predicted MAP@100 of 0.28-0.36 across runs. Actual best was 0.2487. The hypothesis that zero-shot cross-encoders would significantly improve over BM25+Bo1 on Robust04 was incorrect. BM25+Bo1 with query expansion is a strong baseline on Robust04, and cross-encoders trained on MS-MARCO short passages do not generalize well to Robust04's longer newswire documents and complex topics.

## Budget Assessment: OK

Zero-shot experiment, no training budget consumed. Total wall-clock time: 7462s (~2 hours). Peak VRAM: 23.8 GB.

## Verdict: **APPROVE**

The experiment is methodologically sound despite disappointing results. There is no data leakage, the code runs correctly (except for the Qwen3 scoring bug which is an implementation issue, not a leakage concern), and all specified runs completed. The negative result is valuable -- it establishes that zero-shot cross-encoder reranking does not beat BM25+Bo1 PRF on MAP@100 for Robust04.

**Status: All runs are `discard`** -- none beat the current best MAP@100=0.2504.

## Results to Log

1. **bge-rerank-top100** (best non-broken run): `discard`
2. **qwen3-rerank-top100**: `discard` (broken scoring, documented)
3. **qwen3-rerank-top1000**: `discard` (broken scoring, documented)

## Recommendations

1. **Cross-encoder + BM25 score fusion**: Instead of replacing BM25+Bo1 scores, try linear interpolation of BM25+Bo1 and cross-encoder scores. This could preserve recall from query expansion while gaining top-rank precision from the cross-encoder.
2. **Fix Qwen3 implementation**: Use `tokenizer.apply_chat_template()` and increase `max_length`. However, given other models' poor MAP@100, fixing Qwen3 alone is unlikely to beat the baseline.
3. **Fine-tuned dense retrieval**: Given that zero-shot approaches (exp02 dense, exp03 reranking) both fail to beat BM25+Bo1, the next experiment should focus on fine-tuned models that can learn Robust04-relevant features from MS-MARCO training data.
4. **Hybrid retrieval**: Combine BM25+Bo1 with a fine-tuned dense retriever for complementary signal, rather than pure reranking.
