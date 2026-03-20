# exp11-llm-rerank: Listwise LLM Reranking (RankGPT-style)

## Literature Review

### RankGPT -- Is ChatGPT Good at Search? (Sun et al., EMNLP 2023, Outstanding Paper Award)
- URL: https://arxiv.org/abs/2304.09542
- GitHub: https://github.com/sunnweiwei/RankGPT
- Key idea: Instruct an LLM to rank passages by relevance using permutation generation. Present N passages with identifiers [1]...[N], ask LLM to output them in descending relevance order.
- Sliding window: Window size W (e.g., 20), stride S (e.g., 10). Slide from back to front of the ranked list. In each window, rerank W passages; the top (W-S) "bubble up" and the next S passages join.
- Results: GPT-4 achieves SOTA on TREC DL19/20, BEIR. GPT-3.5 also competitive. Works zero-shot.
- Key insight: The sliding window approach is critical -- without it, only a small window of docs can be reranked per LLM call.

### RankZephyr -- Zero-Shot Listwise Document Reranking is a Breeze! (Pradeep et al., 2023)
- URL: https://arxiv.org/abs/2312.02724
- Model: castorini/rank_zephyr_7b_v1_full (7B params, based on Zephyr-7B-beta / Mistral-7B)
- Distilled from GPT-3.5/GPT-4 listwise ranking outputs. Fine-tuned for the RankGPT permutation generation task.
- Performance: Matches or exceeds GPT-4 on TREC DL19/20 and BEIR. ~14GB in fp16.
- This is our primary candidate model -- purpose-built for listwise reranking, fits in VRAM.

### FIRST: Faster Improved Listwise Reranking with Single Token Decoding (Reddy et al., EMNLP 2024)
- URL: https://arxiv.org/abs/2406.15657
- Key idea: Instead of generating full permutation, extract first-token logits over passage identifiers. Rank by logit values. 50% faster, comparable accuracy.
- We may try this approach as an optimization if full generation is too slow.

### RankLLM Toolkit (Sharifymoghaddam et al., SIGIR 2025)
- URL: https://github.com/castorini/rank_llm
- Python package for listwise reranking with vLLM backend support for RankZephyr/RankVicuna.
- Implements sliding window, prompt formatting, permutation parsing.
- Data model: Candidate(docid, score, doc), Request(query, candidates), Result(query, candidates).
- Default: window_size=20, stride=10, max_passage_words=300.

### Ranked List Truncation for LLM Re-Ranking (Meng et al., SIGIR 2024)
- URL: https://arxiv.org/abs/2404.18185
- Finding: With strong first-stage retrievers, fixed depth-20 can be as effective as depth-100+.
- Implication: We should test both shallow (top-20) and deeper (top-50, top-100) reranking depths.

### How Good are LLM-based Rerankers? (EMNLP Findings 2025)
- URL: https://arxiv.org/abs/2508.16757
- Robust04 nDCG@10 reference: FLAN-T5-XL ~0.507, FLAN-UL2 ~0.534 with listwise.
- RankFlow achieves +5.14 nDCG@10 over RankGPT-4 on Robust04.
- Listwise approaches degrade ~8% on unseen domains (best generalization among reranking types).

## Goal

Use a listwise LLM reranker (RankZephyr 7B via RankGPT-style sliding window) on top of the current best first-stage fusion pipeline (BM25+Bo1 + Qwen3-Embedding-0.6B, alpha=0.3). The hypothesis is that a 7B model purpose-built for listwise reranking will improve nDCG@10 and MAP@100 over the fusion-only baseline, unlike the 0.6B cross-encoder which actually hurt performance.

## Hypothesis

1. The Qwen3-Reranker-0.6B failed because it is a pointwise cross-encoder that independently scores each document -- it lacks the comparative reasoning that listwise rerankers provide, and at 0.6B parameters it may be weaker than the 8B embedding signal.
2. RankZephyr (7B) is 12x larger than Qwen3-Reranker-0.6B and was specifically fine-tuned for listwise reranking via distillation from GPT-3.5/4 ranking outputs. It considers documents in context of each other, enabling better relative ordering.
3. The sliding window approach allows the LLM to perform multiple passes, progressively bubbling up the best documents -- more sophisticated than a single-pass pointwise scorer.
4. Literature suggests listwise LLM rerankers significantly improve nDCG@10 (precision-oriented) while maintaining or improving MAP.

## Method

1. **First stage**: Reuse the exp07 pipeline -- BM25+Bo1 retrieval + Qwen3-Embedding-0.6B dense retrieval + linear fusion (alpha=0.3).
   - Cache the 8B embeddings and BM25 run to `.cache/` for reuse across experiments.
   - After fusion is computed, free the embedding model from GPU.

2. **Second stage**: Listwise LLM reranking with RankZephyr 7B.
   - Use the `rank_llm` library with vLLM backend for fast inference.
   - RankGPT-style sliding window: window_size=20, step_size=10.
   - Rerank top-K documents from the fusion run.
   - Try K=20 (fast, ~2 passes), K=50 (medium, ~4 passes), K=100 (deep, ~9 passes).
   - Fallback: If rank_llm has dependency issues, implement sliding window manually with vLLM.

3. **VRAM management**:
   - Phase 1: Load Qwen3-8B (~16GB), encode documents, build FAISS index, free model.
   - Phase 2: Load RankZephyr 7B (~14GB via vLLM), perform reranking, free model.
   - Peak VRAM: ~20GB (never both models loaded simultaneously).

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Embedding model | Qwen/Qwen3-Embedding-0.6B |
| Embedding batch size | 256 |
| Doc max length | 512 |
| Query max length | 512 |
| Fusion alpha | 0.3 (dense weight) |
| BM25 k1, b | 0.9, 0.4 |
| Bo1 fb_docs, fb_terms | 5, 30 |
| Reranker model | castorini/rank_zephyr_7b_v1_full |
| Reranker VRAM (fp16) | ~14GB |
| Sliding window size | 20 |
| Sliding window step | 10 |
| Rerank depths | 20, 50, 100 |
| Max passage words | 300 (truncate doc text for prompt) |
| vLLM tensor_parallel | 1 |
| vLLM context size | 4096 |

## Runs

### Run 1: `fusion-baseline`
- The first-stage fusion pipeline only (no reranking). Should reproduce exp07 MAP@100=0.2929.
- Serves as the comparison baseline for this experiment.

### Run 2: `rerank-top20`
- Rerank only top-20 fusion results with RankZephyr.
- Window=20, step=10 (2 sliding window passes over 20 docs).
- Fast. Should primarily improve nDCG@10.

### Run 3: `rerank-top50`
- Rerank top-50 fusion results.
- Window=20, step=10 (~4 passes per query).
- Medium speed. Should improve both nDCG@10 and MAP@100.

### Run 4: `rerank-top100`
- Rerank top-100 fusion results.
- Window=20, step=10 (~9 passes per query).
- Slower but maximizes reranking benefit. Best chance for MAP@100 improvement.

## Expected Outcome

- **nDCG@10**: Expect 0.54-0.58 (current: 0.5258). Listwise rerankers are strongest on precision-oriented metrics.
- **MAP@100**: Expect 0.29-0.32 (current: 0.2929). Modest improvement; MAP depends on recall which reranking cannot improve.
- **Recall@100**: Should remain ~0.51 (unchanged; reranking only reorders, doesn't add new docs).
- If shallow reranking (top-20) already improves nDCG@10, that confirms the LLM adds value over the 8B embeddings.
- If MAP@100 does not improve even with top-100, then the fusion ranking is already near-optimal for MAP.

## Baseline Comparison

- exp07 fusion-only: MAP@100=0.2929, nDCG@10=0.5258, MAP@1000=0.3495, Recall@100=0.5103
- exp07 fusion+Qwen3-Reranker-0.6B: MAP@100=0.2797 (WORSE), nDCG@10=0.5384

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (zero-shot reranking, no training)
- [x] Training does NOT use Robust04 qrels (no training at all)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (pre-trained model, zero-shot)
