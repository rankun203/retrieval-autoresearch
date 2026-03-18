# exp23-qwen3-reranker: BM25+Bo1 with Qwen3-Reranker-0.6B (top-100)

## Goal
Test whether a small LLM-based reranker (Qwen3-Reranker-0.6B) can improve the BM25+Bo1 first-stage results by reranking the top-100 candidates per query.

## Hypothesis
LLM-based rerankers with yes/no relevance judgment can capture deeper semantic relevance than BM25's lexical matching. Even a 0.6B parameter model should significantly improve precision metrics (nDCG@10, MAP@100) when applied to a strong BM25+Bo1 candidate pool. The thinking-mode architecture (with hidden chain-of-thought) may enable better reasoning about document relevance.

## Method
- Load precomputed BM25+Bo1 TREC run file from exp22
- Keep top-100 candidates per query
- Rerank using Qwen3-Reranker-0.6B: format each query-document pair as a relevance judgment prompt, extract P(yes) from the last token logits
- Score = softmax(logit_yes, logit_no)[yes]
- Prompt template uses system instruction for yes/no judgment with thinking mode enabled

## Key parameters
- Reranker: Qwen/Qwen3-Reranker-0.6B (float16)
- RERANK_TOP_K: 100
- Batch size: 4 (documents are long)
- MAX_CONTENT_TOKENS: 512
- Task instruction: "Given a web search query, retrieve relevant passages that answer the query"

## Expected outcome
- Significant nDCG@10 improvement over BM25+Bo1 baseline (0.4662)
- MAP@100 improvement over BM25+Bo1 (0.2504)
- Recall@100 unchanged (same candidate pool, just reordered)

## Baseline comparison
- BM25+Bo1 (exp22): MAP@100=0.2504, nDCG@10=0.4662
- Dense + cross-encoder reranker (exp19): MAP@100=0.2106, nDCG@10=0.4775

## Results
- MAP@100=0.2552, nDCG@10=0.5292, recall@100=0.4527
- New best nDCG@10 at the time (+13.5% over BM25+Bo1 alone)
- MAP@100 slightly better than BM25+Bo1 (0.2552 vs 0.2504)
- Recall@100 unchanged (same candidate pool, just reordered as expected)
- VRAM usage: 2.6 GB (very efficient)
- Reranking was extremely fast (0.014s evaluation time for 100 docs/query)
