# exp24-qwen3-reranker-top1k: BM25+Bo1 with Qwen3-Reranker-0.6B (top-1000)

## Goal
Test whether reranking a larger candidate pool (top-1000 vs top-100) with Qwen3-Reranker-0.6B improves MAP by surfacing relevant documents ranked beyond position 100 in the BM25+Bo1 first stage.

## Hypothesis
BM25+Bo1 retrieves relevant documents beyond rank 100 that the reranker could promote into the top-100. By expanding the reranking pool from 100 to 1000, we increase the recall ceiling available to the reranker, which should improve MAP@100 (and enable meaningful MAP@1000 measurement).

## Method
- Same pipeline as exp23 but with RERANK_TOP_K=1000 instead of 100
- Load precomputed BM25+Bo1 TREC run file from exp22
- Keep top-1000 candidates per query
- Rerank all 1000 using Qwen3-Reranker-0.6B with P(yes) scoring
- Evaluate MAP@100, MAP@1000, nDCG@10, recall@100

## Key parameters
- Reranker: Qwen/Qwen3-Reranker-0.6B (float16)
- RERANK_TOP_K: 1000 (up from 100 in exp23)
- Batch size: 4
- MAX_CONTENT_TOKENS: 512
- ~249K query-doc pairs to rerank (249 queries x ~1000 docs)

## Expected outcome
- MAP@100 improvement over exp23 (0.2552) due to higher recall ceiling
- First meaningful MAP@1000 measurement
- nDCG@10 similar to exp23 (top-10 ranking mostly from top-100 candidates)
- Longer runtime (~10x more documents to rerank)

## Baseline comparison
- exp23 (rerank top-100): MAP@100=0.2552, nDCG@10=0.5292
- BM25+Bo1 (exp22): MAP@100=0.2504, nDCG@10=0.4662

## Results
- MAP@100=0.2596, MAP@1000=0.3026, nDCG@10=0.5304, recall@100=0.4660
- MAP@100 improved over exp23 (0.2596 vs 0.2552), confirming the recall ceiling hypothesis
- recall@100 improved from 0.4527 to 0.4660 (reranker promoted relevant docs from ranks 100-1000)
- MAP@1000=0.3026, first meaningful deep MAP measurement
- nDCG@10 slightly better (0.5304 vs 0.5292)
- VRAM usage: 2.6 GB (same as exp23)
- New best MAP@100 at the time
