# exp27-hybrid-fusion: Dense + BM25 RRF Fusion with Qwen3-Reranker

## Goal
Combine dense retrieval (e5-base-v2) and sparse retrieval (BM25+Bo1) via Reciprocal Rank Fusion (RRF), then rerank the fused top-100 with Qwen3-Reranker-0.6B to exploit complementary retrieval signals.

## Hypothesis
Dense and sparse retrievers find different relevant documents: BM25 excels at exact term matching while dense models capture semantic similarity. Fusing their ranked lists via RRF should increase recall beyond either system alone. Applying the Qwen3 reranker on the higher-recall fused pool should yield better MAP@100 than reranking either system's output alone.

## Method
1. **Train e5-base-v2** bi-encoder on MS-MARCO (600s, InfoNCE with symmetric in-batch negatives)
2. **Dense retrieval**: Encode 528K Robust04 docs, build FAISS GPU index, retrieve top-1000 per query
3. **Load BM25+Bo1** precomputed run file (top-1000 per query from exp22)
4. **RRF fusion**: Merge dense and BM25 ranked lists with Reciprocal Rank Fusion (k=60)
5. **Rerank top-100** from the fused list using Qwen3-Reranker-0.6B with P(yes) scoring
6. Evaluate all stages: dense-only, fused, and fused+reranked

## Key parameters
- Dense encoder: intfloat/e5-base-v2, batch=64, LR=1e-5, temp=0.05
- MAX_QUERY_LEN: 96, MAX_DOC_LEN: 220
- RRF k=60 (standard fusion parameter)
- Reranker: Qwen/Qwen3-Reranker-0.6B, RERANK_TOP_K=100
- RERANK_BATCH: 4, MAX_CONTENT_TOKENS: 512

## Expected outcome
- Fused recall@100 should exceed both dense and BM25 individually
- Fused+reranked MAP@100 should exceed exp24 (0.2596, BM25-only + reranker)
- Dense-only MAP@100 should match exp15 (~0.177)

## Baseline comparison
- exp24 (BM25+Bo1 + Qwen3-Reranker top-1000): MAP@100=0.2596, nDCG@10=0.5304
- exp15 (dense-only e5-base-v2): MAP@100=0.1772
- BM25+Bo1 alone (exp22): MAP@100=0.2504

## Results
- **Hybrid RRF + Qwen3-Reranker top-100**: MAP@100=0.2675, nDCG@10=0.5441, recall@100=0.4843
- New best MAP@100 (0.2675) and nDCG@10 (0.5441) at the time
- Fused recall@100 (0.4843) exceeded both BM25 (0.4527) and dense alone, confirming complementarity
- Dense-only MAP@100=0.1801 (consistent with exp15)
- The hybrid approach validated the fusion + rerank paradigm that became the standard pipeline for subsequent experiments
