# exp30-hard-negatives: 2-Phase Hard Negative Mining

## Goal
Improve the dense retriever by mining hard negatives from the target corpus (Robust04) and using them for a second training phase, then evaluate with hybrid RRF fusion and optional reranking.

## Hypothesis
A bi-encoder trained only on MS-MARCO produces embeddings poorly calibrated for Robust04's domain. By using the phase-1 model to retrieve from Robust04 and collecting top-ranked non-relevant documents as hard negatives, phase-2 training can teach the model to distinguish difficult cases specific to the target corpus. This domain adaptation via hard negative mining should substantially improve dense retrieval quality.

## Method
1. **Phase 1 (300s)**: Train e5-base-v2 on MS-MARCO with InfoNCE loss (symmetric in-batch negatives)
2. **Mine hard negatives**: Encode all 528K Robust04 docs with phase-1 model, retrieve top-100 per query, collect non-relevant docs as hard negatives (10 per query) and relevant docs as positives
3. **Phase 2 (300s)**: Continue training with mixed batches -- 70% MS-MARCO + 30% Robust04 hard negative triples (query, relevant doc, hard negative doc)
4. **Evaluate**: Encode corpus with phase-2 model, build FAISS index, retrieve top-1000
5. **Fuse with BM25+Bo1** via RRF (k=60) and optionally rerank top-100 with Qwen3-Reranker

## Key parameters
- Encoder: intfloat/e5-base-v2, batch=64, LR=1e-5, temp=0.05
- Phase 1: 300s MS-MARCO, Phase 2: 300s mixed (total 600s)
- Mining: top-100 candidates, 10 hard negatives per query, top-3 hardest per positive
- ROBUST04_BATCH_RATIO: 0.3 (30% of phase-2 batches from Robust04)
- RRF k=60 for BM25+dense fusion
- Reranker: Qwen/Qwen3-Reranker-0.6B, RERANK_TOP_K=100

## Expected outcome
- Dense-only MAP@100 improvement over exp15 (0.1772) from domain adaptation
- Hybrid RRF MAP@100 should exceed exp27 (0.2675)

## Baseline comparison
- exp27 (hybrid RRF + Qwen3-Reranker): MAP@100=0.2675, nDCG@10=0.5441
- exp15 (dense-only, no HN mining): MAP@100=0.1772
- BM25+Bo1 (exp22): MAP@100=0.2504

## Results
Three evaluation configurations from the same trained model:

1. **Dense-only**: MAP@100=0.2358 (+33% over exp15's 0.1772, confirming HN mining helps dense)
2. **Hybrid RRF (no rerank)**: MAP@100=0.3275, nDCG@10=0.5921, recall@100=0.5577 -- **NEW BEST**, milestone 3 achieved
3. **Hybrid RRF + Qwen3-Reranker top-100**: MAP@100=0.3149 -- WORSE than RRF alone (reranker degrades good fusion)

Key findings:
- Hard negative mining dramatically improved the dense retriever (+33% MAP@100)
- The improved dense model made RRF fusion much more effective (0.3275 vs exp27's 0.2675)
- Critically, the Qwen3-Reranker HURT performance when applied to the already-strong RRF fusion (0.3149 vs 0.3275), suggesting the reranker is not reliably better than RRF ranking for these candidates
- This discovery shifted the project strategy away from reranking and toward improving the dense retriever directly
