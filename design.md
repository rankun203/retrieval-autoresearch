# exp22-bm25-prf: BM25 + Bo1 PRF Baseline via PyTerrier

## Goal
Establish a strong sparse retrieval baseline using BM25 with Bo1 pseudo-relevance feedback (PRF) on Robust04, to compare against and complement the dense retrieval approach.

## Hypothesis
Classical BM25 with query expansion via Bo1 PRF should provide a competitive baseline on Robust04, a traditional newswire collection well-suited to lexical methods. This baseline also provides first-stage candidates for neural reranking experiments.

## Method
- Use PyTerrier's native BM25 retriever with Bo1 query expansion
- Pipeline: BM25 first pass -> Bo1 query expansion (top docs feedback) -> BM25 second pass with expanded query
- Tuned BM25 parameters (k1, b) and Bo1 parameters (fbDocs, fbTerms) across two configurations
- Added Porter stemming in the final configuration

## Key parameters
- BM25: k1=0.9, b=0.4
- Bo1 PRF: fbDocs=5, fbTerms=30
- Porter stemming enabled (final config)
- No GPU required (CPU-only pipeline)

## Expected outcome
- MAP@100 in the 0.20-0.30 range (BM25+PRF is strong on Robust04)
- Provides candidate pool for downstream reranking experiments

## Baseline comparison
- Dense-only baseline (exp15 e5-base-v2): MAP@100=0.1772, nDCG@10=0.4421
- This is the first sparse retrieval experiment in the project

## Results
Three configurations tested:
1. **BM25(k1=0.9,b=0.4)+Bo1(5,30)**: MAP@100=0.2504, nDCG@10=0.4662, recall@100=0.4527 -- **keep**
2. **BM25(k1=1.2,b=0.75)+Bo1(10,40)**: MAP@100=0.2304, nDCG@10=0.4373 -- discard (worse)
3. **BM25(k1=0.9,b=0.4)+Bo1(5,30)+Porter stem**: MAP@100=0.2433, nDCG@10=0.4547 -- **keep** (stemming slightly worse MAP but kept for variant)

Best config (MAP@100=0.2504) substantially outperformed dense-only retrieval (0.1772) and established the BM25+Bo1 baseline used by all subsequent reranking and hybrid experiments.
