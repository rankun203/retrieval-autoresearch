# mar16 Series: Dense Retrieval Hyperparameter Exploration

## Goal
Systematically explore encoder architectures, training hyperparameters, and reranking strategies to maximize dense retrieval performance on Robust04, starting from a MiniLM-L6-v2 baseline.

## Hypothesis
A series of incremental improvements -- better encoder, tuned temperature, symmetric negatives, optimized sequence lengths, and cross-encoder reranking -- should compound to substantially outperform the initial baseline. Each change targets a specific bottleneck identified in the previous iteration.

## Method
Iterative experimentation on a bi-encoder trained with InfoNCE loss on MS-MARCO triples, evaluated on Robust04 (249 topics). Each experiment modifies one or two variables from the previous best, following a systematic sweep:
1. **Encoder selection**: MiniLM-L6-v2 -> e5-small-v2 (with query/passage prefixes) -> e5-base-v2
2. **Loss function tuning**: temperature from 0.02 to 0.05, symmetric in-batch negatives
3. **Learning rate**: 2e-5 -> 1e-5 (conservative fine-tuning)
4. **Sequence lengths**: MAX_DOC_LEN 180->220, MAX_QUERY_LEN 64->96 (for verbose TREC topics)
5. **Cross-encoder reranking**: MiniLM-L-6-v2 cross-encoder on top-K dense results (K=100, 1000, 200)

## Key parameters (final best configuration, exp19)
- Encoder: intfloat/e5-base-v2 (mean pooling, 768-dim)
- Batch size: 64, LR: 1e-5, temperature: 0.05
- MAX_QUERY_LEN: 96, MAX_DOC_LEN: 220
- Symmetric InfoNCE loss (bidirectional query-passage and passage-query)
- Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2, RERANK_TOP_K=200
- Training budget: 600s on MS-MARCO triples
- FAISS flat inner product index on GPU

## Expected outcome
- Progressive improvement from each change
- Cross-encoder reranking should give the largest single boost (rerankers are stronger than bi-encoders)

## Baseline comparison
- Starting point: MiniLM-L6-v2, batch=128, temp=0.02, no prefixes, no reranking

## Results

| Exp | Change | nDCG@10 | MAP@100 | Status |
|-----|--------|---------|---------|--------|
| baseline | MiniLM-L6-v2 InfoNCE batch=128 | 0.3521 | 0.1260 | keep |
| exp1 | e5-small-v2 + query/passage prefixes | 0.3978 | 0.1499 | keep |
| exp5 | temperature=0.05 | 0.3996 | - | keep |
| exp6 | symmetric in-batch negatives | 0.4109 | - | keep |
| exp7 | LR=1e-5 | 0.4129 | - | keep |
| exp8 | e5-base-v2 batch=64 | 0.4256 | - | keep |
| exp12 | MAX_DOC_LEN=220 | 0.4365 | - | keep |
| exp15 | MAX_QUERY_LEN=96 | 0.4421 | 0.1772 | keep |
| exp16 | cross-encoder rerank top-100 | 0.4743 | 0.1996 | keep |
| exp17 | rerank top-1000 | 0.4745 | 0.2220 | keep |
| exp19 | RERANK_TOP_K=200 | 0.4775 | 0.2106 | keep |

Key findings:
- Encoder upgrade (MiniLM -> e5-small -> e5-base) gave the largest cumulative dense-only gain
- Symmetric in-batch negatives and temperature tuning each added ~1-2% nDCG@10
- Longer sequence lengths helped significantly for Robust04's verbose newswire documents
- Cross-encoder reranking boosted nDCG@10 from 0.442 to 0.477 (+8%)
- Top-1000 reranking improved MAP@100 (0.222 vs 0.200) but top-200 was best for nDCG@10
- Several experiments were discarded: cosine warmup, weight decay, CLS pooling, L-12 reranker
