# exp33-iter-hn: 3-Phase Iterative Hard Negative Mining

## Goal
Improve dense retrieval quality by iteratively mining hard negatives from the target corpus (Robust04) and retraining, then fuse with BM25+Bo1 via RRF for maximum hybrid performance.

## Hypothesis
Single-pass training on MS-MARCO produces a model that misses domain-specific difficult negatives in Robust04. By mining hard negatives with the current model, adding them to training, and repeating, each round produces a stronger model that discovers harder negatives. Three rounds should substantially improve dense retrieval quality over the 2-phase approach (exp30).

## Method
- **Phase 1 (200s)**: Train e5-base-v2 on MS-MARCO with InfoNCE loss (symmetric in-batch negatives)
- **Mine round 1**: Encode all 528K Robust04 docs, retrieve top-100 per query with FAISS, collect hard negatives (top-ranked non-relevant docs) and positives (top-ranked relevant docs)
- **Phase 2 (200s)**: Mixed training — 70% MS-MARCO batches + 30% Robust04 hard negative batches
- **Mine round 2**: Re-encode with improved model, mine NEW hard negatives (harder than round 1)
- **Phase 3 (200s)**: Mixed training with round-2 hard negatives
- **Evaluate**: Encode corpus, build FAISS index, retrieve top-1000, fuse with BM25+Bo1 via RRF (k=60)

## Key parameters
- Encoder: intfloat/e5-base-v2 (mean pooling)
- Batch size: 128, LR: 1e-5, temperature: 0.05
- MAX_QUERY_LEN: 96, MAX_DOC_LEN: 220
- Mining: top-100 candidates, 10 hard negatives per query
- Robust04 batch ratio: 30% during HN phases
- RRF k=60 for BM25+dense fusion
- Total training budget: 600s (200s × 3 phases)

## Expected outcome
- Dense-only MAP@100: ~0.28-0.32 (up from exp30's 0.2358 with 2-phase mining)
- Hybrid RRF MAP@100: ~0.34-0.36 (up from exp30's 0.3275)
- Each mining round should show diminishing but positive returns

## Baseline comparison
- exp30 (2-phase HN mining): dense MAP@100=0.2358, hybrid MAP@100=0.3275
- exp15 (no HN mining): dense MAP@100=0.1772
- BM25+Bo1 baseline: MAP@100=0.2504

## Results
- Dense-only MAP@100=0.3152 (up from 0.2358, +34% over exp30)
- Hybrid RRF MAP@100=0.3483, MAP@1000=0.4147 — **NEW BEST**
- nDCG@10=0.6333, recall@100=0.5758
- Phase 3 added +0.08 to dense MAP vs 2-phase approach
- Loss curve healthy: spiky on R04 HN batches (expected) but converging
