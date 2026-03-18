# exp39-linear-interp: Linear Score Interpolation for Dense+BM25 Fusion

## Goal
Test whether linear score interpolation outperforms RRF(k=60) for fusing dense and BM25 retrieval runs.

## Hypothesis
RRF is rank-based and discards score magnitude information. Linear interpolation (alpha * dense_score + (1-alpha) * bm25_score) preserves score magnitude after normalization, which may better weight high-confidence retrievals. With tuned alpha, this could outperform RRF.

## Method
- Load existing run files (no training needed):
  - Dense: exp33-iter-hn-dense.run (e5-base-v2, 3-phase HN mining)
  - BM25: exp22-bm25-prf.run (BM25+Bo1)
- Parse both into {qid: {doc_id: score}} format
- Per-query min-max normalize scores to [0, 1]
- For alpha in {0.1, 0.2, ..., 0.9}: fused_score = alpha * dense + (1-alpha) * bm25
- Evaluate each alpha with evaluate_run()
- Compare against RRF(k=60) baseline
- Save best fused run

## Key parameters
- Alpha sweep: 0.1 to 0.9 (step 0.1)
- Score normalization: per-query min-max to [0, 1]
- Dense run: exp33 (MAP@100=0.3152 dense-only)
- BM25 run: exp22 (MAP@100=0.2504)
- RRF baseline: k=60 (MAP@100=0.3483)

## Expected outcome
- Best alpha likely 0.5-0.7 (dense is stronger than BM25)
- Could match or slightly beat RRF (0.3483) if score magnitudes are informative
- If worse, confirms RRF's robustness for this task

## Baseline comparison
- RRF(k=60) MAP@100=0.3483 (exp33)
- Dense-only MAP@100=0.3152
- BM25-only MAP@100=0.2504
