# exp09d-prf-factorial: Factorial BM25 x PRF Parameter Search

## Goal

Find the globally optimal BM25 + PRF parameter combination for each PRF technique (Bo1, KL, RM3) by doing a proper joint/factorial sweep. The previous exp09 found BM25 params without PRF first, then tested PRF on those params -- but the optimal BM25 params may differ per PRF technique.

## Hypothesis

The optimal BM25 (k1, b) parameters interact with PRF settings. A joint sweep will find better configurations than the sequential approach used in exp09. Specifically:
- PRF-heavy configurations (many fb_docs, many fb_terms) may prefer lower k1 and b to allow the expanded query to dominate
- Light PRF (few fb_docs, few fb_terms) should converge to the no-PRF optimum
- Two-round PRF (double expansion) may further improve the best single-round configs

## Method

CPU-only experiment using the existing cached Terrier index. For each PRF technique, sweep ALL combinations of BM25 params and PRF params jointly:

1. **Phase 1 - Bo1 factorial**: 16 BM25 x 30 PRF = 480 runs
2. **Phase 2 - KL factorial**: 16 BM25 x 30 PRF = 480 runs
3. **Phase 3 - RM3 factorial**: 16 BM25 x 96 PRF = 1536 runs
4. **Phase 4 - Two-round PRF**: Top-5 configs per technique, double expansion
5. **Phase 5 - High fb_docs**: Top-3 configs per technique with fb_docs in [25, 30, 50]

Total: ~2500+ runs at ~15-25s each = ~12-17 hours.

## Key Parameters

### BM25 grid (shared across all PRF techniques)
- k1: [0.5, 0.7, 0.9, 1.2]
- b: [0.2, 0.3, 0.4, 0.5]
- Total: 4 x 4 = 16 configs

### Bo1/KL PRF grid
- fb_docs: [3, 5, 10, 15, 20]
- fb_terms: [10, 20, 30, 50, 75, 100]
- Total: 5 x 6 = 30 configs per technique

### RM3 PRF grid
- fb_docs: [3, 5, 10, 15]
- fb_terms: [10, 20, 30, 50]
- fb_lambda: [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
- Total: 4 x 4 x 6 = 96 configs

### Other
- num_results: 1000
- Index: existing cached Terrier index at ~/.cache/autoresearch-retrieval/terrier_index

## Runs

### Run 1: `bo1-factorial`
- Full factorial BM25 x Bo1 sweep (480 configs)
- Output: logs/factorial_results.csv, runs/best-bo1.run

### Run 2: `kl-factorial`
- Full factorial BM25 x KL sweep (480 configs)
- Output: appended to logs/factorial_results.csv, runs/best-kl.run

### Run 3: `rm3-factorial`
- Full factorial BM25 x RM3 sweep (1536 configs)
- Output: appended to logs/factorial_results.csv, runs/best-rm3.run

### Run 4: `two-round-prf`
- Top-5 configs per technique with double expansion (BM25>>PRF>>BM25>>PRF>>BM25)
- Output: runs/best-2round-*.run if improved

### Run 5: `high-fb-docs`
- Top-3 configs per technique with fb_docs in [25, 30, 50]
- Output: additional rows in CSV

## Expected Outcome

- Current BM25+Bo1 baseline: MAP@100 = 0.2504 (k1=0.9, b=0.4, fd=5, ft=30)
- Current best sparse-only: MAP@100 = 0.2583 (CombSUM of 6 models from exp09b)
- Expected: find a single BM25+PRF config reaching MAP@100 ~0.255-0.260
- Confirm whether BM25 params truly interact with PRF settings
- Two-round PRF may push slightly further

## Baseline Comparison

- exp01 BM25+Bo1: MAP@100 = 0.2504 (k1=0.9, b=0.4, fd=5, ft=30)
- exp09 best BM25+KL: MAP@100 = 0.2503
- exp09b best single: MAP@100 = 0.2514 (InL2+KL fd=3 ft=20)
- exp09b CombSUM: MAP@100 = 0.2583 (fusion of 6 models)
- Overall best: MAP@100 = 0.2929 (Qwen3-Embedding-8B + BM25+Bo1 fusion)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, CPU-only sweep)
- [x] Training does NOT use Robust04 qrels (qrels used only for final evaluation)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A)
- [x] `evaluate_run()` called only for evaluation, not during training (no training)
- [x] No test-time information flows into model weights (no model training)
