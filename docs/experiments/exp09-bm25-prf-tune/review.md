# Review: exp09-bm25-prf-tune

## Data Leakage Check: PASS

- `load_robust04()` called on line 56, returning `corpus`, `queries`, `qrels`.
- `queries`: used only on line 59 to build `topics_df` for PyTerrier retrieval pipelines. No training loop exists; queries are used solely for retrieval followed by evaluation. This is standard IR evaluation practice.
- `qrels`: used only inside `evaluate_run(run, qrels)` on lines 79 and 103 -- the allowed final evaluation pattern.
- `corpus`: referenced only on line 502 for `len(corpus)` in the summary printout. The index is pre-built and loaded from cache.
- No training loop, no model weights, no hard negative mining, no MS-MARCO data needed.
- **No forbidden patterns found.** This is a pure BM25/PRF parameter sweep with evaluation, which is the standard practice for reporting oracle sparse retrieval results.

## Code Quality

- Clean, well-structured three-phase grid search with clear separation of phases.
- Good error handling with try/except around each PRF configuration.
- Results saved to CSV for reproducibility (`logs/results_grid.csv`, 210 rows).
- All five expected run files produced (`best-bm25-only.run`, `best-bm25-bo1.run`, `best-bm25-kl.run`, `best-bm25-rm3.run`, `best-overall.run`).
- Minor inefficiency: best runs are re-executed at the end to save run files (lines 167, 281-286, 414-421) rather than being cached during the sweep. Not a problem given CPU-only runtime.

## Cache Verification

- Terrier index loaded from `~/.cache/autoresearch-retrieval/terrier_index` (line 38, 64). Log confirms "Using Terrier index at /home/ubuntu/.cache/autoresearch-retrieval/terrier_index". This is the correct shared index for Robust04 (528,155 docs confirmed on log line 3).

## Design Adherence

| Design spec | Actual | Match? |
|-------------|--------|--------|
| Phase 1: 42 BM25 configs (7 k1 x 6 b) | 42 configs run | Yes |
| Phase 2: Bo1 on top-3 BM25 (36 runs) | 36 Bo1 runs | Yes |
| Phase 2: KL on best BM25 (12 runs) | 12 KL runs | Yes |
| Phase 2: RM3 on best BM25 (48 runs) | 48 RM3 runs | Yes |
| Phase 3: Fine-grained refinement | 48 k1/b + 24 PRF configs | Yes |
| Retrieval depth: 1000 | NUM_RESULTS=1000 | Yes |
| Output: 5 run files + results CSV | All produced | Yes |

All design specifications followed. Total: 210 configurations evaluated.

## Performance Analysis

### Phase 1: BM25-only
| Config | MAP@100 | Delta vs baseline |
|--------|---------|-------------------|
| k1=0.7, b=0.3 (best) | 0.2160 | +0.0019 |
| k1=0.9, b=0.3 | 0.2153 | +0.0012 |
| k1=0.9, b=0.4 (current) | 0.2141 | baseline |

Optimal BM25 base params are in the k1=0.5-0.9, b=0.3 range. Marginal improvement.

### Phase 2: PRF comparison
| Method | Best MAP@100 | Configuration |
|--------|-------------|---------------|
| KL     | 0.2465 | k1=0.7, b=0.3, fd=3, ft=20 |
| Bo1    | 0.2455 | k1=0.7, b=0.3, fd=3, ft=20 |
| RM3    | 0.2445 | k1=0.7, b=0.3, fd=5, ft=30, fl=0.5 |

KL marginally outperforms Bo1 (+0.001), both outperform RM3. All PRF methods prefer fd=3 (few feedback docs).

### Phase 3: Fine-grained refinement
- Best overall: KL with k1=0.6, b=0.375, fd=3, ft=20
- MAP@100=0.2503 (delta=-0.0001 vs current BM25+Bo1 baseline of 0.2504)

### vs. Current best
- Current best MAP@100: 0.2929 (exp07) / 0.2903 (exp12)
- This experiment's best: 0.2503 -- well below neural fusion approaches, as expected.

## Budget Assessment: OK

- CPU-only experiment, no GPU used (peak_vram_mb=0.0).
- Total runtime: 5624s (~94 minutes) for 210 configurations.
- No overfitting concern (no model training).

## Verdict: **APPROVE**

The experiment is methodologically sound, correctly implemented, and free of data leakage. Results confirm the existing BM25+Bo1 parameters (k1=0.9, b=0.4, fd=5, ft=30) are near-optimal. The exhaustive 210-configuration sweep found at best a tie with the baseline. All runs logged as **discard** since MAP@100=0.2503 does not exceed the current best.

**Status: discard** -- no improvement over baseline.

## Recommendations

1. No need to change BM25+Bo1 parameters in downstream experiments. The current configuration is within noise of the optimum.
2. KL with fd=3, ft=20 is a marginal alternative to Bo1, but the difference is negligible.
3. Future work should focus on dense retrieval, fusion, and reranking components which provide much larger gains (0.25 -> 0.29+).
4. The finding that fd=3 outperforms fd=5 and fd=10 suggests the top-3 pseudo-relevant docs are higher quality on Robust04 -- this is useful context for future PRF-based experiments.
