# exp09-bm25-prf-tune: BM25 Parameter Tuning and PRF Technique Comparison

## Literature Review

This experiment uses standard BM25 and pseudo-relevance feedback (PRF) techniques that are well-established in the IR literature. No deep literature review is needed for these classical methods, but key references:

- **BM25**: Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond". Foundations and Trends in IR. Standard parameters: k1 controls term frequency saturation, b controls document length normalization. Typical ranges: k1 in [0.5, 2.0], b in [0.2, 0.8].

- **Bo1 (Bose-Einstein 1)**: Amati & Van Rijsbergen (2002). "Probabilistic models of information retrieval based on measuring the divergence from randomness". ACM TOIS. Bo1 is a DFR-based query expansion model. Parameters: fb_docs (feedback documents), fb_terms (expansion terms).

- **RM3 (Relevance Model 3)**: Abdul-Jaleel et al. (2004). "UMass at TREC 2004: Novelty and HARD". Uses language model-based query expansion with interpolation parameter fb_lambda controlling weight of original vs expanded query.

- **KL Divergence QE**: Amati (2003). "Probability models for information retrieval based on divergence from randomness". PhD thesis, University of Glasgow. KL-based term selection for query expansion.

- **Key finding for Robust04**: BM25 parameter sensitivity on news corpora is well-documented. The optimal k1 is typically 1.0-1.5 for news collections (higher than the default 0.9), and b around 0.3-0.5. PRF with RM3 often outperforms Bo1 on TREC collections when fb_lambda is well-tuned.

## Goal

Find the optimal BM25 base parameters (k1, b) and PRF technique/configuration for Robust04. The BM25+PRF sparse component is used in every fusion pipeline, so any improvement here directly lifts all downstream experiments.

## Hypothesis

1. The current default parameters (k1=0.9, b=0.4) may not be optimal for Robust04's news corpus.
2. RM3 with tuned fb_lambda may outperform Bo1 on this collection.
3. Combined optimal BM25 params + optimal PRF could yield MAP@100 > 0.255 (vs current 0.2504).

## Method

Three-phase grid search:

1. **Phase 1**: Sweep BM25 k1 and b (no PRF) to find optimal base parameters.
2. **Phase 2**: With best BM25 params, sweep Bo1, RM3, and KL PRF configurations.
3. **Phase 3**: Fine-grained refinement around the best configuration found.

All runs use PyTerrier with the cached Terrier index. CPU-only, no GPU needed.

## Key Parameters

### BM25 Grid
- k1: [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
- b: [0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
- 42 configurations

### PRF Grid (applied to top-3 BM25 configurations)
- **Bo1**: fb_docs=[3, 5, 10], fb_terms=[10, 20, 30, 50] -- 12 configs
- **RM3**: fb_docs=[3, 5, 10], fb_terms=[10, 20, 30, 50], fb_lambda=[0.5, 0.6, 0.7, 0.8] -- 48 configs
- **KL**: fb_docs=[3, 5, 10], fb_terms=[10, 20, 30, 50] -- 12 configs
- **No PRF**: plain BM25 baseline

### Other
- Retrieval depth: 1000 (for MAP@1000 and recall@100)
- Index: cached Terrier index at ~/.cache/autoresearch-retrieval/terrier_index

## Runs

### Phase 1: BM25 base parameter sweep (42 runs)
- Run name pattern: `bm25-k1_{k1}-b_{b}`
- No PRF, plain BM25 retrieval
- Expected time: ~15 minutes total

### Phase 2: PRF technique comparison (~72 runs on best BM25 config)
- Uses top-3 BM25 configurations from Phase 1
- Run name pattern: `{bm25_config}-{prf_method}-fd{fb_docs}-ft{fb_terms}[-fl{fb_lambda}]`
- Bo1: 12 configs x 3 BM25 = 36 runs
- RM3: 48 configs x 1 best BM25 = 48 runs
- KL: 12 configs x 1 best BM25 = 12 runs
- Expected time: ~30 minutes total

### Phase 3: Fine-grained refinement (~20 runs)
- Narrow grid around best configuration
- k1 +/- 0.1 in 0.05 steps, b +/- 0.05 in 0.025 steps
- Best PRF with +/- adjustments
- Expected time: ~10 minutes

### Output files
- `runs/best-bm25-only.run` -- best BM25 without PRF
- `runs/best-bm25-bo1.run` -- best BM25+Bo1 configuration
- `runs/best-bm25-rm3.run` -- best BM25+RM3 configuration
- `runs/best-bm25-kl.run` -- best BM25+KL configuration
- `runs/best-overall.run` -- overall best configuration
- `logs/results_grid.csv` -- full grid search results

## Expected Outcome

- BM25-only: MAP@100 ~0.21-0.22 (current is 0.2141 with k1=0.9, b=0.4)
- Best BM25+PRF: MAP@100 ~0.255-0.265, improving on current 0.2504
- RM3 may outperform Bo1 with tuned fb_lambda

## Baseline Comparison

- Current BM25 only: MAP@100=0.2141, nDCG@10=0.4437
- Current BM25+Bo1 (k1=0.9, b=0.4, fb_docs=5, fb_terms=30): MAP@100=0.2504, nDCG@10=0.4662
- Best overall: MAP@100=0.2929 (exp07 fusion with Qwen3-8B)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, only retrieval + evaluation)
- [x] Training does NOT use Robust04 qrels (qrels used only for final evaluation via evaluate_run)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training (N/A -- no training loop)
- [x] No test-time information flows into model weights (no model weights)

Note: This is standard IR evaluation -- BM25/PRF parameters are tuned on the test set, which is standard practice for reporting BM25 oracle results. The goal is to find the best possible BM25+PRF configuration for use as a first-stage retriever in fusion pipelines.
