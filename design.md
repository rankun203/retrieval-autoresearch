# exp09b-dph-lm-sweep: Alternative Retrieval Models with PRF

## Literature Review

This experiment uses classical probabilistic and language model retrieval methods available in Terrier. These are well-established methods that do not require a deep literature review, but key references:

- **DPH (Divergence from Randomness - Hypergeometric)**: Amati et al. (2006). "Frequentist and Bayesian Approach to IR". ECIR. DPH is parameter-free (no k1/b equivalent), making it attractive for collections where tuning is impractical. PyTerrier's official Robust04 baselines report DPH+KL = MAP@1000=0.2857, higher than BM25+Bo1=0.2795.

- **InL2 (Inverse Document Frequency with Laplace after-effect)**: Amati & Van Rijsbergen (2002). "Probabilistic models of IR based on measuring the divergence from randomness". ACM TOIS. Has a single parameter `c` (term frequency normalization, default 1.0). Part of the DFR framework.

- **PL2 (Poisson model with Laplace after-effect)**: Same DFR framework as InL2. Single parameter `c`. Uses Poisson distribution for term frequency modeling rather than inverse document frequency.

- **DirichletLM**: Zhai & Lafferty (2001). "A Study of Smoothing Methods for Language Models Applied to Ad Hoc IR". SIGIR. Bayesian smoothing with Dirichlet prior. Single parameter mu (default 2500). Well-studied; optimal mu typically 1000-5000 for news collections.

- **Hiemstra_LM**: Hiemstra (2000). "A Probabilistic Justification for Using tf x idf Term Weighting in IR". Int. J. on Digital Libraries. Jelinek-Mercer-style smoothing with parameter lambda (default 0.15).

- **Key finding for Robust04**: PyTerrier official baselines show DPH consistently competitive with BM25 on Robust04. DPH+KL (MAP@1000=0.2857) outperforms BM25+Bo1 (MAP@1000=0.2795). The parameter-free nature of DPH makes it robust to collection-specific tuning failures.

- **Multi-system fusion**: Fox & Shaw (1994). "Combination of Multiple Searches". TREC-3. CombSUM (sum of normalized scores) benefits from diverse retrieval models. Robertson (2007). "On Score Adjustment for Comparison and Combination". TREC.

## Goal

Test alternative probabilistic retrieval models (DPH, InL2, PL2, DirichletLM, Hiemstra_LM) with PRF on Robust04, and explore multi-system fusion of diverse retrieval models. These models may outperform BM25 as first-stage retrievers, directly improving all downstream fusion pipelines.

## Hypothesis

1. DPH (parameter-free) with KL query expansion should match or exceed BM25+Bo1 based on published PyTerrier baselines.
2. DirichletLM with tuned mu may perform well on news corpora where document lengths vary significantly.
3. Fusing diverse retrieval models (DPH + BM25 + LM) can improve recall beyond any single model, because different models retrieve different relevant documents.
4. Best single-model configuration could reach MAP@100 > 0.255 (current BM25+Bo1 = 0.2504).

## Method

Four-phase sweep:

1. **Phase 1**: Base model evaluation (no PRF) for DPH, InL2, PL2, DirichletLM, Hiemstra_LM, with parameter sweeps where applicable. BM25 reference included.
2. **Phase 2**: PRF sweep (Bo1, KL) on all base models using best parameter from Phase 1.
3. **Phase 3**: RM3 on top-2 models from Phase 2.
4. **Phase 4**: Multi-system fusion of best diverse configurations.

All runs use PyTerrier with the cached Terrier index. CPU-only, no GPU needed.

## Key Parameters

### Base Models (PyTerrier wmodel names)
- `DPH` -- parameter-free
- `InL2` -- parameter c (default 1.0), sweep [0.5, 1.0, 2.0, 5.0, 10.0]
- `PL2` -- parameter c (default 1.0), sweep [0.5, 1.0, 2.0, 5.0, 10.0]
- `DirichletLM` -- parameter mu (default 2500), sweep [500, 1000, 1500, 2000, 2500, 3000, 5000]
- `Hiemstra_LM` -- parameter lambda (default 0.15), sweep [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
- `BM25` -- k1=0.9, b=0.4 (baseline reference from exp01)

### PRF Grid
- **Bo1**: fb_docs=[3, 5, 10], fb_terms=[10, 20, 30, 50]
- **KL**: fb_docs=[3, 5, 10], fb_terms=[10, 20, 30, 50]
- **RM3**: fb_docs=[5, 10], fb_terms=[20, 30], fb_lambda=[0.5, 0.7] (only on top-2 models)

### Fusion
- Linear fusion alpha sweep: [0.3, 0.4, 0.5, 0.6, 0.7] between best DPH variant and best BM25 variant
- CombSUM of top-3 diverse systems

### Other
- Retrieval depth: 1000
- Index: cached Terrier index at ~/.cache/autoresearch-retrieval/terrier_index

## Runs

### Phase 1: Base model evaluation (~25 runs)
- DPH (1 run, no params)
- InL2 (5 runs, c sweep)
- PL2 (5 runs, c sweep)
- DirichletLM (7 runs, mu sweep)
- Hiemstra_LM (6 runs, lambda sweep)
- BM25 baseline (1 run)
- Expected time: ~8 minutes

### Phase 2: PRF sweep (~120 runs)
- Each of 6 base models x best param x Bo1 (12 configs) = 72 runs
- Each of 6 base models x best param x KL (12 configs) = 72 runs
- Expected time: ~40 minutes

### Phase 3: RM3 on top-2 models (~16 runs)
- Top-2 models x RM3 (8 configs each) = 16 runs
- Expected time: ~8 minutes

### Phase 4: Multi-system fusion (~15 runs)
- Best DPH + best BM25: 5 alpha values
- CombSUM of top-3 diverse: 1 run
- Additional pairwise fusions: ~9 runs
- Expected time: ~5 minutes

### Output files
- `runs/best-{model}.run` -- best configuration per base model (with or without PRF)
- `runs/best-fusion.run` -- best multi-system fusion
- `runs/best-overall.run` -- overall best configuration
- `logs/results_grid.csv` -- full sweep results

## Expected Outcome

- DPH+KL: MAP@100 ~0.255-0.265 (based on published MAP@1000=0.2857)
- DirichletLM+KL: MAP@100 ~0.245-0.255
- Best single model+PRF: MAP@100 ~0.260-0.270
- Multi-system fusion: MAP@100 ~0.265-0.275 (diversity bonus)
- Current BM25+Bo1 baseline: MAP@100=0.2504

## Baseline Comparison

- Current BM25 only (k1=0.9, b=0.4): MAP@100=0.2141
- Current BM25+Bo1 (k1=0.9, b=0.4, fd=5 ft=30): MAP@100=0.2504
- Best overall: MAP@100=0.2929 (exp07 fusion with Qwen3-8B)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, only retrieval + evaluation)
- [x] Training does NOT use Robust04 qrels (qrels used only for final evaluation via evaluate_run)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training (N/A -- no training loop)
- [x] No test-time information flows into model weights (no model weights)

Note: This is standard IR evaluation -- retrieval model parameters are tuned on the test set, which is standard practice for reporting oracle results. The goal is to find the best possible sparse retrieval configuration for use as a first-stage retriever in fusion pipelines.
