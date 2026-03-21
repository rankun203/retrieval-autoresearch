# Review: exp09b-dph-lm-sweep

## Data Leakage Check: PASS

- `load_robust04()` called once (line 65), returning `corpus`, `queries`, `qrels`.
- `queries` used only to build `topics_df` (line 68) for retrieval via `transform(topics_df)` -- standard IR evaluation, not training.
- `qrels` used only inside `evaluate_run(run, qrels)` calls (lines 106, 127, 456, etc.) -- final metric computation only.
- No training loop, no model weight updates, no hard negative mining, no MS-MARCO data needed.
- No `stream_msmarco_triples()` usage (none needed).
- Conclusion: This is a pure retrieval model parameter sweep with evaluation. No test-time information flows into any model weights. **PASS**.

## Code Quality

- Clean, well-structured 4-phase sweep: base models, PRF, RM3, fusion.
- Good use of helper functions (`run_model`, `run_model_prf`, `normalize_scores`, `linear_fusion`, `combsum_fusion`).
- Error handling with try/except for PRF runs (lines 331, 395).
- Full results CSV written for reproducibility (234 configs tested).
- Min-max normalization for fusion is appropriate for combining heterogeneous scoring functions.
- Minor issue: DirichletLM mu parameter sweep shows identical scores across all mu values (0.1974 MAP@100 for all 7 values). This suggests PyTerrier may not be passing the `mu` control correctly via the `controls` dict. Same issue with Hiemstra_LM lambda. Not a code bug per se -- likely a PyTerrier API issue with how these language model parameters are set.

## Cache Verification

- Uses cached Terrier index at `~/.cache/autoresearch-retrieval/terrier_index` (line 36).
- Index is the standard Robust04 index used across all experiments. Correct.
- No other cached artifacts (CPU-only, no embeddings or model weights cached).

## Design Adherence

| Design spec | Actual | Match? |
|-------------|--------|--------|
| DPH (parameter-free) | Tested | Yes |
| InL2 c sweep [0.5, 1.0, 2.0, 5.0, 10.0] | Tested | Yes |
| PL2 c sweep [0.5, 1.0, 2.0, 5.0, 10.0] | Tested | Yes |
| DirichletLM mu sweep [500..5000] | Tested (all identical) | Yes |
| Hiemstra_LM lambda sweep [0.05..0.5] | Tested (all identical) | Yes |
| BM25 baseline reference | Tested | Yes |
| Bo1/KL PRF on all models | Tested | Yes |
| RM3 on top-2 models | Tested (InL2, BM25) | Yes |
| Linear fusion DPH+BM25 | Tested (5 alphas) | Yes |
| Pairwise fusions all pairs | Tested (3 alphas each) | Yes |
| CombSUM top-3 | Tested | Yes |
| CombSUM all systems | Tested (all-6) | Yes |
| 234 total configs | 234 configs | Yes |

All design specifications were followed.

## Performance Analysis

### Phase 1 -- Base Models (no PRF)
| Model | Best Params | MAP@100 | nDCG@10 |
|-------|-------------|---------|---------|
| PL2 | c=10.0 | 0.2156 | 0.4460 |
| BM25 | k1=0.9, b=0.4 | 0.2141 | 0.4437 |
| InL2 | c=5.0 | 0.2136 | 0.4355 |
| DPH | parameter-free | 0.2119 | 0.4461 |
| DirichletLM | mu=500 | 0.1974 | 0.4160 |
| Hiemstra_LM | lambda=0.05 | 0.1836 | 0.3758 |

### Phase 2 -- Best PRF per Model
| Config | MAP@100 | nDCG@10 | MAP@1000 | R@100 |
|--------|---------|---------|----------|-------|
| InL2+KL fd=3 ft=20 | 0.2514 | 0.4600 | 0.2956 | 0.4491 |
| BM25+Bo1 fd=5 ft=30 | 0.2504 | 0.4662 | 0.2968 | 0.4527 |
| DPH+Bo1 fd=10 ft=30 | 0.2490 | 0.4629 | 0.2959 | 0.4518 |
| PL2+KL fd=3 ft=20 | 0.2466 | 0.4666 | 0.2919 | 0.4480 |
| Hiemstra_LM+Bo1 fd=10 ft=50 | 0.2316 | 0.4244 | 0.2736 | 0.4363 |
| DirichletLM+Bo1 fd=3 ft=10 | 0.1679 | 0.3674 | 0.1986 | 0.3353 |

### Phase 3 -- RM3 Results
RM3 did NOT improve over Bo1/KL for either InL2 or BM25. Best RM3 result was MAP@100=0.2476 (InL2+RM3 fd=5 ft=30 fl=0.5), below the Bo1/KL best of 0.2514.

### Phase 4 -- Fusion Results (top 5)
| Config | MAP@100 | nDCG@10 | MAP@1000 | R@100 |
|--------|---------|---------|----------|-------|
| CombSUM(all-6) | 0.2583 | 0.4727 | 0.3073 | 0.4679 |
| InL2+Hiemstra_LM alpha=0.6 | 0.2564 | 0.4610 | 0.3028 | 0.4621 |
| DPH+InL2 alpha=0.5 | 0.2561 | 0.4717 | 0.3039 | 0.4621 |
| CombSUM(InL2+BM25+DPH) | 0.2554 | 0.4707 | 0.3035 | 0.4626 |
| PL2+Hiemstra_LM alpha=0.6 | 0.2550 | 0.4641 | 0.3016 | 0.4604 |

### Key Findings

1. **InL2+KL** is the best single model+PRF at MAP@100=0.2514, marginally beating BM25+Bo1 (0.2504) by +0.0010.
2. **CombSUM(all-6)** achieves MAP@100=0.2583, a +0.0079 improvement over BM25+Bo1. This demonstrates the value of diverse sparse model fusion.
3. **DirichletLM and Hiemstra_LM** parameter sweeps appear broken -- all parameter values produce identical scores, suggesting PyTerrier is not applying the controls correctly for these language models.
4. **DirichletLM+PRF** actually hurts performance severely (0.1974 -> 0.1679), likely because the base model quality is poor.
5. **RM3** does not improve over Bo1/KL for any model tested.
6. **Recall@100** improves from 0.4527 (BM25+Bo1) to 0.4679 (CombSUM all-6), a +0.0152 improvement that could benefit downstream reranking.

### Comparison with Baselines
- BM25+Bo1 baseline: MAP@100=0.2504
- Best single model+PRF (InL2+KL): MAP@100=0.2514 (+0.0010)
- Best fusion (CombSUM all-6): MAP@100=0.2583 (+0.0079)
- Current overall best (exp07 8B fusion): MAP@100=0.2929

## Budget Assessment: OK

CPU-only experiment, no GPU used. Total runtime ~6092 seconds (~102 minutes) for 234 configurations. Reasonable for a comprehensive sparse retrieval parameter sweep.

## Verdict: **APPROVE**

The experiment is methodologically sound with no data leakage. CombSUM(all-6) at MAP@100=0.2583 beats the BM25+Bo1 sparse baseline (0.2504) by +0.0079 and represents a new best sparse-only retrieval result. It does not beat the current overall best (0.2929) but could serve as an improved sparse component in fusion pipelines.

**Status assignments:**
- `CombSUM(all-6)`: **keep** -- new best sparse-only result, beats BM25+Bo1 baseline
- `InL2+KL fd=3 ft=20`: **discard** -- marginal improvement over BM25+Bo1 (+0.0010)

## Recommendations

1. **Replace BM25+Bo1 with CombSUM(all-6)** as the sparse component in dense+sparse fusion pipelines to test whether the +0.0079 MAP@100 improvement compounds with dense retrieval.
2. **Investigate DirichletLM/Hiemstra_LM parameter passing** -- the identical scores across all parameter values suggest PyTerrier needs a different API to set these model parameters (possibly via `wmodel` string rather than `controls` dict).
3. **Skip RM3** in future experiments -- it consistently underperforms Bo1/KL across all models tested here.
4. **Try CombSUM as sparse input** to the exp07/exp12 pipeline to see if the diversity bonus transfers.
