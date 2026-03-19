# Review: exp01-bm25-baseline

## Data Leakage Check: PASS

- `load_robust04()` called at line 32, returning `corpus`, `queries`, `qrels`.
- `corpus`: Used only for building the Terrier index (line 46-49). ALLOWED.
- `queries`: Used only to construct `topics_df` for retrieval (line 70). This is the standard evaluation-time query input, not training. ALLOWED.
- `qrels`: Used only in `evaluate_run(run, qrels)` at line 97, the final evaluation step. ALLOWED.
- No training occurs in this experiment (zero-shot lexical baseline).
- No hard negative mining. No model weights. No gradient updates.

**Verdict: No data leakage detected.**

## Code Quality

- Clean, well-structured script with clear sections.
- Proper use of `prepare.py` utilities (`load_robust04`, `evaluate_run`, `write_trec_run`).
- Index caching under `~/.cache/` avoids redundant rebuilds.
- Summary block prints all required metrics in the expected format.
- Minor: `TOP_K = 1000` retrieves 1000 docs but only top-100 are evaluated for MAP@100. This is fine -- the full 1000 are used for Bo1 feedback and MAP@1000.

No issues found.

## Design Adherence

| Design spec | Actual | Match? |
|-------------|--------|--------|
| BM25 k1=0.9, b=0.4 | k1=0.9, b=0.4 | Yes |
| Bo1 fb_docs=5, fb_terms=30 | fb_docs=5, fb_terms=30 | Yes |
| TOP_K=1000 | 1000 | Yes |
| Run name: bm25-bo1-default | bm25-bo1-default.run | Yes |
| Expected MAP@100: 0.25-0.30 | 0.2504 | Yes (low end) |
| Expected nDCG@10: 0.40-0.45 | 0.4662 | Yes (slightly above) |
| Expected recall@100: 0.35-0.45 | 0.4527 | Yes (upper end) |

Full adherence to design.

## Performance Analysis

| Metric | Value |
|--------|-------|
| nDCG@10 | 0.466202 |
| MAP@1000 | 0.296779 |
| MAP@100 | 0.250376 |
| recall@100 | 0.452732 |
| Eval duration | 22.6s |
| Peak VRAM | 0.0 MB (CPU-only) |

These results are consistent with published BM25+PRF baselines on Robust04. For reference, the literature reports BM25 alone at MAP ~0.25 and Bo1 PRF adding 5-15% relative improvement. The MAP@1000 of 0.297 is in line with published numbers (e.g., Anserini BM25+RM3 reports ~0.29-0.30).

This is the first experiment, so there is no prior result to compare against.

## Budget Assessment

OK -- no training budget required for a lexical baseline.

## Verdict: APPROVE

This is a clean, well-executed BM25+Bo1 baseline. No data leakage, correct implementation, and results within expected ranges. As the first experiment, this establishes the reference point (MAP@100 = 0.2504) for all future experiments.

## Recommendations

- Future dense retrieval experiments should target MAP@100 > 0.25 to beat this baseline.
- Consider a BM25-only run (without Bo1) to measure the exact PRF contribution.
- The gap between MAP@100 (0.250) and MAP@1000 (0.297) suggests that expanding beyond top-100 captures additional relevant documents -- future experiments retrieving only top-100 should note this.
