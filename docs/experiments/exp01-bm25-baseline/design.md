# exp01-bm25-baseline

## Goal

Establish BM25 baselines on Robust04 using PyTerrier. We run two variants:
1. Plain BM25 (no query expansion)
2. BM25 + Bo1 pseudo-relevance feedback (PRF)

These serve as the floor for all subsequent dense retrieval experiments.

## Hypothesis

BM25 is a strong lexical baseline. Bo1 query expansion should improve MAP by
10-20% over plain BM25 by expanding queries with terms from top-ranked documents.
Expected MAP@100 for BM25+Bo1 is around 0.25 based on literature.

## Method

- Load Robust04 corpus, queries, qrels via `prepare.py`
- Build a Terrier index (cached on disk after first run)
- Run 1: Plain BM25 with standard parameters
- Run 2: BM25 + Bo1 PRF pipeline (retrieve -> expand -> re-retrieve)
- Evaluate both runs and report metrics

No training is involved -- this is a zero-shot lexical baseline.

## Key Parameters

| Parameter      | Run 1 (BM25)   | Run 2 (BM25+Bo1) |
|----------------|-----------------|-------------------|
| BM25 k1        | 0.9             | 0.9               |
| BM25 b         | 0.4             | 0.4               |
| Top-K          | 1000            | 1000              |
| Bo1 fb_docs    | N/A             | 5                 |
| Bo1 fb_terms   | N/A             | 30                |
| Training time  | 0               | 0                 |
| GPU required   | No              | No                |

## Runs

### Run: `bm25-plain`
- **Description**: Plain BM25 retrieval, no query expansion
- **Parameter overrides**: No Bo1 step
- **Output**: `runs/bm25-plain.run`

### Run: `bm25-bo1`
- **Description**: BM25 followed by Bo1 query expansion and re-retrieval
- **Parameter overrides**: Bo1 with fb_docs=5, fb_terms=30
- **Output**: `runs/bm25-bo1.run`

## Expected Outcome

- **BM25 plain**: MAP@100 ~ 0.20-0.23, nDCG@10 ~ 0.40-0.43
- **BM25+Bo1**: MAP@100 ~ 0.24-0.26, nDCG@10 ~ 0.43-0.46

These estimates are based on standard Robust04 literature values.

## Baseline Comparison

This IS the baseline. All future experiments compare against these numbers.

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training at all)
- [x] Training does NOT use Robust04 qrels (no training at all)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (no model weights)
