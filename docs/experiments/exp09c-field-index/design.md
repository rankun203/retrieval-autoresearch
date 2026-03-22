# exp09c-field-index: BM25F Field-Aware Indexing on Robust04

## Literature Review

BM25F is the field-aware extension of BM25, well-established in IR:

- **Robertson, Zaragoza, Taylor (2004)** "Simple BM25 extension to multiple weighted fields" — CIKM 2004. Assigns per-field weights and per-field length normalization (b) parameters. Title fields typically get higher weight since title matches are stronger relevance signals.

- **Terrier implementation** — PyTerrier supports field-aware indexing via `IterDictIndexer` with `fields=["title", "text"]`. Models BM25F and PL2F use per-field statistics. BM25F uses `w.0`, `w.1` controls for field weights and `bm25f.b_0`, `bm25f.b_1` for per-field b parameters.

Key implementation detail: The existing Terrier index (`terrier_index`) was built without field statistics (`fields.count=0`). A new index with field support must be built.

## Goal

Test whether field-aware retrieval (BM25F with title boosting) improves over standard BM25 and BM25+PRF on Robust04, and whether field-aware systems can improve the sparse-only CombSUM result (MAP@100=0.2583).

## Hypothesis

Robust04 documents have title and body fields. Title matches are stronger relevance signals. BM25F should improve precision by boosting title matches. The expected gain is modest (1-3%) since BM25+Bo1 is already well-tuned. The main value is providing a diversified sparse system for CombSUM fusion.

## Method

1. Build a new Terrier index with field support (title, text) at `~/.cache/autoresearch-retrieval/terrier_index_fields`
2. Sweep BM25F title weight: w.0 in [1.0, 2.0, 3.0, 5.0, 8.0], w.1=1.0
3. Sweep per-field b parameters for best title weight
4. Test best BM25F + Bo1/KL PRF
5. Test PL2F (field-aware PL2)
6. Fuse best field system with BM25+Bo1 and with CombSUM systems

## Key Parameters

| Parameter | Values |
|-----------|--------|
| Index fields | ["title", "text"] |
| Index cache | `~/.cache/autoresearch-retrieval/terrier_index_fields` |
| BM25F w.0 (title weight) | 1.0, 2.0, 3.0, 5.0, 8.0 |
| BM25F w.1 (body weight) | 1.0 |
| BM25F bm25f.b_0 (title b) | 0.2, 0.4, 0.6 |
| BM25F bm25f.b_1 (body b) | 0.3, 0.4, 0.5 |
| PRF methods | Bo1, KL |
| PRF fb_docs | 3, 5, 10 |
| PRF fb_terms | 10, 20, 30 |
| PL2F c | 1.0, 2.0, 5.0, 10.0 |
| NUM_RESULTS | 1000 |
| Runtime | CPU-only, ~2 hours |

## Runs

### Phase 1: `bm25f-weight-sweep`
- BM25F with title weight sweep, default b parameters
- Output: `runs/bm25f-best-weight.run`

### Phase 2: `bm25f-b-sweep`
- Per-field b parameter sweep for best title weight
- Output: `runs/bm25f-best-tuned.run`

### Phase 3: `bm25f-prf`
- Best BM25F + Bo1/KL PRF sweep
- Output: `runs/bm25f-prf-best.run`

### Phase 4: `pl2f-sweep`
- PL2F with field index, c sweep + PRF
- Output: `runs/pl2f-best.run`

### Phase 5: `fusion`
- Fuse best field systems with BM25+Bo1 baseline
- CombSUM of diverse field-aware + non-field systems
- Output: `runs/best-overall.run`

## Expected Outcome

- BM25F with title boost: MAP@100 ~0.215-0.225 (modest gain over plain BM25 0.2141)
- BM25F + PRF: MAP@100 ~0.250-0.260
- Fusion with existing systems: MAP@100 ~0.260-0.270
- Unlikely to beat dense+sparse best (0.2929) but could improve sparse-only CombSUM baseline

## Baseline Comparison

| System | MAP@100 |
|--------|---------|
| Plain BM25 (k1=0.9, b=0.4) | 0.2141 |
| BM25+Bo1 baseline | 0.2504 |
| CombSUM(all-6) from exp09b | 0.2583 |
| Best overall (Qwen3-8B fusion) | 0.2929 |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, retrieval sweep only)
- [x] Training does NOT use Robust04 qrels (only used in final evaluate_run)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (no model training)
