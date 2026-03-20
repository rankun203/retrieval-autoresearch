# Review: exp10-backbone-sweep

## Data Leakage Check: PASS

Verified every reference to `qrels`, `queries`, and `load_robust04` in `train.py`:

- **`load_robust04()`** (line 101): Returns `corpus`, `queries`, `qrels`.
- **`corpus`**: Used for document text extraction (lines 104-110) and encoding (line 302). Corpus is allowed.
- **`queries`**: Used to build BM25 topics (line 159) and encode query embeddings (lines 340-341). This is a zero-shot experiment -- all query usage is part of the final retrieval/evaluation pipeline. No training occurs.
- **`qrels`**: Used exclusively in `evaluate_run()` calls at lines 403 and 415. Final evaluation only. ALLOWED.
- **No training**: No `stream_msmarco_triples()`, no gradient updates, no hard negative mining, no loss computation. All models are used zero-shot.

No data leakage detected.

## Code Quality

**Good:**
- Clean model configuration dict (lines 53-84) makes the sweep easy to extend.
- Proper memory management: model, tokenizer, embeddings, and FAISS index freed after each model (lines 367-369, 396).
- Embedding caching via `build_cache_key` avoids redundant computation across reruns.
- BM25+Bo1 run cached to JSON for reuse across models.
- Consistent pooling dispatch (mean vs CLS) per model config.
- L2 normalization applied to all embeddings before inner product search.

**Minor issues:**
- `import faiss` inside the loop (line 373) and `import json` inside conditional branches (lines 154, 175) -- functional but could be at top level.
- The summary block (lines 428-468) only reports the single "best" run's metrics in the structured `---` format. The review extracts per-run metrics from the earlier log output instead.

## Cache Verification

From the log, all embeddings were encoded fresh (no stale caches used):
- e5-large-v2: Cached to `.cache/embeddings_intfloat_e5-large-v2_dataset-robust04_max_length-512_pooling-mean/` -- model name, dataset, pooling all correct.
- bge-base-en-v1.5: Cached to `.cache/embeddings_BAAI_bge-base-en-v1.5_dataset-robust04_max_length-512_pooling-cls/` -- CLS pooling correctly reflected.
- modernbert-embed-base: Cached to `.cache/embeddings_nomic-ai_modernbert-embed-base_dataset-robust04_max_length-512_pooling-mean/` -- correct.
- BM25+Bo1: Ran fresh (23.0s), cached to disk. Parameters match design.

PASS.

## Design Adherence

Design specified 6 runs (3 models x 2 variants). All 6 produced:

| Run file | Present | Matches design |
|----------|---------|---------------|
| dense-e5-large-v2.run | Yes | Yes |
| fusion-e5-large-v2.run | Yes | Yes |
| dense-bge-base-en-v1.5.run | Yes | Yes |
| fusion-bge-base-en-v1.5.run | Yes | Yes |
| dense-modernbert-embed-base.run | Yes | Yes |
| fusion-modernbert-embed-base.run | Yes | Yes |

Model-specific configurations (prefixes, pooling, batch sizes, dimensions) all match design.md. Fusion alpha=0.3 as specified. No reranking, as specified. PASS.

## Performance Analysis

| Run | MAP@100 | nDCG@10 | MAP@1000 | Recall@100 |
|-----|---------|---------|----------|------------|
| dense-e5-large-v2 | 0.193507 | 0.455338 | 0.222558 | 0.380684 |
| **fusion-e5-large-v2** | **0.280164** | **0.511255** | **0.331360** | **0.495554** |
| dense-bge-base-en-v1.5 | 0.190147 | 0.458975 | 0.219666 | 0.366498 |
| fusion-bge-base-en-v1.5 | 0.272627 | 0.502061 | 0.324631 | 0.488385 |
| dense-modernbert-embed-base | 0.183781 | 0.452095 | 0.211400 | 0.350109 |
| fusion-modernbert-embed-base | 0.272272 | 0.503303 | 0.323515 | 0.481600 |

**vs. current best (exp07 Qwen3-8B fusion, MAP@100 = 0.2929):**
- Best run here: fusion-e5-large-v2 at 0.2802, which is 0.0127 below the current best.
- None of the 6 runs exceed 0.2929.

**vs. design expectations:**
- All results fall within the predicted ranges in design.md. No anomalies.

**vs. prior baselines:**
- e5-large-v2 dense (0.1935) vs e5-base-v2 dense (0.1697): +14% from scaling up, as expected.
- fusion-e5-large-v2 (0.2802) slightly exceeds Qwen3-0.6B fusion (0.2762), showing a 335M encoder model can match a 600M decoder embedding model in fusion.

**Key finding:** Fusion uplift is remarkably consistent across all three models (+0.087 to +0.089 MAP@100), confirming the BM25+Bo1 fusion pipeline is the primary driver and backbone quality differences are secondary in this zero-shot setting.

## Budget Assessment: OK

Zero-shot experiment, no training. Total wall time 4424.6s (~74 min). Peak VRAM 6315.3 MB (6.2 GB). Reasonable.

## Verdict: **APPROVE**

All 6 runs completed successfully. No data leakage. Results are plausible and within expected ranges. No run exceeds the current best MAP@100 of 0.2929, so all 6 are logged as `discard`.

### Status assignments:
- All 6 runs: `discard` (none exceed MAP@100 = 0.2929)

## Recommendations

1. Further backbone sweeps of small/medium models are unlikely to beat 0.2929 in zero-shot mode. Focus on fine-tuning or advanced retrieval techniques.
2. e5-large-v2 could be a strong candidate for domain adaptation / fine-tuning experiments since it achieves the best fusion result (0.2802) at much lower compute cost than Qwen3-8B.
3. The consistent ~0.087 fusion uplift suggests exploring whether learned fusion weights or attention-based fusion could improve over linear interpolation.
4. The tight clustering of fusion results (0.272-0.280) despite varying backbone quality (0.184-0.194 dense) confirms diminishing returns from backbone improvement alone. The BM25 signal dominates in fusion.
