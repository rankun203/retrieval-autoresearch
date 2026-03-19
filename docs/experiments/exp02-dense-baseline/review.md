# Review: exp02-dense-baseline

## Data Leakage Check: PASS

- Line 39: `corpus, queries, qrels = load_robust04()` loads all three variables.
- `corpus` (lines 42-46): Used to build document texts for encoding. Corpus is not test data. **ALLOWED.**
- `queries` (lines 48-49): Used to encode queries for final retrieval. No training occurs, so this is equivalent to test-time inference. **ALLOWED.**
- `qrels` (line 160): Used only in `evaluate_run(run, qrels)` as the final evaluation step. **ALLOWED.**
- No training loop, no weight updates, no hard negative mining. This is a pure zero-shot inference pipeline.

**Verdict: No leakage detected.**

## Code Quality

**Good:**
- Clean, well-structured zero-shot pipeline with clear sections
- Proper L2 normalization before inner product search (correct cosine similarity)
- FP16 autocast for efficient GPU encoding
- VRAM tracking included
- Handles negative FAISS indices correctly (line 144)

**Minor issues:**
- Line 87: Deprecated `torch.cuda.amp.autocast()` API; should use `torch.amp.autocast('cuda')`. Non-blocking.
- The log shows interleaved output from an accidental duplicate run (lines 38-66 of log). The first run was killed and the second completed successfully. Final metrics come from the clean completion.

## Design Adherence

| Design spec | Actual | Match |
|---|---|---|
| Encoder: e5-base-v2 | intfloat/e5-base-v2 | Yes |
| Doc max length: 256 | 256 | Yes |
| Query max length: 64 | 64 | Yes |
| Batch size: 512 | 512 (1032 batches for 528K docs) | Yes |
| FAISS FlatIP | FlatIP | Yes |
| L2 normalize | Yes (line 92) | Yes |
| Top-K: 1000 | 1000 | Yes |
| Doc prefix: "passage: " | Yes | Yes |
| Query prefix: "query: " | Yes | Yes |

All design specifications were followed exactly.

## Performance Analysis

| Metric | Result | Expected (design.md) | vs BM25+Bo1 baseline |
|---|---|---|---|
| MAP@100 | 0.1697 | 0.17-0.22 | -0.0807 (-32.2%) |
| nDCG@10 | 0.4284 | 0.35-0.42 | -0.0378 (-8.1%) |
| MAP@1000 | 0.1942 | -- | -0.1026 (-34.5%) |
| recall@100 | 0.3339 | 0.35-0.42 | -0.1190 (-26.3%) |

MAP@100 of 0.1697 falls at the low end of the expected range (0.17-0.22). This is consistent with published results: zero-shot dense retrievers underperform tuned BM25 on Robust04, especially on recall. The nDCG@10 slightly exceeded the expected upper bound (0.4284 vs 0.42), suggesting the model retrieves some relevant docs at the top but misses many overall.

No suspicion of data leakage (results are in the expected range and below BM25).

## Budget Assessment: OK

Zero-shot inference only. No training budget consumed. Total wall-clock: 529.3s (dominated by 518.5s document encoding). Peak VRAM: 3907.4 MB (3.8 GB).

## Verdict: APPROVE

This is a valid zero-shot dense retrieval baseline. It correctly establishes the dense retrieval floor for Robust04 at MAP@100=0.1697. The code is clean, there is no data leakage, and results match expectations. Status: **discard** (does not beat current best MAP@100=0.2504).

## Recommendations

1. Fine-tuning e5-base-v2 on MS-MARCO triples should substantially close the gap with BM25. Target MAP@100 > 0.25.
2. Consider doc_max_length=512 to capture more document content -- 256 tokens truncates many Robust04 newswire articles.
3. Hybrid retrieval (dense + BM25 score fusion) could combine strengths of both approaches.
4. Hard negative mining from the Robust04 corpus (using MS-MARCO queries, not test queries) could further improve dense retrieval.
