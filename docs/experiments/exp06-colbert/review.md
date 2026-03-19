# Review: exp06-colbert

## Data Leakage Check: PASS

Verified every reference to `qrels`, `queries`, and `load_robust04` in `train.py`:

- **`load_robust04()`** (line 70): Returns `corpus`, `queries`, `qrels`.
- **`corpus`**: Used for document text retrieval (lines 73-79), encoding documents (lines 305-306). Corpus is allowed.
- **`queries`**: Used to encode query embeddings (lines 327-329), build BM25 topics (line 110), and format reranker inputs (line 566). All are retrieval/evaluation uses. There is NO training in this experiment (zero-shot only), so no test information flows into model weights.
- **`qrels`**: Used exclusively in `evaluate_run()` calls at lines 384, 442, 493, and 599. All are final evaluation. ALLOWED.
- **No training**: No `stream_msmarco_triples()`, no gradient updates, no hard negative mining. All models are used zero-shot.

No data leakage detected.

## Code Quality: CRITICAL IMPLEMENTATION BUGS

### Bug 1: ColBERTv2 linear projection head not loaded (CRITICAL)

The code uses `AutoModel.from_pretrained(COLBERT_MODEL)` at line 135, which instantiates `BertModel`. The ColBERTv2 checkpoint contains a `linear.weight` parameter (the trained 768->128 dim projection head), but `BertModel` has no `linear` attribute. The load report in the log confirms the weight was silently discarded:

```
linear.weight                | UNEXPECTED |
```

As a result, `has_linear = False` (log line 24), and the code falls back to the truncation path at line 200:

```python
embs = hidden[:, :, :COLBERT_DIM]  # fallback: truncate
```

This takes the first 128 dimensions of BERT's 768-dim hidden state, which are NOT trained ColBERT embeddings. They are arbitrary slices of the BERT representation with no meaningful similarity structure for MaxSim scoring. This single bug completely invalidates all ColBERT-based scoring in this experiment.

**Fix**: Use a custom `nn.Module` that wraps `BertModel` and includes a `nn.Linear(768, 128)`, then load the full state dict. Or load `linear.weight` separately from the checkpoint with `torch.load()`.

### Bug 2: [Q]/[D] marker tokens not found in vocabulary

Log line 26: `Using [unused0]/[unused1] as Q/D markers`. The ColBERTv2 tokenizer does not register `[Q]` and `[D]` as recognized tokens by default when loaded via `AutoTokenizer`. The fallback to `[unused0]`/`[unused1]` means the marker token embeddings are not the ones trained during ColBERTv2 pre-training, adding noise to query/document representations. This is secondary to Bug 1 but contributes to degraded performance.

### Code Style Notes

- Overall structure is clean and well-organized with clear phase separation.
- Memory management with chunked processing and disk caching is good practice.
- The Qwen3-Reranker integration is correct (reused proven pattern from exp03b/exp04).
- The fusion and min-max normalization logic is correct.

## Design Adherence

The design specified 4 runs and all 4 were produced:

| Run | Produced | Matches Design |
|-----|----------|----------------|
| colbert-rerank-bm25 | Yes | Yes |
| colbert-fullretrieval | Yes | Yes |
| colbert-bm25-fusion | Yes | Yes |
| colbert-fusion-reranked | Yes | Yes |

All parameters match design.md specifications (DOC_MAXLEN=180, QUERY_MAXLEN=32, ENCODE_BATCH=256, FUSION_ALPHA=0.3, RERANKER_MAX_LENGTH=768, etc.).

## Performance Analysis

| Run | MAP@100 | nDCG@10 | MAP@1000 | Recall@100 | vs BM25+Bo1 (0.2504) |
|-----|---------|---------|----------|------------|----------------------|
| colbert-rerank-bm25 | 0.1194 | 0.3082 | 0.1613 | 0.3002 | -52.3% |
| colbert-fullretrieval | 0.0809 | 0.2311 | 0.0935 | 0.1924 | -67.7% |
| colbert-bm25-fusion | 0.2438 | 0.4718 | 0.2874 | 0.4452 | -2.6% |
| colbert-fusion-reranked | 0.2792 | 0.5411 | 0.3221 | 0.4900 | +11.5% |

**Current best**: MAP@100 = 0.2827 (exp05-hybrid-rrf). None of these runs beat it.

### Root Cause of Poor ColBERT Performance

The ColBERT standalone results are dramatically below expectations. Published ColBERTv2 achieves MRR@10 of ~39.7 on MS-MARCO dev, and zero-shot transfer to Robust04 should yield MAP@100 in the 0.20-0.30 range. The observed 0.0809 MAP@100 for full retrieval is a clear indicator of broken embeddings, directly attributable to Bug 1 (missing projection head).

Supporting evidence:
- ColBERT reranking of BM25 top-1000 (0.1194) is WORSE than BM25 alone (0.2504), meaning ColBERT scoring is actively harmful -- consistent with garbage embeddings.
- The fusion with BM25 (0.2438) partially recovers because BM25 carries 70% of the weight and the broken ColBERT signal (30%) is partially diluted by min-max normalization.
- The Qwen3-Reranker (0.2792) further recovers by rescoring with a competent model, achieving near the current best (0.2827). The small gap from 0.2827 is likely due to the ColBERT-corrupted fusion providing a slightly worse candidate pool than the Qwen3-Embedding fusion used in exp05.

**This is an implementation failure, not a methodology failure.** ColBERT late interaction remains a promising approach and should be retried with correct model loading.

## Budget Assessment: OK

- No training performed (zero-shot).
- Total runtime: 5450s (~91 minutes), dominated by Qwen3-Reranker (4726s).
- Peak VRAM: 21.5 GB (21461 MB).
- Budget assessment from log: OK.

## Verdict: **APPROVE** (all runs as `discard`)

Despite critical implementation bugs in ColBERT model loading, there is no data leakage, the experiment completed cleanly, and all 4 runs produced valid (though degraded) results. All runs should be logged as `discard` since none beats the current best MAP@100 of 0.2827.

### Status assignments:
- **colbert-rerank-bm25**: `discard` (MAP@100=0.1194, broken ColBERT embeddings)
- **colbert-fullretrieval**: `discard` (MAP@100=0.0809, broken ColBERT embeddings)
- **colbert-bm25-fusion**: `discard` (MAP@100=0.2438, below current best)
- **colbert-fusion-reranked**: `discard` (MAP@100=0.2792, below current best 0.2827)

## Recommendations

1. **Fix model loading (highest priority)**: Load ColBERTv2 with a custom model class that includes the linear projection head. Verify `has_linear = True` before proceeding with any encoding.

2. **Add embedding sanity check**: Before full corpus encoding, compute MaxSim between a known relevant query-doc pair and an irrelevant pair. If scores are similar, abort early.

3. **Consider using the colbert-ai library**: The `colbert-ai` package handles model loading, query augmentation ([MASK] padding), marker tokens, and MaxSim correctly. Using it avoids reimplementation bugs.

4. **Fix marker tokens**: Register `[Q]` and `[D]` in the tokenizer or use the exact token IDs from the ColBERTv2 config.

5. **Retry is warranted**: A correctly implemented ColBERT + BM25 fusion + Qwen3-Reranker pipeline has strong potential to beat 0.2827. The Qwen3-Reranker already achieved 0.2792 with broken first-stage input; proper ColBERT retrieval should provide a much better candidate pool.
