# Review: exp12-doc-expansion

## Data Leakage Check: PASS

### generate_queries.py (Phase 1)
- Line 49: `corpus, queries, qrels = load_robust04()` -- loads all three variables, but `queries` and `qrels` are **never referenced** after this line. Only `corpus` is used (lines 52, 88) to iterate over documents and extract text.
- The LLM (Qwen2.5-3B-Instruct) generates synthetic queries purely from document text via a prompt (lines 93-105). No Robust04 test queries or relevance judgments influence generation.
- Verdict: **PASS** -- no leakage in Phase 1.

### train.py (Phase 2)
- Line 85: `corpus, queries, qrels = load_robust04()`
- `queries` used at line 161 (BM25 topics_df for evaluation retrieval) and lines 284-285 (encoding queries for dense evaluation retrieval). Both are standard evaluation-time usage.
- `qrels` used only at lines 343, 358, 412 -- all inside `evaluate_run()` for final metric computation.
- `doc_queries` (line 98) loads LLM-generated synthetic queries from cache -- these are NOT Robust04 test queries.
- Verdict: **PASS** -- no leakage in Phase 2.

## Code Quality

**Strengths:**
- Clean two-phase separation (vLLM generation vs. retrieval) avoids CUDA conflicts
- Resumable generation with JSONL cache and line-count verification (lines 60-82 of generate_queries.py)
- Proper GPU memory cleanup between phases
- Cache key includes all relevant parameters (model, queries_per_doc, temperature, max_doc_chars, dataset)
- Well-structured fusion with min-max normalization per query

**Minor issues:**
- `import re` inside `parse_queries()` function (line 141 of generate_queries.py) -- should be at module level. Functional but inefficient when called 528K times.
- `generate_queries.py` loads `queries` and `qrels` from `load_robust04()` but never uses them. Could destructure as `corpus, _, _` for clarity. Not a correctness issue.

## Cache Verification

- **Expansion cache**: `doc_expansion_Qwen_Qwen2.5-3B-Instruct_dataset-robust04_max_doc_chars-1500_queries_per_doc-5_temperature-0.7` -- path includes correct model name, dataset, and all generation parameters. Log confirms 528,155 docs loaded from cache.
- **Embedding cache**: `embeddings_Qwen_Qwen3-Embedding-0.6B_dataset-robust04-expanded-q5_max_length-512_pooling-last_token` -- path correctly identifies this as expanded corpus embeddings (`robust04-expanded-q5`), distinct from non-expanded embeddings. Log confirms fresh encoding of all 528,155 docs in 5515.8s.
- **Terrier index**: `terrier_index_expanded_q5` -- separate from baseline index. Log confirms cached index was used.

## Design Adherence

| Design spec | Actual | Match? |
|-------------|--------|--------|
| LLM: Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct | Yes |
| 5 queries per doc | Avg 5.0/doc (log) | Yes |
| Temperature 0.7 | 0.7 | Yes |
| BM25 k1=0.9, b=0.4 | 0.9, 0.4 | Yes |
| Bo1 fb_docs=5, fb_terms=30 | 5, 30 | Yes |
| Embedding: Qwen3-Embedding-0.6B | Qwen3-Embedding-0.6B | Yes |
| Fusion alphas 0.3, 0.5 | 0.3, 0.5 | Yes |
| 4 runs (bm25, dense, fusion-a03, fusion-a05) | 4 runs produced | Yes |

All design specifications were followed exactly.

## Performance Analysis

| Run | MAP@100 | nDCG@10 | Recall@100 | vs Baseline |
|-----|---------|---------|------------|-------------|
| bm25-bo1-expanded | 0.2744 | 0.4886 | 0.4805 | +0.0240 vs BM25+Bo1 (0.2504) |
| dense-expanded | 0.2111 | 0.5035 | 0.3901 | +0.0006 vs dense (0.2105) |
| fusion-a03-expanded | 0.2903 | 0.5243 | 0.5057 | +0.0141 vs fusion (0.2762) |
| fusion-a05-expanded | 0.2806 | 0.5462 | 0.4875 | -- |

**Key findings:**
1. **BM25 benefited most** from doc expansion: +0.0240 MAP@100, +0.0278 recall@100. Vocabulary enrichment from synthetic queries directly addresses term mismatch.
2. **Dense retrieval barely improved**: +0.0006 MAP@100, -0.0011 recall@100. The 0.6B model truncates at 512 tokens, so appended queries may be cut off for longer documents, and the model already captures semantic meaning from the original text.
3. **fusion-a03-expanded (0.2903)** is the new best 0.6B-class result, surpassing exp05's fusion+reranker (0.2827) by +0.0076 MAP@100 without needing a reranker.
4. **Recall@100 improved** to 0.5057, up from 0.4920 (exp05 fusion). This creates headroom for future reranking.
5. Still below exp07's 8B result (0.2929) by -0.0026, but exp07 used an 8B model with ~35K seconds encoding time vs ~5.5K seconds here.

## Budget Assessment: OK

- No training involved (zero-shot LLM generation + zero-shot embedding)
- Phase 1 generation: ~280 minutes for 528K docs at ~31 docs/s
- Phase 2 encoding: 5515.8s (~92 min) for 528K docs
- Peak VRAM: 18,875 MB (during embedding encoding)
- Total Phase 2 wall time: 5574.1s (~93 min)

## Verdict: **APPROVE**

**Reasons:**
1. No data leakage -- synthetic queries generated purely from document text, no test queries or qrels used during generation
2. fusion-a03-expanded achieves MAP@100=0.2903, a new best for the 0.6B model class (+0.0076 over exp05)
3. Recall@100=0.5057 creates meaningful headroom for future reranking experiments
4. Clean implementation with proper caching and two-phase separation

**Status assignments:**
- `fusion-a03-expanded`: **keep** -- new 0.6B best MAP@100
- `bm25-bo1-expanded`: **discard** -- useful data point but not best
- `dense-expanded`: **discard** -- minimal improvement
- `fusion-a05-expanded`: **discard** -- lower MAP@100 than a03

## Recommendations

1. **Combine doc expansion with reranking**: The recall@100 improvement (0.4920 -> 0.5057) means a reranker now has more relevant docs to promote. Qwen3-Reranker-0.6B on top of fusion-a03-expanded could push MAP@100 further.
2. **Try more queries per doc**: 5 queries helped BM25 substantially. 10 queries (as in original docTTTTTquery) might help more, at the cost of longer generation time.
3. **Try a larger generation LLM**: Qwen2.5-7B-Instruct might generate higher-quality queries, improving both sparse and dense retrieval.
4. **Dense model sees limited benefit**: Future dense improvements should focus on training/fine-tuning rather than doc expansion, since the 512-token truncation limits how much expansion text the model actually sees.
