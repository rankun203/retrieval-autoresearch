# Review: exp14-splade

## Data Leakage Check: PASS

Thorough line-by-line trace of `train.py`:

1. **`qrels` usage** (line 72): `corpus, queries, qrels = load_robust04()`. The `qrels` variable appears only in `evaluate_run(...)` calls at lines 457, 473, 498, 513 -- all in the final evaluation section. No use during encoding, retrieval, or fusion weight selection. CLEAN.

2. **`queries` usage** (line 72): The `queries` dict is used at:
   - Line 112: Building `topics_df` for BM25+Bo1 retrieval (evaluation-time retrieval, not training)
   - Lines 229-230: Encoding queries with SPLADE for retrieval (evaluation-time retrieval)
   - Line 329: Encoding queries with Qwen3-0.6B for dense retrieval (evaluation-time retrieval)

   All query usage is for final retrieval and evaluation. There is no training loop in this experiment (zero-shot). CLEAN.

3. **No training**: This is a zero-shot experiment. Pre-trained SPLADE model is loaded and used as-is. No fine-tuning, no hard negative mining, no gradient updates. No training-time information flow.

4. **Corpus usage** (lines 75-81, 159): Corpus text is loaded for SPLADE document encoding. Corpus is not test data. CLEAN.

**Verdict: PASS** -- No data leakage detected.

## Code Quality

- Well-structured pipeline with clear separation of stages (BM25, SPLADE encoding, dense retrieval, fusion, evaluation)
- Proper caching of SPLADE document embeddings with metadata
- Memory management with explicit `gc.collect()` and `torch.cuda.empty_cache()` after model use
- Chunked encoding (10K docs per chunk) to manage memory for 528K docs
- Handles multiple sparse tensor formats (torch sparse, scipy sparse, dense)
- Min-max normalization before fusion is appropriate for combining different score distributions
- Minor note: design.md mentions "splade-v3" but code uses `splade-cocondenser-ensembledistil` due to gating -- this is documented in the config (line 45 comment)

## Cache Verification

- **SPLADE doc embeddings**: Encoded fresh (2521.4s). Cache path: `.cache/splade_embeddings_naver_splade-cocondenser-ensembledistil_dataset-robust04_max_length-256`. Includes correct model name, dataset, and max_length. CORRECT.
- **Dense embeddings**: Loaded from cache at `.cache/embeddings_Qwen_Qwen3-Embedding-0.6B_dataset-robust04_max_length-512_pooling-last_token`. Verified 528,155 embeddings, dim=1024. CORRECT.
- **Terrier index**: Loaded from existing cache at `~/.cache/autoresearch-retrieval/terrier_index`. CORRECT.

## Design Adherence

| Design Spec | Actual | Match? |
|-------------|--------|--------|
| SPLADE model: splade-v3 | splade-cocondenser-ensembledistil | PARTIAL (v3 gated, documented) |
| Doc max length: 256 | 256 | YES |
| Encode batch: 64 | 64 | YES |
| BM25+Bo1 config | k1=0.9, b=0.4, fb_docs=5, fb_terms=30 | YES |
| Dense model: Qwen3-0.6B | Qwen3-0.6B (cached) | YES |
| Run 1: SPLADE standalone | Completed | YES |
| Run 2: SPLADE+BM25 fusion sweep | 3 alpha values tested | YES |
| Run 3: Triple fusion sweep | 5 weight configs tested | YES |
| Run 4: SPLADE+dense only | Completed | YES |

All 10 planned runs completed successfully.

## Performance Analysis

### Results Summary

| Run | MAP@100 | nDCG@10 | Recall@100 |
|-----|---------|---------|------------|
| splade-v3-zeroshot | 0.2158 | 0.4630 | 0.4100 |
| splade-bm25-fusion-a03 | 0.2684 | 0.4908 | 0.4811 |
| splade-bm25-fusion-a05 | 0.2662 | 0.5056 | 0.4779 |
| splade-bm25-fusion-a07 | 0.2471 | 0.4967 | 0.4500 |
| triple-fusion-s02-d02-b06 | 0.2817 | 0.5123 | 0.4958 |
| triple-fusion-s03-d02-b05 | 0.2814 | 0.5237 | 0.4916 |
| **triple-fusion-s02-d03-b05** | **0.2830** | **0.5274** | **0.4942** |
| triple-fusion-s015-d025-b06 | 0.2822 | 0.5184 | 0.4982 |
| triple-fusion-s01-d03-b06 | 0.2814 | 0.5192 | 0.4982 |
| splade-dense-fusion-a05 | 0.2422 | 0.5163 | 0.4361 |

### vs Current Best

- Current best MAP@100: 0.2929 (exp07, Qwen3-Embedding-8B + BM25 linear fusion)
- Best this experiment: 0.2830 (triple fusion SPLADE+dense+BM25)
- Delta: -0.0099 (3.4% below current best)

### vs Expectations from Design

- SPLADE standalone: 0.2158 (within expected 0.18-0.24) -- domain mismatch confirmed
- SPLADE+BM25 fusion: 0.2684 (within expected 0.26-0.28)
- Triple fusion: 0.2830 (within expected 0.28-0.30)
- All expectations met.

### Key Findings

1. **SPLADE standalone underperforms BM25+Bo1** (0.2158 vs 0.2504): MS MARCO-trained SPLADE does not transfer well to news domain. Recall@100 of 0.41 is notably lower than BM25+Bo1's 0.4527.

2. **SPLADE+BM25 fusion helps** but best (0.2684) is below simple dense+BM25 fusion (0.2762 from exp05). SPLADE's contribution to fusion is weaker than dense embeddings.

3. **Triple fusion does not beat current best**: Best triple at 0.2830 is below exp12's dual fusion with doc expansion (0.2903). Adding a third retriever provided diminishing returns.

4. **SPLADE did not break the recall@100 plateau**: Best recall@100 in this experiment is 0.4982, below current best of 0.5103 (exp07).

5. **Model choice limitation**: splade-cocondenser-ensembledistil (MRR@10 38.3 on MS MARCO) is weaker than the gated splade-v3 (MRR@10 40.2). Results might improve slightly with v3, but unlikely to close the gap.

## Budget Assessment: OK

Zero-shot experiment, no training. Total runtime 3229.4s (54 min), dominated by SPLADE doc encoding (2521s). Peak VRAM 8.3 GB. Reasonable resource usage.

## Verdict: **APPROVE**

All runs completed successfully. No data leakage. Results are within expected ranges. None beat current best MAP@100 (0.2929), so all runs are `discard` status.

Logging two representative runs to results.tsv:
1. **splade-v3-zeroshot** -- documents SPLADE zero-shot baseline on Robust04
2. **triple-fusion-s02-d03-b05** -- documents best achievable with SPLADE in the mix

## Recommendations

1. **SPLADE is not competitive on Robust04 without domain adaptation**: The MS MARCO to news domain gap is significant. Fine-tuning SPLADE on domain data might help but is unlikely to surpass dense methods.

2. **Focus future fusion on stronger individual retrievers**: Dense + BM25 fusion is strictly stronger than any SPLADE combination tested. Improving the dense component (domain-adapted embeddings) is a better investment.

3. **Skip learned sparse for Robust04**: Learned sparse retrieval adds complexity without benefit on this benchmark when compared to BM25+Bo1 with PRF.
