# exp03b-qwen3-rerank-fix

## Goal

Re-run Qwen3-Reranker-0.6B cross-encoder reranking with the CORRECT prompt format. The exp03 implementation was broken: wrong prompt template (missing `<think>` block in suffix, missing colons in format tags) caused catastrophic results (MAP@100=0.013 at depth 1000, 0.128 at depth 100). With the correct format from the official model card, Qwen3-Reranker-0.6B should substantially beat BM25+Bo1 (MAP@100=0.2504).

## Hypothesis

Qwen3-Reranker-0.6B is a strong 0.6B-parameter reranker trained on massive relevance data. The exp03 failure was purely a prompt formatting bug, not a model capability issue. With the correct prompt (proper `<Instruct>:`, `<Query>:`, `<Document>:` tags with colons, and the `<think>\n\n</think>` block in the suffix), the model should produce meaningful relevance scores. We expect it to substantially improve MAP@100 over BM25+Bo1 baseline, likely reaching 0.30+.

## Method

1. Run BM25+Bo1 first-stage retrieval (identical to exp01 baseline) to get top-1000 candidates per query.
2. Load Qwen3-Reranker-0.6B with the CORRECT prompt format from the official model card:
   - System prompt: "Judge whether the Document meets the requirements based on the Query and the Instruct provided."
   - Format: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
   - Suffix includes `<think>\n\n</think>` block
   - Score via log_softmax over yes/no token logits at the last position
3. Rerank BM25 candidates at depth 100 and depth 1000.
4. Evaluate all runs.

Key differences from exp03 (the broken version):
- Prefix/suffix now include the `<think>\n\n</think>` block in suffix
- Format tags use colons: `<Instruct>:`, `<Query>:`, `<Document>:` (exp03 was missing these)
- max_length increased from 512 to 8192 (model supports 32K, Robust04 docs can be long)
- Tokenization: content tokenized separately, then prefix/suffix tokens prepended/appended
- Scoring: log_softmax over [no, yes] logits, take exp of yes probability

## Key Parameters

| Parameter | Value |
|-----------|-------|
| First-stage retriever | BM25+Bo1 (k1=0.9, b=0.4, fb_docs=5, fb_terms=30) |
| First-stage top-K | 1000 |
| Reranker model | Qwen/Qwen3-Reranker-0.6B |
| Rerank depths | 100, 1000 |
| Max input length | 8192 tokens |
| Scoring batch size | 64 |
| Instruction | "Given a web search query, retrieve relevant passages that answer the query" |
| Scoring method | log_softmax over [no, yes] token logits at last position |
| dtype | float16 |
| Training | None (zero-shot) |
| GPU | L40S 46GB |

## Runs

### Run 1: `qwen3-rerank-top100`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-100 with Qwen3-Reranker-0.6B (correct format)
- **Parameter overrides**: rerank_depth=100
- **Expected output**: `runs/qwen3-rerank-top100.run`

### Run 2: `qwen3-rerank-top1000`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-1000 with Qwen3-Reranker-0.6B (correct format)
- **Parameter overrides**: rerank_depth=1000
- **Expected output**: `runs/qwen3-rerank-top1000.run`

## Expected Outcome

| Run | Expected MAP@100 | Rationale |
|-----|-------------------|-----------|
| qwen3-rerank-top100 | ~0.30-0.35 | Strong reranker on high-precision BM25 candidates |
| qwen3-rerank-top1000 | ~0.32-0.38 | Deeper pool gives reranker more relevant docs to promote |

For reference, exp03's BGE reranker (which DID work correctly) achieved:
- top-100: MAP@100=0.2487, nDCG@10=0.5059
- top-1000: MAP@100=0.2431, nDCG@10=0.5046

Qwen3-Reranker should beat BGE since it is a newer, stronger model -- but only with the correct prompt format.

## Baseline Comparison

- **exp01-bm25-baseline**: MAP@100=0.2504 (BM25+Bo1 PRF, current best)
- **exp03 Qwen3 (BROKEN)**: MAP@100=0.013-0.128 (wrong prompt format)
- **exp03 BGE reranker**: MAP@100=0.2487 (correct implementation, did not beat baseline)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training -- zero-shot)
- [x] Training does NOT use Robust04 qrels (no training -- zero-shot)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (pre-trained model used as-is)
