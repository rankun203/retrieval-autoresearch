# exp03-cross-encoder-rerank

## Goal

Improve MAP@100 over the BM25+Bo1 baseline (0.2504) by adding zero-shot cross-encoder reranking on top of BM25+Bo1 first-stage retrieval.

## Hypothesis

Cross-encoders jointly encode query-document pairs and capture fine-grained token interactions that BM25 cannot. Pre-trained cross-encoders (trained on MS-MARCO) should generalize well to Robust04 in a zero-shot setting, significantly improving precision metrics. We expect MAP@100 to reach 0.30+ since cross-encoders are the standard approach for boosting retrieval quality on top of lexical first-stage retrievers.

## Method

1. Run BM25+Bo1 first-stage retrieval (same as exp01 baseline) to get top-1000 candidates per query.
2. For each query, score all candidate documents using a cross-encoder model.
3. Re-sort candidates by cross-encoder score.
4. Evaluate the reranked results.

We test three cross-encoder models of increasing capability:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` -- small, fast, well-established (~80MB)
- `BAAI/bge-reranker-v2-m3` -- strong multilingual reranker (~1.1GB)
- `Qwen/Qwen3-Reranker-0.6B` -- newest, strongest, needs special handling for EOS/last-token pooling (~1.2GB)

We also compare reranking depth: top-100 vs top-1000 BM25 candidates.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| First-stage retriever | BM25+Bo1 (k1=0.9, b=0.4, fb_docs=5, fb_terms=30) |
| First-stage top-K | 1000 |
| Cross-encoder models | ms-marco-MiniLM-L-6-v2, bge-reranker-v2-m3, Qwen3-Reranker-0.6B |
| Rerank depths | 100, 1000 |
| Max input length | 512 tokens (cross-encoder default) |
| Scoring batch size | 64 (MiniLM), 32 (bge-reranker), 16 (Qwen3-Reranker) |
| Training | None (zero-shot) |
| GPU | L40S 46GB |

## Runs

### Run 1: `minilm-rerank-top100`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-100 with cross-encoder/ms-marco-MiniLM-L-6-v2
- **Parameter overrides**: rerank_depth=100, model=cross-encoder/ms-marco-MiniLM-L-6-v2, batch_size=64
- **Expected output**: `runs/minilm-rerank-top100.run`

### Run 2: `minilm-rerank-top1000`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-1000 with cross-encoder/ms-marco-MiniLM-L-6-v2
- **Parameter overrides**: rerank_depth=1000, model=cross-encoder/ms-marco-MiniLM-L-6-v2, batch_size=64
- **Expected output**: `runs/minilm-rerank-top1000.run`

### Run 3: `bge-rerank-top100`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-100 with BAAI/bge-reranker-v2-m3
- **Parameter overrides**: rerank_depth=100, model=BAAI/bge-reranker-v2-m3, batch_size=32
- **Expected output**: `runs/bge-rerank-top100.run`

### Run 4: `bge-rerank-top1000`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-1000 with BAAI/bge-reranker-v2-m3
- **Parameter overrides**: rerank_depth=1000, model=BAAI/bge-reranker-v2-m3, batch_size=32
- **Expected output**: `runs/bge-rerank-top1000.run`

### Run 5: `qwen3-rerank-top100`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-100 with Qwen/Qwen3-Reranker-0.6B
- **Parameter overrides**: rerank_depth=100, model=Qwen/Qwen3-Reranker-0.6B, batch_size=16
- **Expected output**: `runs/qwen3-rerank-top100.run`

### Run 6: `qwen3-rerank-top1000`
- **Description**: BM25+Bo1 top-1000 retrieval, rerank top-1000 with Qwen/Qwen3-Reranker-0.6B
- **Parameter overrides**: rerank_depth=1000, model=Qwen/Qwen3-Reranker-0.6B, batch_size=16
- **Expected output**: `runs/qwen3-rerank-top1000.run`

## Expected Outcome

| Run | Expected MAP@100 | Rationale |
|-----|-------------------|-----------|
| minilm-rerank-top100 | ~0.28-0.30 | MiniLM is decent but small; reranking top-100 limits recall |
| minilm-rerank-top1000 | ~0.29-0.31 | More candidates gives cross-encoder more to work with |
| bge-rerank-top100 | ~0.30-0.33 | Stronger model, should improve over MiniLM |
| bge-rerank-top1000 | ~0.31-0.35 | Best combination of model quality and candidate depth |
| qwen3-rerank-top100 | ~0.30-0.34 | Newest model, potentially strongest |
| qwen3-rerank-top1000 | ~0.32-0.36 | Should be the best overall |

Cross-encoder reranking should provide a substantial improvement over BM25+Bo1 alone. The key question is how much, and whether larger models and deeper reranking depths justify the compute cost.

## Baseline Comparison

- **exp01-bm25-baseline**: MAP@100=0.2504 (BM25+Bo1 PRF, no reranking)
- All runs should beat this baseline significantly.

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training at all -- zero-shot)
- [x] Training does NOT use Robust04 qrels (no training at all -- zero-shot)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (pre-trained models used as-is)
