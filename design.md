# Experiment: exp07-qwen3-embed-8b

## Goal

Replace Qwen3-Embedding-0.6B with Qwen3-Embedding-8B in the proven hybrid retrieval + reranking pipeline from exp05. The 8B model is the top-ranked model on MTEB multilingual (score 70.58 vs 0.6B's 64.33) and should provide substantially stronger dense retrieval, improving hybrid fusion quality and final reranking results.

## Hypothesis

The Qwen3-Embedding-8B model produces 4096-dimensional embeddings (vs 1024 for 0.6B) from an 8B parameter decoder backbone. Its MTEB retrieval score (70.88 en) is roughly 10% higher than the 0.6B variant (64.64). Since the 0.6B zero-shot dense retrieval achieved MAP@100=0.2105, the 8B model should achieve approximately MAP@100=0.25-0.28. When fused with BM25+Bo1 (alpha=0.3 dense, 0.7 sparse) and reranked by Qwen3-Reranker-0.6B, the stronger first-stage candidates should push MAP@100 above the current best of 0.2827.

### Literature Context

- **Qwen3-Embedding-8B**: 8B params, 36 layers, 4096-dim embeddings, 32K context, last-token pooling, instruction-aware. Uses same `Instruct: ...\nQuery:` format as 0.6B. Supports Matryoshka Representation Learning (MRL) for flexible output dimensions.
- **Hybrid retrieval**: exp05 demonstrated that linear fusion (alpha=0.3) + reranking consistently beats either component alone. The gain from fusion should be even larger with a stronger dense component.

## Method

1. **BM25+Bo1 retrieval**: Reuse cached Terrier index, retrieve top-1000 per query
2. **Qwen3-Embedding-8B dense retrieval**: Zero-shot encoding of all 528K corpus documents and 249 queries using fp16 with flash_attention_2, FAISS flat inner product search, retrieve top-1000. Use full 4096-dim embeddings.
3. **Linear fusion**: alpha=0.3 (dense=0.3, sparse=0.7) -- the best from exp05
4. **Reranking**: Free embedding model from GPU, then load Qwen3-Reranker-0.6B with news-short instruction, ml768, rerank top-1000

### Qwen3-Embedding-8B Usage (from model card)

- Query format: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query_text}`
- Document format: raw text (no instruction)
- Pooling: last token (via `last_token_pool` function)
- Model class: `AutoModel` (not `AutoModelForCausalLM`)
- Normalize embeddings to unit length (cosine similarity)
- padding_side: left (required for last-token pooling)
- Embedding dimension: 4096

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embedding model | Qwen/Qwen3-Embedding-8B | 8B params, 4096-dim |
| Reranker model | Qwen/Qwen3-Reranker-0.6B | Same as exp05 |
| Doc max length | 512 tokens | Balance quality/speed for 528K docs |
| Query max length | 512 tokens | Generous for Robust04 title queries |
| Encode batch size | 64 | ~25GB VRAM with 8B model fp16 |
| BM25 k1/b | 0.9/0.4 | From exp01 |
| Bo1 fb_docs/fb_terms | 5/30 | From exp01 |
| BM25 top-K | 1000 | Standard depth |
| Dense top-K | 1000 | Standard depth |
| Fusion alpha | 0.3 | Best from exp05 (dense=0.3, sparse=0.7) |
| Reranker max_length | 768 | Best from exp03b |
| Reranker batch_size | 64 | From exp03b |
| Reranker depth | 1000 | Best from exp03b |
| Reranker instruction | news-short | Best from exp04 |

## Runs

### Run 1: `dense-qwen3-8b-zeroshot`
- **Description**: Zero-shot Qwen3-Embedding-8B dense retrieval, top-1000
- **Parameter overrides**: None
- **Expected output**: `runs/dense-qwen3-8b-zeroshot.run`
- **Purpose**: Establish 8B dense retrieval baseline, compare to 0.6B (MAP@100=0.2105)

### Run 2: `linear-a03`
- **Description**: Linear fusion of BM25+Bo1 and 8B dense retrieval (alpha=0.3 dense, 0.7 sparse)
- **Parameter overrides**: None
- **Expected output**: `runs/linear-a03.run`
- **Purpose**: Test hybrid fusion with stronger dense component

### Run 3: `linear-a03-reranked`
- **Description**: Best fusion (linear-a03) reranked with Qwen3-Reranker-0.6B (news-short, ml768, top-1000)
- **Parameter overrides**: None
- **Expected output**: `runs/linear-a03-reranked.run`
- **Purpose**: Full pipeline -- should achieve new best

## Expected Outcome

- Dense-only (8B): MAP@100 ~0.25-0.28 (up from 0.2105 with 0.6B)
- Linear fusion (a=0.3): MAP@100 ~0.28-0.30 (up from 0.2762 with 0.6B)
- Fusion + reranker: MAP@100 ~0.29-0.32 (up from 0.2827 with 0.6B)

## Baseline Comparison

- Qwen3-Embedding-0.6B dense-only: MAP@100 = 0.2105
- BM25+Bo1: MAP@100 = 0.2504
- Linear(a=0.3) + Qwen3-Reranker (exp05 best): MAP@100 = 0.2827

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training, zero-shot only)
- [x] Training does NOT use Robust04 qrels (no training)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A, no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (all models used zero-shot)
