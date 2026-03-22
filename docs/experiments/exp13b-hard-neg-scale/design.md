# exp13b-hard-neg-scale: Hard Negative Mining v2 with LLM Query Selection and Scaling Study

## Literature Review

### Papers Read (from exp13, still applicable)

1. **NV-Retriever: Improving text embedding models with effective hard negative mining**
   - Authors: Zhuolin Yang et al. (NVIDIA), 2024
   - URL: https://arxiv.org/abs/2407.15831
   - Key takeaways: Positive-aware hard negative selection critical. ~70% BM25-mined negatives are false negatives. Achieved #1 MTEB Retrieval.

2. **GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval**
   - Authors: Kexin Wang et al., NAACL 2022
   - URL: https://arxiv.org/abs/2112.07577
   - Key takeaways: MarginMSE loss works well for domain adaptation without labeled data.

3. **Gecko: Versatile Text Embeddings Distilled from LLMs**
   - Authors: Jinhyuk Lee et al. (Google DeepMind), 2024
   - URL: https://arxiv.org/abs/2403.20327
   - Key takeaways: LLM relabeling of positive/negative passages improves distillation quality.

### Key Learnings from exp13

- **Data insufficiency was the root cause of failure**: Only 139/500 queries produced valid training examples due to keyword filtering + CE > 0.8 threshold. 862 examples across 4 epochs led to memorization (loss ~0.012).
- **LLM scoring infrastructure works**: Think-mode P(yes) gave 48% better pos/neg separation than no-think (0.234 vs 0.158). Mean P(yes) for think was 0.527 (well-calibrated), vs 0.812 for no-think (positivity bias).
- **Insight**: Replace keyword filtering with LLM-based query classification to find 10-30x more news-relevant queries. Replace CE scoring with LLM P(yes) scoring to avoid the 0.8 threshold bottleneck.

## Goal

Fine-tune Qwen3-Embedding-0.6B with 10-30x more training data from LLM-selected news queries, studying the effect of data scale on retrieval quality.

## Hypothesis

exp13 failed because only 139 queries produced valid training examples. By using Qwen3-8B to classify ALL MS-MARCO queries for news-topic relevance (instead of keyword matching), we can find 2000-5000 relevant queries. Each query's BM25 candidates are then scored by the same LLM for relevance, providing continuous MarginMSE training labels. With 10-30x more data, the 0.6B model should generalize rather than memorize, achieving MAP@100 > 0.28 in fusion.

## Method

### 3-Script Pipeline

```
select_news_queries.py    (Phase 0: LLM query classification)
        |
        v
    top-5000 news queries saved to cache
        |
        v
train.py Phase 1          (BM25 mining from Robust04 corpus)
        |
        v
    (query, doc_id, bm25_rank) pairs saved to cache
        |
        v
llm_score.py              (Phase 2: LLM relevance scoring)
        |
        v
    (query, doc_id, P(yes)) scores saved to cache
        |
        v
train.py Phase 3           (Training at multiple scales)
        |
        v
train.py Phase 4           (Evaluation: dense-only + fusion)
```

### Phase 0: LLM Query Selection (select_news_queries.py)
- Stream ALL unique MS-MARCO queries from stream_msmarco_triples()
- Use Qwen3-8B in NO-THINK mode (binary classification, speed over quality)
- Prompt: "Would this query likely appear as a topic discussed in newspaper articles from the 1990s? Answer yes or no."
- Extract P(yes) via logprobs, save top 5000 queries by P(news)
- vLLM with max_model_len=256, max_num_seqs=256 for throughput
- Expected: ~22+ prompts/s with tiny inputs, ~1-2 hours for full scan

### Phase 1: BM25 Mining (train.py)
- Load LLM-selected queries (top 5000 by P(news))
- BM25 retrieve top-50 from Robust04 corpus per query
- Save all (query, doc_id, bm25_rank) candidates -- no CE filtering

### Phase 2: LLM Relevance Scoring (llm_score.py)
- Score ALL (query, doc) pairs with Qwen3-8B in THINK mode
- Extract P(yes) as continuous relevance score
- max_tokens=2048 for think, retry with 4096 if think doesn't finish
- Save scores to cache

### Phase 3: Scaling Study (train.py)
Train at 4 scales, taking top-N queries by P(news):
- 500 queries (1x, matching exp13 scale)
- 1000 queries (2x)
- 2000 queries (4x)
- 4000 queries (8x)

For each query:
- Positive = doc with highest P(yes) per query (must have P(yes) > 0.5)
- Hard negatives = docs with 0.1 < P(yes) < 0.4
- Easy negatives = docs with P(yes) < 0.1
- MarginMSE loss with LLM P(yes) margins as teacher signal

Training config per scale:
- fp32 model, gradient_checkpointing
- Separate q/pos/neg encoding (OOM prevention)
- Learning rate: 2e-5 with warmup
- Batch size: 16
- 600s training budget per scale

### Phase 4: Evaluation (train.py)
For each scale:
- Fine-tuned dense-only retrieval
- Fine-tuned + BM25+Bo1 fusion alpha=0.3

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen/Qwen3-Embedding-0.6B | Encoder to improve |
| LLM judge | Qwen/Qwen3-8B | Quality scoring, fits on L40S |
| Query selection | No-think mode, P(news) > 0.5 | Speed for classification |
| Relevance scoring | Think mode, P(yes) logprobs | Quality for training labels |
| Num queries scanned | ~500K (full MS-MARCO stream) | Exhaustive scan |
| Top queries selected | 5000 | Large pool for scaling study |
| BM25 mining depth | 50 per query | Enough candidates |
| Training batch size | 16 | Memory constraint |
| Learning rate | 2e-5 | Standard fine-tuning |
| Training dtype | fp32 | fp16 causes NaN |
| Doc max length | 512 | Consistent with baselines |
| Query max length | 512 | Consistent with baselines |
| Fusion alpha | 0.3 | Best from exp05/exp07 |
| Time budget | 600s per scale | Fair comparison |
| Negatives per query | up to 7 (5 hard + 2 easy) | Balance difficulty |
| Instruction | "Given a topic query, retrieve relevant news articles that discuss the topic" | News domain |

## Runs

### Run 1: `scale-500-dense`
- Description: 500 queries, fine-tuned dense-only
- Expected output: `runs/scale-500-dense.run`

### Run 2: `scale-500-fusion`
- Description: 500 queries, fine-tuned + BM25+Bo1 fusion alpha=0.3
- Expected output: `runs/scale-500-fusion.run`

### Run 3: `scale-1000-dense`
- Description: 1000 queries, fine-tuned dense-only
- Expected output: `runs/scale-1000-dense.run`

### Run 4: `scale-1000-fusion`
- Description: 1000 queries, fine-tuned + BM25+Bo1 fusion alpha=0.3
- Expected output: `runs/scale-1000-fusion.run`

### Run 5: `scale-2000-dense`
- Description: 2000 queries, fine-tuned dense-only
- Expected output: `runs/scale-2000-dense.run`

### Run 6: `scale-2000-fusion`
- Description: 2000 queries, fine-tuned + BM25+Bo1 fusion alpha=0.3
- Expected output: `runs/scale-2000-fusion.run`

### Run 7: `scale-4000-dense`
- Description: 4000 queries, fine-tuned dense-only
- Expected output: `runs/scale-4000-dense.run`

### Run 8: `scale-4000-fusion`
- Description: 4000 queries, fine-tuned + BM25+Bo1 fusion alpha=0.3
- Expected output: `runs/scale-4000-fusion.run`

## Expected Outcome

| Scale | Dense MAP@100 | Fusion MAP@100 | Rationale |
|-------|---------------|----------------|-----------|
| 500 (1x) | ~0.19-0.21 | ~0.27-0.28 | Similar to exp13 but better labels |
| 1000 (2x) | ~0.20-0.22 | ~0.27-0.29 | More data helps |
| 2000 (4x) | ~0.21-0.23 | ~0.28-0.30 | Diminishing returns begin |
| 4000 (8x) | ~0.21-0.24 | ~0.28-0.30 | Diminishing returns |

Target: MAP@100 > 0.28 fusion at 4x+ scale (beating zero-shot fusion 0.2762).

## Baseline Comparison

| Method | MAP@100 | Source |
|--------|---------|--------|
| BM25+Bo1 | 0.2504 | exp01 |
| 0.6B zero-shot fusion | 0.2762 | exp05 |
| 0.6B + doc expansion fusion | 0.2903 | exp12 |
| 8B fusion (current best) | 0.2929 | exp07 |
| exp13 finetuned fusion (139 queries) | 0.2724 | exp13 (discard) |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries
- [x] Training does NOT use Robust04 qrels (relevance judgments)
- [x] Hard negative mining uses MS-MARCO queries only (LLM-selected subset)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights
- [x] Robust04 corpus used only for document encoding/indexing (allowed)
- [x] Retrieval from Robust04 corpus uses MS-MARCO queries only (allowed)
