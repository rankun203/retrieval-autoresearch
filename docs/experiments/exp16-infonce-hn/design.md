# exp16-infonce-hn: Hard Negative Training with InfoNCE Loss

## Literature Review

### Papers Read

1. **NV-Retriever: Improving text embedding models with effective hard negative mining**
   - Authors: Zhuolin Yang et al. (NVIDIA), 2024
   - URL: https://arxiv.org/abs/2407.15831
   - Key takeaways: Positive-aware hard negative selection is critical. ~70% of BM25-mined negatives are false negatives. LLM relabeling improves quality significantly.

2. **GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval**
   - Authors: Kexin Wang et al., NAACL 2022
   - URL: https://arxiv.org/abs/2112.07577
   - Key takeaways: MarginMSE works for domain adaptation but is sensitive to label quality. Contrastive losses (InfoNCE) are more robust to noisy labels than regression losses.

3. **Gecko: Versatile Text Embeddings Distilled from LLMs**
   - Authors: Jinhyuk Lee et al. (Google DeepMind), 2024
   - URL: https://arxiv.org/abs/2403.20327
   - Key takeaways: LLM relabeling of positive/negative passages dramatically improves distillation quality. Binary labels from LLM P(yes) scoring sufficient.

### Key Learnings from Prior Experiments

- **exp33 (data-leaked, proved approach)**: e5-base-v2 with InfoNCE + 3-phase iterative HN mining achieved dense MAP@100=0.3152, hybrid 0.3483. Demonstrated that InfoNCE + iterative mining is highly effective.
- **exp13/exp13b (MarginMSE, catastrophic forgetting)**: MarginMSE with binary no-think P(yes) scores caused catastrophic forgetting at ALL scales (500/1000/2000/4000 queries). The regression loss is fundamentally unstable for this task.
- **Key differences**: InfoNCE (contrastive, learns relative ordering) vs MarginMSE (regression, learns absolute margins). InfoNCE is more forgiving of noisy labels because it only needs correct relative ordering.

## Goal

Fine-tune Qwen3-Embedding-0.6B with InfoNCE loss using LLM-scored hard negatives from Robust04, following the proven exp33 approach but with MS-MARCO queries (not test queries) for training.

## Hypothesis

exp13b's catastrophic forgetting was caused by MarginMSE loss, not insufficient data. By switching to InfoNCE with:
1. In-batch negatives (symmetric) for efficient contrastive learning
2. Mixed MS-MARCO + Robust04 HN training (70/30)
3. Lower LR (1e-5 vs 2e-5)
4. 3-phase iterative mining

...we should achieve MAP@100 > 0.28 in fusion with the 0.6B model, potentially approaching 0.30.

## Method

### Run A: No-think labels, full scale (4997 queries)

Uses cached 249K no-think P(yes) scores as binary labels:
- P(yes) > 0.5 = positive candidate
- P(yes) < 0.5 = negative candidate
- Highest P(yes) per query = positive document
- Lowest P(yes) per query (within BM25 top-50) = hard negative

Training pipeline:
1. Load cached scores (249K scored pairs for 4997 queries)
2. Build positive/negative assignments from P(yes) threshold
3. Phase 1 (200s): Train on MS-MARCO with InfoNCE
4. Mine round 1: Re-encode Robust04 with MS-MARCO queries, collect HN
5. Phase 2 (200s): Mixed training (70% MS-MARCO + 30% cached Robust04 HN)
6. Mine round 2: Re-encode with improved model
7. Phase 3 (200s): Mixed training (70% MS-MARCO + 30% round-2 HN)
8. Evaluate: dense-only + BM25+Bo1 linear fusion alpha=0.3

### Run B: Think-mode labels, 1000 queries

Same training pipeline as Run A, but with higher-quality think-mode LLM labels:
- Sample 1000 queries from cached 5000 LLM-selected news queries
- Score 50K pairs with Qwen3-8B in THINK mode (separate llm_think_score.py)
- Think-mode provides better-calibrated scores (mean P(yes) ~0.53 vs no-think ~0.81)
- Expected ~21 hours for think scoring at ~0.66 pairs/sec

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen/Qwen3-Embedding-0.6B | Encoder to improve |
| Pooling | last_token (EOS) | Qwen3 architecture |
| Loss | InfoNCE (symmetric) | Proven in exp33, avoids MarginMSE forgetting |
| Temperature | 0.05 | From exp33 |
| Learning rate | 1e-5 | Lower than exp13b (2e-5) to prevent forgetting |
| Batch size | 32 | Fit in L40S VRAM with fp32 + gradient checkpointing |
| Doc max length | 512 | Consistent with cached embeddings |
| Query max length | 512 | Consistent with Qwen3 encoding |
| Training dtype | fp32 (model), fp16 forward | Prevent NaN, reduce VRAM |
| Gradient checkpointing | Yes | Required for 0.6B model fp32 training |
| Phases | 3 x 200s | Iterative mining |
| HN per query | 10 | Top non-relevant from mining |
| Robust04 batch ratio | 0.3 | 70% MS-MARCO + 30% Robust04 HN |
| Fusion alpha | 0.3 | Best from exp05/exp07 (dense weight) |
| Encode batch size | 256 | For inference/encoding |
| Instruction | "Given a topic query, retrieve relevant news articles that discuss the topic" | News domain |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True | Prevent OOM fragmentation |

## Runs

### Run A: `nothink-dense` and `nothink-fusion`
- Description: Full scale (4997 queries), no-think P(yes) labels, 3-phase InfoNCE training
- Parameter overrides: None (default config)
- Expected output: `runs/nothink-dense.run`, `runs/nothink-fusion.run`

### Run B: `think-dense` and `think-fusion`
- Description: 1000 queries, think-mode P(yes) labels, 3-phase InfoNCE training
- Parameter overrides: Uses think-mode scored data from llm_think_score.py
- Expected output: `runs/think-dense.run`, `runs/think-fusion.run`
- Prerequisite: llm_think_score.py must be run first (~21 hours)

## Expected Outcome

| Run | Dense MAP@100 | Fusion MAP@100 | Rationale |
|-----|--------------|----------------|-----------|
| A (no-think) | ~0.22-0.24 | ~0.28-0.30 | More data + InfoNCE should prevent forgetting |
| B (think) | ~0.23-0.25 | ~0.29-0.31 | Better labels should help further |

Target: MAP@100 > 0.29 fusion (beating current best 0.2929 with 0.6B model).

## Baseline Comparison

| Method | MAP@100 | Source |
|--------|---------|--------|
| BM25+Bo1 | 0.2504 | exp01 |
| 0.6B zero-shot fusion | 0.2762 | exp05 |
| 0.6B + doc expansion fusion | 0.2903 | exp12 |
| 8B fusion (current best) | 0.2929 | exp07 |
| exp13b MarginMSE fusion (best) | 0.2303 | exp13b (discard, catastrophic forgetting) |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries
- [x] Training does NOT use Robust04 qrels (relevance judgments)
- [x] Hard negative mining uses MS-MARCO queries only (LLM-selected news queries)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights
- [x] Robust04 corpus used only for document encoding/indexing (allowed)
- [x] Retrieval from Robust04 corpus uses MS-MARCO queries only (allowed)
- [x] LLM-scored labels derived from MS-MARCO queries + Robust04 corpus (allowed)
