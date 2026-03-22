# Review: exp13b-hard-neg-scale

## Data Leakage Check: PASS

Thorough line-by-line review of all three scripts:

**train.py**:
- `load_robust04()` called at line 528. Returns `corpus, queries, qrels`.
- `corpus` used for: doc text lookup in BM25 mining (line 177), doc embedding (line 591), eval encoding (line 641). ALLOWED.
- `queries` used ONLY at line 461 inside `evaluate_model()` for final evaluation encoding. ALLOWED.
- `qrels` used ONLY at lines 478 and 486 inside `evaluate_model()` via `evaluate_run()`. ALLOWED.
- Training data comes entirely from MS-MARCO queries (via `select_news_queries.py` -> `stream_msmarco_triples()`), never from Robust04 queries/qrels.
- BM25 mining (Phase 1, line 162) uses LLM-selected MS-MARCO queries, not Robust04 queries. ALLOWED.

**select_news_queries.py**:
- Streams from `stream_msmarco_triples()` only (line 116). No Robust04 queries/qrels used. PASS.

**llm_score.py**:
- Loads BM25 mining data (which was mined with MS-MARCO queries). No Robust04 queries/qrels used. PASS.

**Verdict**: No test-time information flows into model weights or training data selection.

## Code Quality

Generally good 3-script pipeline architecture. Issues noted:

1. **Design deviation**: Design specifies Phase 2 uses "THINK mode" (line 79 of design.md), but `llm_score.py` actually uses NO-THINK mode (prompt includes `<think>\n</think>\n\n` to suppress reasoning, MAX_TOKENS=32). The code comment on line 27 acknowledges "Think mode infeasible for 249K pairs". This means teacher scores are binary-biased no-think P(yes) rather than well-calibrated think-mode scores. The design's exp13 analysis showed think mode had 48% better pos/neg separation (0.234 vs 0.158 margin), so this deviation likely degraded label quality significantly.

2. **No learning rate warmup implemented**: Config specifies `WARMUP_STEPS = 100` (line 41) but no warmup scheduler is created -- only a plain `AdamW` optimizer with constant LR (line 338). The 2e-5 LR may be too aggressive without warmup for a pretrained model.

3. **No gradient accumulation**: With TRAIN_BATCH=4 (line 39), effective batch size is very small. Combined with high LR and no warmup, this leads to noisy gradients early in training.

4. **Loss curve anomaly**: The loss curve printed in the summary shows a strange pattern -- the 0%-50% range shows standard decreasing loss, but 90%-100% jumps back up (0.4372 -> 0.3672). This is because the loss curve is computed across ALL scales concatenated, mixing scale-500 (which overfit to ~0.09) with scale-4000 (which barely dropped from 0.58 to 0.38).

## Cache Verification

- **BM25 mining cache**: Loaded from `.cache/bm25_mining_exp13b_BM25_depth-50_top_k-5000` -- correct model (BM25), correct parameters (depth=50, top_k=5000). PASS.
- **LLM scores cache**: Loaded from `.cache/llm_relevance_exp13b_Qwen_Qwen3-8B_num_pairs-249151_num_queries-4997` -- correct model (Qwen3-8B), correct pair count. PASS.
- **Embeddings cache**: Loaded from `.cache/embeddings_Qwen_Qwen3-Embedding-0.6B_dataset-robust04_max_length-512_pooling-last_token` -- correct model and parameters. PASS.
- **Note**: Embeddings cache is the BASE model's corpus embeddings, used only for the initial doc_ids list. The fine-tuned model re-encodes the entire corpus for each scale (confirmed in log). PASS.

## Design Adherence

| Aspect | Design | Actual | Match? |
|--------|--------|--------|--------|
| Scales | 500, 1000, 2000, 4000 | 500, 1000, 2000, 4000 | YES |
| Training budget | 600s per scale | 600-601s per scale | YES |
| Learning rate | 2e-5 | 2e-5 | YES |
| Batch size | 16 | 4 (TRAIN_BATCH=4) | NO |
| Phase 2 scoring | Think mode | No-think mode | NO |
| Fusion alpha | 0.3 | 0.3 | YES |
| Query selection | No-think LLM classification | No-think LLM classification | YES |
| Runs produced | 8 (4 dense + 4 fusion) | 8 | YES |

Two significant deviations:
1. Batch size reduced from 16 to 4 (OOM prevention for fp32) -- reasonable engineering choice.
2. Phase 2 scoring switched from think to no-think mode -- major quality compromise that likely contributed to poor results.

## Performance Analysis

### Results Summary

| Scale | Dense MAP@100 | Fusion MAP@100 | Dense nDCG@10 | Fusion nDCG@10 | Steps | Valid Examples |
|-------|--------------|----------------|---------------|----------------|-------|---------------|
| 500 | 0.0157 | 0.2236 | 0.0864 | 0.4528 | 608 | 485 |
| 1000 | 0.0410 | 0.2303 | 0.1573 | 0.4635 | 615 | 961 |
| 2000 | 0.0504 | 0.2302 | 0.2009 | 0.4632 | 615 | 1918 |
| 4000 | 0.0323 | 0.2236 | 0.1422 | 0.4568 | 621 | 3825 |

### Key Observations

1. **Catastrophic forgetting**: Dense-only MAP@100 ranges 0.0157-0.0504, vs 0.2105 zero-shot. Training destroyed the model's general retrieval capability rather than improving it. The model went from useful dense retrieval to near-random.

2. **Fusion masks the damage**: Fusion results (0.2236-0.2303) are well below even BM25+Bo1 alone (0.2504), meaning the fine-tuned dense component is actually _hurting_ the BM25 signal in fusion. The dense model contributes negative value.

3. **Scale 500 overfits severely**: Loss drops from 0.57 to 0.086 in 600s (2 epochs), classic memorization. Scale 1000 sees 1 epoch, still drops substantially (0.55 -> 0.15). Scale 4000 barely moves (0.58 -> 0.38 in <1 epoch) -- undertrained but still catastrophically forgets.

4. **More data does NOT help**: Scale 4000 (3825 examples, 13K tuples) performs worse than scale 1000 (961 examples, 3.4K tuples) on both dense and fusion. This rules out data quantity as the bottleneck.

5. **Non-monotonic scaling**: Dense MAP peaks at scale-2000 then drops at scale-4000. This suggests the lower-P(news) queries added at larger scales introduce noise rather than useful signal.

### Root Cause Analysis

The fundamental issue is the combination of:
- **Binary no-think P(yes) teacher scores**: Mean P(yes)=0.285 with bimodal distribution (most scores near 0 or 1). MarginMSE with binary margins (near 0 or 1) effectively becomes a hard-label loss, losing the soft supervision advantage.
- **MarginMSE without regularization**: No KL divergence to the original model, no weight decay beyond optimizer default. The model is free to drift arbitrarily far from the pretrained weights.
- **High learning rate (2e-5) without warmup**: Despite `WARMUP_STEPS=100` being defined, no scheduler was created. The constant 2e-5 LR from step 0 is aggressive for fine-tuning a pretrained embedding model.

### Comparison to Baselines

| Method | MAP@100 | Delta vs this |
|--------|---------|---------------|
| BM25+Bo1 | 0.2504 | +0.0201 |
| 0.6B zero-shot fusion | 0.2762 | +0.0459 |
| exp13 finetuned fusion (139q) | 0.2724 | +0.0421 |
| exp12 doc expansion fusion | 0.2903 | +0.0600 |
| 8B fusion (current best) | 0.2929 | +0.0626 |
| **This exp best (scale-1000 fusion)** | **0.2303** | -- |

This experiment's best result is worse than plain BM25+Bo1 and worse than exp13 (which only had 139 training queries). The hypothesis that more data would help was disproven -- the training methodology itself is the bottleneck.

## Budget Assessment

- Scale 500: OVERFIT (loss 0.57 -> 0.086 in 2 epochs)
- Scale 1000: UNDERTRAINED (loss 0.55 -> 0.15 in 1 epoch, still declining)
- Scale 2000: UNDERTRAINED (loss 0.45 -> 0.27 in <1 epoch)
- Scale 4000: UNDERTRAINED (loss 0.58 -> 0.38 in <1 epoch)

However, budget assessment is irrelevant since more training would only cause more forgetting. The training objective itself is the problem.

## Infrastructure Wins

Despite poor retrieval results, the experiment validated important infrastructure:
1. **LLM query selection works**: Found 4787/4997 queries with positives (vs 139/500 in exp13). 10x improvement in data pipeline.
2. **LLM scoring at scale**: 249K pairs scored at 18.5 pairs/s with 100% valid P(yes) extraction. The scoring infrastructure is production-ready.
3. **Scaling study framework**: Clean 4-scale training+eval pipeline with per-scale model reload. Reusable for future experiments.

## Verdict: **APPROVE**

The experiment is methodologically sound (no leakage, correct pipeline), produced all 8 planned runs, and generated valuable negative results. The infrastructure improvements (LLM query selection, large-scale scoring) are reusable. All 8 runs are logged as `discard` since none beat the current best MAP@100 of 0.2929.

## Recommendations

For future fine-tuning experiments:
1. **Use contrastive loss (InfoNCE/NTXent)** instead of MarginMSE. Contrastive loss with in-batch negatives naturally preserves representation structure and is less prone to catastrophic forgetting.
2. **Use think-mode P(yes)** for teacher scores. The design correctly identified this but it was switched to no-think for speed. Consider scoring a smaller subset (e.g., 20K pairs) with think mode rather than 249K pairs with no-think.
3. **Add regularization**: KL divergence to original model, or freeze lower layers, or use very low LR (1e-6) with proper warmup scheduler.
4. **Implement the warmup scheduler**: The code defines WARMUP_STEPS=100 but never creates a scheduler.
5. **Consider ANCE-style iterative mining**: Use the fine-tuned model to mine harder negatives each round, rather than static BM25 negatives.
