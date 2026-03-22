# exp13-hard-negatives: Hard Negative Mining with LLM Relevance Judging

## Literature Review

### Papers Read

1. **NV-Retriever: Improving text embedding models with effective hard negative mining**
   - Authors: Zhuolin Yang et al. (NVIDIA), 2024
   - URL: https://arxiv.org/abs/2407.15831
   - Key takeaways: Positive-aware hard negative selection is critical. ~70% of BM25-mined hard negatives from MS-MARCO are actually false negatives. Filter negatives using CE score thresholds relative to the positive score (CE_neg < 0.95 * CE_pos). Achieved #1 on MTEB Retrieval.

2. **GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval**
   - Authors: Kexin Wang et al., NAACL 2022
   - URL: https://arxiv.org/abs/2112.07577
   - Key takeaways: MarginMSE loss = MSE(bi_encoder_margin, cross_encoder_margin) where margin = score(q, pos) - score(q, neg). Cross-encoder teacher provides soft labels. Works well for domain adaptation without labeled data.

3. **RLHN: Retrieval-augmented Learning with Hard Negatives via LLM Cascading**
   - Authors: Castorini group, EMNLP 2025
   - Key takeaways: Cascading LLM relabeling of hard negatives improves OOD retrieval by +3.2 nDCG. LLM judging cleans false negatives from CE-mined candidates.

4. **R-GPL: Dynamic Remining During Training**
   - Authors: Jan 2025
   - Key takeaways: Re-encoding corpus with updated model and remining harder negatives iteratively improves training quality. 2-3 rounds of remining sufficient.

5. **Gecko: Versatile Text Embeddings Distilled from LLMs**
   - Authors: Jinhyuk Lee et al. (Google DeepMind), 2024
   - URL: https://arxiv.org/abs/2403.20327
   - Key takeaways: LLM relabeling of positive/negative passages improves distillation data quality. Two-step: generate synthetic data, then relabel with LLM.

6. **TREMA: Multi-Criteria Decomposition for LLM Relevance Judging**
   - Authors: Farzi & Dietz, ICTIR 2025
   - URL: https://arxiv.org/html/2507.09488
   - Key takeaways: LLaMA-3-8B with multi-criteria (Exactness, Topicality, Coverage, Context Fit) beats GPT-4o with naive prompting. Won LLMJudge challenge.

7. **UMBRELA: UMbrella Benchmark for Relevance Assessment with LLMs**
   - URL: https://arxiv.org/html/2406.06519v1
   - Key takeaways: Step-by-step CoT reasoning with graded 0-3 scale improves reliability. Anti-leniency instructions reduce false positives.

### Key Implementation Details from Literature

- **MarginMSE loss**: `loss = MSE(cos_sim(q, pos) - cos_sim(q, neg), CE_score(q, pos) - CE_score(q, neg))`
- **Positive-aware filtering**: Hard negatives must score below 0.95x the positive CE score but above 0.3 absolute
- **LLM judging**: Pointwise, one doc per prompt, graded 0-3, binarize at threshold 2
- **Anti-leniency**: "A passage that merely mentions query terms without substantively addressing the information need should receive 0 or 1"
- **Expected label distribution**: ~40-50% grade 0, 25-30% grade 1, 15-20% grade 2, 5-10% grade 3

## Goal

Fine-tune Qwen3-Embedding-0.6B with high-quality hard negatives to beat the current best MAP@100 of 0.2929 (Qwen3-8B fusion) using only a 0.6B model. Target: MAP@100 > 0.30.

## Hypothesis

The zero-shot Qwen3-Embedding-0.6B achieves MAP@100=0.2762 in fusion with BM25+Bo1 (exp05). The gap to the 8B model (0.2929) is due to embedding quality, not architecture. By fine-tuning with high-quality hard negatives from cross-encoder scoring + LLM false-negative cleanup, we can close or exceed this gap. The 0.6B model has sufficient capacity -- it just needs better training signal from the Robust04 news domain.

## Method

### 3-Phase Pipeline Diagram

```
                    MS-MARCO Queries (500 news-filtered)
                                |
                                v
                  +---------------------------+
                  | Retrieve top-500 per query |
                  | from Robust04 corpus using |
                  | BM25 (lexically-hard negs  |
                  | for the dense encoder)     |
                  +---------------------------+
                                |
                                v
            +-----------------------------------------+
            |        PHASE 1: CE Teacher Scoring       |
            |                                          |
            |  Score all candidates with                |
            |  BAAI/bge-reranker-v2-m3 (568M)          |
            |  AutoModelForSequenceClassification       |
            |                                          |
            |  Positive-aware filtering (NV-Retriever): |
            |  - Pos = highest CE-scored doc (CE > 0.8) |
            |  - Hard neg: CE < 0.95*pos AND CE > 0.3   |
            |  - Easy neg: CE < 0.3 (random sample)     |
            |                                          |
            |  Save mining data to .cache/ for reuse    |
            +-----------------------------------------+
                                |
                                v
            +-----------------------------------------+
            |  PHASE 2: LLM Relevance Scoring          |
            |  (llm_cleanup.py — SEPARATE PROCESS)     |
            |                                          |
            |  Score ALL candidates with Qwen3-8B:     |
            |  - Thinking mode: <think>CoT</think>     |
            |  - After think, first token = yes/no     |
            |  - Extract P(yes) from logprobs as       |
            |    continuous relevance score [0,1]       |
            |                                          |
            |  This gives:                             |
            |  - CoT reasoning quality (thinking)      |
            |  - Continuous scores (token probs)       |
            |  - Can be used directly for MarginMSE    |
            |                                          |
            |  Pointwise, 1 doc per prompt, 768 tokens |
            |  Anti-leniency instruction in prompt     |
            |  vLLM logprobs for probability extraction |
            |                                          |
            |  Also experiment with:                   |
            |  - Bigger reranker as alternative teacher |
            |    (e.g., bge-reranker-v2-m3, 568M)     |
            |  - Compare LLM vs CE vs both as teacher  |
            +-----------------------------------------+
                                |
                                v
            +-----------------------------------------+
            |     PHASE 3: Training + Iterative Remine |
            |                                          |
            |  Teacher options (experiment with both): |
            |  a) LLM P(yes) scores from Phase 2      |
            |  b) CE reranker scores from Phase 1     |
            |  c) Combined/ensemble                    |
            |                                          |
            |  Fine-tune Qwen3-Embedding-0.6B:         |
            |  - Load model in fp32 (fp16 = NaN loss)  |
            |  - gradient_checkpointing_enable()       |
            |  - Encode q/pos/neg SEPARATELY (OOM fix) |
            |  - MarginMSE loss (student vs teacher)   |
            |                                          |
            |  2 rounds of iterative remining:         |
            |  - After round 1: re-encode corpus with  |
            |    updated model, mine new candidates,   |
            |    re-score with LLM+CE, retrain         |
            +-----------------------------------------+
                                |
                                v
            +-----------------------------------------+
            |           EVALUATION                     |
            |                                          |
            |  Run 1: Fine-tuned 0.6B dense-only      |
            |  Run 2: + BM25+Bo1 fusion alpha=0.3     |
            |  Run 3: + doc expansion + fusion         |
            +-----------------------------------------+
```

### Key Design Decisions

- **Base model**: Qwen3-Embedding-0.6B -- the encoder we are improving
- **Training queries**: ~500 MS-MARCO queries filtered for news-topic relevance. NEVER Robust04 test queries.
- **Training corpus**: Robust04 528K docs (corpus text is allowed, queries/qrels are not)
- **Candidate mining**: BM25 for round 0 (lexically-hard negatives for the dense encoder), bi-encoder for later rounds
- **News instruction**: "Given a topic query, retrieve relevant news articles that discuss the topic"
- **Training**: Full fine-tuning (not LoRA) of 0.6B model in fp32 -- small enough for full FT
- **VRAM management**: Free embedding model before loading CE model, free CE model before training
- **Retrieval**: Numpy dot product (NOT faiss-gpu which initializes CUDA)

### LLM as Teacher: Thinking Mode + Token Probabilities

The key innovation: use Qwen3-8B's **thinking mode** to get both CoT reasoning AND continuous scores.

**How it works (Option C — generate with logprobs):**
1. Prompt asks: "Is this passage relevant to the query? Answer yes or no."
2. Model generates full response with `logprobs=5` enabled in vLLM
3. Output: `<think>...reasoning about relevance...</think>\n\nyes` (or no)
4. Parse the output: find the token position after `</think>\n\n`
5. Extract logprobs at that position → compute `P(yes) = softmax(logit_yes, logit_no)[1]`
6. Use `P(yes)` as continuous relevance score in [0, 1]

**Logprobs size estimate:**
- ~300 tokens per response × top-5 logprobs × 50 bytes = ~15KB per judgment
- 250K judgments × 15KB = ~3.75GB during inference (not persisted)
- We only save the final P(yes) float per judgment → negligible storage

**Detection of think completion:**
- vLLM returns logprobs for every generated token in the sequence
- Tokenize the output text, find `</think>` token position
- The answer token (yes/no) follows the newlines after `</think>`
- Read its logprob from the returned logprobs array

**No-think mode (for comparison):**
- Official Qwen3 way: prepend `<think>\n</think>\n\n` to force skip thinking
- The model immediately generates yes/no without reasoning
- Same P(yes) extraction from logprobs

### Thinking vs No-Thinking Comparison

Two experimental conditions to compare:

| Condition | Prompt suffix | Expected behavior |
|-----------|--------------|-------------------|
| **think** | (none — thinking is default) | `<think>...CoT reasoning...</think>\n\nyes/no` |
| **no-think** | Prepend `<think>\n</think>\n\n` to assistant response | Direct `yes/no` without reasoning |

**Hypothesis:** Thinking mode produces better-calibrated P(yes) scores because the model reasons through relevance before committing. No-think gives a quick gut reaction. We expect:
- Think P(yes): better separation between relevant/irrelevant docs
- No-think P(yes): faster but noisier, may have stronger positivity bias
- Training with think-scored data should produce a better fine-tuned model

Both conditions run on the same 250K (query, doc) pairs for direct comparison.

### Multi-Teacher Experiment

Compare different teacher signals:
1. **LLM think P(yes)** — Qwen3-8B with thinking + token probabilities
2. **LLM no-think P(yes)** — Qwen3-8B without thinking (faster, noisier)
3. **CE reranker** — BGE-reranker-v2-m3 or Qwen3-Reranker-0.6B scores
4. **Combined** — average or weighted combination of LLM + CE scores
4. **Bigger reranker** — try a larger reranker model if VRAM allows

This lets us empirically determine which teacher produces the best training signal.

### News Query Selection Strategy

From MS-MARCO, select queries that match news-domain topics:
- Keywords indicating news relevance: government, policy, war, economy, election, climate, disaster, health, law, international, military, etc.
- Also include broad topic queries that overlap with TREC topics (politics, science, environment, etc.)
- Select ~500 queries total for manageable computation

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen/Qwen3-Embedding-0.6B | Current 0.6B encoder to improve |
| CE teacher | BAAI/bge-reranker-v2-m3 | 568M cross-encoder, different arch from student |
| LLM judge | Qwen/Qwen3-8B | Good quality, fits on L40S (via vLLM, separate process) |
| Training queries | ~500 news-filtered MS-MARCO | Domain-relevant subset |
| Mining method | BM25 (round 0), bi-encoder (later) | BM25 gives lexically-hard negatives |
| Retrieval depth | 500 per query | Balance coverage vs compute |
| CE batch size | 16 | BGE-reranker is 568M, needs smaller batches |
| CE max length | 512 | Sufficient for scoring |
| CE positive threshold | 0.8 | High confidence positives |
| CE hard neg range | 0.3 to 0.95*pos | NV-Retriever-style filtering |
| CE easy neg threshold | < 0.3 | For training stability mix |
| LLM borderline range | 0.7 to 0.95 * pos_score | Focus LLM budget on ambiguous cases |
| LLM doc length | 768 tokens | Sufficient for news articles |
| LLM grade threshold | >= 2 = relevant (remove) | Conservative: keep only clear non-relevant |
| Training batch size | 16 | Encode q/pos/neg separately to avoid OOM |
| Learning rate | 2e-5 | Standard for fine-tuning embedders |
| Training dtype | fp32 | fp16 causes NaN loss |
| Warmup steps | 100 | ~10% of training |
| Doc max length | 512 | Consistent with prior exps |
| Query max length | 512 | Consistent with prior exps |
| Negatives per query | 7 (5 hard + 2 easy) | Balance difficulty |
| Fusion alpha | 0.3 | Best from exp05/exp07 |
| Remining rounds | 2 | Diminishing returns after 3 |
| TIME_BUDGET | 600s per training phase | Per-phase budget |

## Runs

### Run 1: `finetuned-dense-only`
- Description: Fine-tuned 0.6B encoder, dense-only retrieval on Robust04
- Parameter overrides: none (default fine-tuned model)
- Expected output: `runs/finetuned-dense-only.run`

### Run 2: `finetuned-fusion-a03`
- Description: Fine-tuned 0.6B + BM25+Bo1 linear fusion alpha=0.3
- Parameter overrides: fusion with BM25+Bo1 baseline
- Expected output: `runs/finetuned-fusion-a03.run`

### Run 3: `finetuned-expanded-fusion-a03`
- Description: Fine-tuned 0.6B + doc expansion + BM25+Bo1 fusion alpha=0.3
- Parameter overrides: uses expanded corpus embeddings from exp12
- Expected output: `runs/finetuned-expanded-fusion-a03.run`

## Expected Outcome

- **Run 1** (dense-only): MAP@100 ~0.22-0.24 (up from 0.2105 zero-shot)
- **Run 2** (fusion): MAP@100 ~0.29-0.31 (up from 0.2762 zero-shot fusion, target: beat 0.2929)
- **Run 3** (expanded+fusion): MAP@100 ~0.29-0.32 (combining doc expansion with better embeddings)

Rationale: NV-Retriever showed +3-5% gains from positive-aware hard negative mining. RLHN showed +3.2 nDCG from LLM-cleaned negatives. Combined with domain-targeted query selection, we expect meaningful improvement over zero-shot on Robust04.

## Baseline Comparison

| Method | MAP@100 | Source |
|--------|---------|--------|
| BM25+Bo1 | 0.2504 | exp01 |
| 0.6B zero-shot dense | 0.2105 | exp05 |
| 0.6B fusion alpha=0.3 | 0.2762 | exp05 |
| 0.6B fusion + reranker | 0.2827 | exp05 |
| 8B fusion alpha=0.3 | 0.2929 | exp07 (current best) |
| 0.6B + doc expansion fusion | 0.2903 | exp12 |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries
- [x] Training does NOT use Robust04 qrels (relevance judgments)
- [x] Hard negative mining uses MS-MARCO queries only (news-filtered subset)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights
- [x] Robust04 corpus is used only for document encoding/indexing (allowed)
- [x] Retrieval from Robust04 corpus uses MS-MARCO queries only (allowed)
