# LLM Relevance Judgment & Hard Negative Mining — Research Notes

Compiled from deep research for exp13 (hard negative mining experiment).

## LLM Positivity Bias

- LLMs systematically over-predict relevance vs human judges (Alaofi et al. SIGIR 2024)
- Non-relevance labels are more trustworthy than positive labels
- False positives correlate with query term presence in passages (keyword overlap bias)
- Bias worst at intermediate grades; truly irrelevant docs are identified ~75% accurately
- ~70% of BM25 hard negatives from MS-MARCO are actually false negatives (NV-Retriever, Qwen3 findings)

## Best Prompting Strategy: Multi-Criteria Decomposition (TREMA)

Won LLMJudge challenge. LLaMA-3-8B with this approach beat GPT-4o with naive prompting.

Four criteria scored independently (0-3 each):
1. **Exactness**: How precisely does the passage answer the query?
2. **Topicality**: Is the passage about the same subject as the query?
3. **Coverage**: How much of the passage is dedicated to discussing the query?
4. **Contextual Fit**: Does the passage provide relevant background or context?

Aggregation: Sum (0-12), map: 10-12→grade 3, 7-9→grade 2, 5-6→grade 1, 0-4→grade 0.

Source: Farzi & Dietz, ICTIR 2025; https://arxiv.org/html/2507.09488

## Single-Call Alternative: UMBRELA-Style

Step-by-step reasoning in single call:
1. What is the underlying information need?
2. How well does content match this need?
3. Is the passage primarily about this topic or mentions it in passing?

Scale: 0=nothing to do, 1=related but doesn't answer, 2=some answer but unclear, 3=dedicated and exact answer.

Source: UMBRELA, https://arxiv.org/html/2406.06519v1

## Debiasing Techniques

- Add anti-leniency instruction: "A passage that merely mentions query terms without substantively addressing the information need should receive 0 or 1."
- Collect graded 0-3, then binarize (more reliable than direct binary)
- Monitor label distributions: expect ~40-50% grade 0, 25-30% grade 1, 15-20% grade 2, 5-10% grade 3
- Temperature=0 for consistency

## Hard Negative Mining Pipeline (3-Phase)

### Phase 1: Cross-Encoder Teacher + Positive-Aware Filtering
- Retrieve top-500 per MS-MARCO query from Robust04 via BM25
- Score with Qwen3-Reranker-0.6B cross-encoder
- Hard negatives: CE score < 0.95 × positive_score AND > 0.3
- Train with MarginMSE loss

### Phase 2: LLM False Negative Cleanup
- Judge borderline negatives with LLM (Qwen3-8B or similar)
- Contrastive prompt: reasons NOT relevant first, then relevant, then verdict
- Remove any the LLM judges as relevant

### Phase 3: Iterative Remining (R-GPL style)
- Re-encode corpus with updated model
- Mine new harder negatives
- 2-3 rounds

## Key Papers

- NV-Retriever (NVIDIA 2024): positive-aware mining, MTEB Retrieval #1
- RLHN (Castorini, EMNLP 2025): cascading LLM relabeling, +3.2 nDCG on OOD
- R-GPL (Jan 2025): dynamic remining during training
- Gecko (Google DeepMind 2024): LLM relabeling for distillation
- SyNeg (Dec 2024): mix synthetic + mined negatives
- "Don't Retrieve, Generate" (Apr 2025): corpus-mined > synthetic negatives
- GPL (NAACL 2022): foundational cross-encoder teacher pipeline

## Practical Settings

- Model: 8B locally via vLLM is sufficient (proven in LLMJudge)
- Pointwise, one doc per prompt, with CoT reasoning
- 768 tokens per doc (news articles need sufficient context)
- Let vLLM batch internally (continuous batching)
- For our setting: 500 news-selected MS-MARCO queries × 500 docs = 250K judgments
- Grade 1 docs = ideal hard negatives (topically related but not relevant)
