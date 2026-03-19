# BEIR Benchmark Evaluation — Design Notes

## Idea

Add a BEIR evaluation agent that runs after experiment review (pass/accept), evaluating the retrieval system against BEIR benchmark datasets and appending results to `beir-results.tsv`.

## Concerns

### Time/GPU Cost
- BEIR has ~18 datasets. Encoding each dataset's corpus (some 500K+ docs) takes 10-30 min per dataset.
- Full BEIR evaluation = 3-8 hours per experiment — longer than most experiments themselves.

### Pipeline Mismatch
- Our best systems are multi-stage pipelines (BM25 + dense + fusion + reranking).
- BEIR benchmarks individual retrievers. Evaluating the full pipeline requires BM25 indices for every BEIR dataset, which is a big setup cost.

### Diminishing Value for Zero-Shot Models
- Most experiments use pre-trained models (Qwen3-Embedding, Qwen3-Reranker) whose BEIR/MTEB scores are already published on model cards.
- Re-measuring known quantities adds hours for no new information.

### When BEIR IS Valuable
- Once we start **fine-tuning** models (Priority 3: MarginMSE, hard negative mining), BEIR evaluation checks if fine-tuning on MS-MARCO hurts generalization.
- That's the critical use case — detecting overfitting to in-domain data.

## Suggested Approach

1. **Skip BEIR for zero-shot experiments** — link to published MTEB scores instead.
2. **Add BEIR evaluation only for fine-tuned experiments** — where generalization is a real concern.
3. **Use a subset** (3-5 datasets) to keep evaluation under 1 hour:
   - TREC-COVID (biomedical)
   - NFCorpus (nutrition/medical)
   - SciFact (scientific claims)
   - FiQA (financial QA)
   - ArguAna (argument retrieval)
4. **Evaluate only the dense retriever component**, not the full multi-stage pipeline.

## Implementation Plan (when ready)

- Create `.claude/agents/experiment-beir-eval.md`
- Runs after review, before cleanup
- Uses `beir` Python package for standardized evaluation
- Appends to `beir-results.tsv` with columns: experiment, dataset, ndcg@10, map@100, recall@100, model, notes
- Triggered only when the experiment involves a fine-tuned model (orchestrator decides)
