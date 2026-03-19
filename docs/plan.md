# Experiment Plan

Prioritized list of experiments to try. Check off as completed.
Reference: `docs/ir-survey-202603.md` for paper details and results.

## Current best

(none yet)

## Targets

- [ ] MAP@100 ≥ 0.20
- [ ] MAP@100 > 0.25
- [ ] MAP@100 > 0.30
- [ ] MAP@100 > 0.40
- [ ] MAP@100 > 0.50
- [ ] Finish exploring all methods in plan, find overall best

## Basic methods

- [ ] BM25+PRF baseline via pyterrier
- [ ] Utilize query variants to improve query performance (for evaluation, besides standard metrics, also look at good, medium and bad quality title queries and respective results)

## Priority 1: Cross-encoder reranking

- [ ] Dense encoder + cross-encoder rerank (e.g. MiniLM-L-6-v2, Qwen3-Reranker-0.6B)
- [ ] Rerank top-100 vs top-1000 comparison

## Priority 2: Better backbones

- [ ] Fix Qwen3-Embedding-0.6B (EOS/last-token pooling) — zero-shot then fine-tuned
- [ ] `intfloat/e5-large-v2` with proper batch size
- [ ] `BAAI/bge-base-en-v1.5` as alternative to e5

## Priority 3: Training improvements

- [ ] Hard negative mining (iterative, multi-phase) — NOTE: must use proper train/test split, no test query/qrel leakage
- [ ] MarginMSE distillation from cross-encoder
- [ ] Gradient accumulation + cosine LR scheduling
- [ ] Two-stage curriculum: easy negatives → hard negatives

## Priority 4: Hybrid retrieval

- [ ] Dense + BM25 fusion (RRF and linear interpolation)
- [ ] SPLADE-style sparse augmentation on top of dense

## Priority 5: Advanced methods (from survey)

- [ ] ColBERT late interaction
- [ ] Listwise reranking with LLM (RankGPT-style)
- [ ] Document expansion: generate synthetic queries per document (docT5query-style)

## Priority 6: Agentic retrieval (see docs/agentic-retrieval-research.md)

- [ ] Agentic retrieval with LLM: multi-round iterative retrieval where LLM examines top docs, scores relevance, generates new aspect-exploring queries

## Diversity / Pooling Analysis

- [ ] Portfolio diversity study across retrieval systems (maximize unique relevant docs for judging pools)

## Memory constraints (L40S, 46GB)

- e5-base-v2: batch=128, encode_batch=512, doc_len≤256 → ~20GB
- e5-large-v2: batch=64, encode_batch=256, doc_len≤220 → ~30GB
- Qwen3-0.6B: batch=64, encode_batch=256 → ~20GB
- Cross-encoder reranking: can keep bi-encoder in memory alongside reranker

## Notes

- Add experiment ideas here as they come up
- Reference papers from `docs/ir-survey-202603.md` by cite_id
- IMPORTANT: Hard negative mining must NOT use test queries or test qrels for training. Use proper train/test splits or cross-validation to avoid data leakage.
