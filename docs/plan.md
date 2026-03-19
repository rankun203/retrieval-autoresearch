# Experiment Plan

Prioritized list of experiments to try. Check off as completed.
Reference: `docs/ir-survey-202603.md` for paper details and results.

## Current best

exp03b-qwen3-rerank-fix: Qwen3-Reranker-0.6B (ml768, top-1000), MAP@100=0.2668

## Targets

- [x] MAP@100 ≥ 0.20
- [x] MAP@100 > 0.25
- [ ] MAP@100 > 0.30
- [ ] MAP@100 > 0.40
- [ ] MAP@100 > 0.50
- [ ] Finish exploring all methods in plan, find overall best

## Baselines

- [x] BM25 baseline via pyterrier
- [x] BM25+PRF baseline via pyterrier
- [x] Dense retrieval baseline with e5-base-v2 — DISCARD: MAP@100=0.1697, below BM25+Bo1 baseline (0.2504); zero-shot e5-base-v2 insufficient for Robust04
- [ ] Fusion baseline (RRF)

## Priority 1: Cross-encoder reranking

- [x] Dense encoder + cross-encoder rerank (e.g. MiniLM-L-6-v2, Qwen3-Reranker-0.6B with correct EOS/last-token pooling), zero-shot then fine-tuned — DISCARD (exp03): zero-shot MiniLM/BGE/Qwen3 (broken) did not beat baseline. KEEP (exp03b): Qwen3-Reranker-0.6B fixed with correct EOS/last-token pooling; ml768 top-1000 achieves MAP@100=0.2668 (new best), nDCG@10=0.5288, MAP@1000=0.3120; longer max_length and deeper reranking pool both help
- [x] Rerank top-100 vs top-1000 comparison — top-1000 consistently beats top-100 across all max_length settings for Qwen3-Reranker

## Priority 2: Better backbones

- [ ] `Qwen/Qwen3-Embedding-0.6B` — 0.6B params, MTEB-en retrieval 61.83, 32K context, ~1.2GB
- [ ] `Qwen/Qwen3-Embedding-8B` — 8B params, #1 MTEB multilingual (70.58), 32K context, ~16GB
- [ ] `jinaai/jina-embeddings-v4` — 3.8B params, multimodal, MTEB-en 55.97, ~8GB
- [ ] `nomic-ai/modernbert-embed-base` — small/fast, 8K context
- [ ] `intfloat/e5-large-v2` with proper batch size
- [ ] `BAAI/bge-base-en-v1.5` as alternative to e5
- [ ] Utilize query variants to improve query performance (for evaluation, besides standard metrics, also look at good, medium and bad quality title queries and respective results)

## Priority 3: Training improvements

- [ ] Hard negative mining (iterative, multi-phase) — use Qwen3.5-9B as relevance judge (never touch qrels for mining)
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
- Qwen3-Embedding-0.6B: ~1.2GB fp16, encode_batch=256 → ~5GB
- Qwen3-Embedding-8B: ~16GB fp16, encode_batch=64 → ~25GB
- Jina-v4 (3.8B): ~8GB fp16, encode_batch=128 → ~15GB
- Qwen3.5-9B (for HN mining judge): ~18GB fp16
- Cross-encoder reranking: can keep bi-encoder in memory alongside reranker

## Notes

- Add experiment ideas here as they come up
- Reference papers from `docs/ir-survey-202603.md` by cite_id
- IMPORTANT: Hard negative mining must NOT use test queries or test qrels for training. Use proper train/test splits or cross-validation to avoid data leakage.
