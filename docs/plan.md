# Experiment Plan

Prioritized list of experiments to try. Check off as completed.
Reference: `docs/ir-survey-202603.md` for paper details and results.

## Current best

exp07-qwen3-embed-8b: BM25+Bo1 + Qwen3-Embedding-8B linear fusion (alpha=0.3), MAP@100=0.2929
exp12-doc-expansion (0.6B class best): BM25+Bo1 + Qwen3-Embedding-0.6B fusion-a03-expanded (Qwen2.5-3B-Instruct doc expansion), MAP@100=0.2903, recall@100=0.5057

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
- [x] Fusion baseline (RRF)

## Priority 1: Cross-encoder reranking

- [x] Dense encoder + cross-encoder rerank (e.g. MiniLM-L-6-v2, Qwen3-Reranker-0.6B with correct EOS/last-token pooling), zero-shot then fine-tuned — DISCARD (exp03): zero-shot MiniLM/BGE/Qwen3 (broken) did not beat baseline. KEEP (exp03b): Qwen3-Reranker-0.6B fixed with correct EOS/last-token pooling; ml768 top-1000 achieves MAP@100=0.2668 (new best), nDCG@10=0.5288, MAP@1000=0.3120; longer max_length and deeper reranking pool both help
- [x] Rerank top-100 vs top-1000 comparison — top-1000 consistently beats top-100 across all max_length settings for Qwen3-Reranker
- [x] Instruction tuning for Qwen3-Reranker — DISCARD (exp04): news-short instruction achieved MAP@100=0.2667 (best variant, +2.7% over default 0.2598 for this config), but still just below current best 0.2668; instruction choice matters (news-short > general > default) but gain insufficient to surpass exp03b result
- [x] Rerank at depth 1000 with news-short instruction — SKIPPED: depth-1000 reranking takes too long (~80 min per run) and exp07 showed reranking can hurt with strong embeddings

## Priority 2: Better backbones

- [x] `Qwen/Qwen3-Embedding-0.6B` — 0.6B params, MTEB-en retrieval 61.83, 32K context, ~1.2GB
- [x] `Qwen/Qwen3-Embedding-8B` — 8B params, #1 MTEB multilingual (70.58), 32K context, ~16GB — KEEP (exp07): BM25+Bo1 linear fusion alpha=0.3 achieves MAP@100=0.2929 (new best); reranking with Qwen3-Reranker HURT performance (linear-a03-reranked < linear-a03), suggesting 8B embeddings already capture sufficient semantic signal
- [x] `jinaai/jina-embeddings-v4` — SKIPPED: multimodal model with non-standard API, MTEB-en 55.97 is lower than Qwen3-0.6B (61.83)
- [x] `nomic-ai/modernbert-embed-base` — DISCARD (exp10): tested in exp10-backbone-sweep, fusion MAP@100 ranged 0.2723-0.2802, did not beat Qwen3-8B fusion (0.2929)
- [x] `intfloat/e5-large-v2` with proper batch size — DISCARD (exp10): tested in exp10-backbone-sweep, fusion MAP@100 ranged 0.2723-0.2802, did not beat Qwen3-8B fusion (0.2929)
- [x] `BAAI/bge-base-en-v1.5` as alternative to e5 — DISCARD (exp10): tested in exp10-backbone-sweep, fusion MAP@100 ranged 0.2723-0.2802, did not beat Qwen3-8B fusion (0.2929)
- [ ] Utilize query variants to improve query performance (for evaluation, besides standard metrics, also look at good, medium and bad quality title queries and respective results)

## BM25/PRF parameter tuning

- [x] BM25/PRF parameter sweep — DISCARD (exp09): 210 configs tested across BM25-only, RM3, Bo1, KL variants; best KL MAP@100=0.2503, confirms default baseline params near-optimal; no meaningful gain from tuning b/k1/feedback terms

## Priority 3: Training improvements

- [ ] Hard negative mining (iterative, multi-phase) — use Qwen3.5-9B as relevance judge (never touch qrels for mining)
- [ ] MarginMSE distillation from cross-encoder
- [ ] Gradient accumulation + cosine LR scheduling
- [ ] Two-stage curriculum: easy negatives → hard negatives

## Priority 4: Hybrid retrieval

- [x] Dense + BM25 fusion (RRF and linear interpolation) — KEEP (exp05): linear fusion alpha=0.5 + Qwen3-Reranker achieved MAP@100=0.2827 (new best), +5.9% over exp03b; fusion consistently outperforms either component alone
- [ ] SPLADE-style sparse augmentation on top of dense

## Priority 5: Advanced methods (from survey)

- [x] ColBERT late interaction — DISCARD (exp06): all runs below current best; implementation bug: projection head not loaded correctly, token embeddings were not properly projected. Should be retried with correct ColBERT model loading (load full checkpoint including projection head).
- [x] ColBERT v2 via PyLate (correct implementation) — DISCARD (exp08): implementation now correct via PyLate, but ColBERTv2 standalone achieves only MAP@100=0.1844 due to domain mismatch (trained on MS-MARCO, tested on Robust04 news). BM25+ColBERT fusion reaches 0.2870 and fusion+rerank reaches same 0.2870, both below current best 0.2929. Domain mismatch limits ColBERT gains on news domain.
- [x] Listwise reranking with LLM (RankGPT-style) — DISCARD (exp11): RankZephyr 7B listwise reranking on 0.6B fusion, best MAP@100=0.2822 (rerank-top50), nDCG@10 improved +8.5% but MAP limited by recall@100 bottleneck; reranking cannot recover docs not in the initial retrieval pool
- [x] Document expansion: generate synthetic queries per document (docT5query-style) — KEEP (exp12): Qwen2.5-3B-Instruct generates 5 queries/doc, fusion-a03-expanded MAP@100=0.2903 (new 0.6B-class best), recall@100=0.5057 (+2.8% over baseline); doc expansion improves recall but 0.6B embeddings still trail Qwen3-8B fusion (0.2929)

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
