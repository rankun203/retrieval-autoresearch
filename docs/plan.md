# Experiment Plan

Prioritized list of experiments to try. Check off as completed.
Reference: `docs/ir-survey-202603.md` for paper details and results.

## Current best

- **exp33 (3-phase iterative HN mining + Hybrid RRF)**: e5-base-v2 with 3 rounds of hard negative mining (200s*3), fused with BM25+Bo1 via RRF
- MAP@100 = 0.3483 | nDCG@10 = 0.6333 | recall@100 = 0.5758 | dense-only MAP@100 = 0.3152
- **exp30 (2-phase HN mining + Hybrid RRF, NO reranker)**: e5-base-v2 with hard negative mining, fused with BM25+Bo1 via RRF
- MAP@100 = 0.3275 | MAP@1000 = 0.3896 | nDCG@10 = 0.5921 | recall@100 = 0.5577 | dense-only MAP@100 = 0.2358
- NOTE: Qwen3-Reranker on this HURTS (MAP@100=0.3149) — fusion quality already high enough that reranker degrades it
- **exp27 (Hybrid RRF + Qwen3-Reranker-0.6B)**: dense(e5-base-v2) + BM25+Bo1 fused via RRF, reranked top-100 with Qwen3-Reranker-0.6B
- MAP@100 = 0.2675 | nDCG@10 = 0.5441 | recall@100 = 0.4843
- **exp24 (BM25+Bo1 + Qwen3-Reranker-0.6B top-1000)**: BM25+Bo1 top-1000 reranked with Qwen3-Reranker-0.6B
- MAP@100 = 0.2596 | MAP@1000 = 0.3026 | nDCG@10 = 0.5304 | recall@100 = 0.4660
- **exp23 (BM25+Bo1 + Qwen3-Reranker-0.6B top-100)**: BM25+Bo1 top-100 reranked with Qwen3-Reranker-0.6B
- MAP@100 = 0.2552 | MAP@1000 = 0.2552 | nDCG@10 = 0.5292 | recall@100 = 0.4527
- **exp22 (BM25+Bo1)**: pyterrier BM25(k1=0.9,b=0.4) + Bo1(fbDocs=5,fbTerms=30)
- MAP@100 = 0.2504 | MAP@1000 = 0.2968 | nDCG@10 = 0.4662 | recall@100 = 0.4527
- **exp17 (best dense)**: e5-base-v2 fine-tuned + cross-encoder/ms-marco-MiniLM-L-6-v2 rerank top-1000
- MAP@100 = 0.2220 | nDCG@10 = 0.4745 | recall@100 = 0.4111

## Targets

- [x] MAP@100 ≥ 0.20 (milestone 1) — achieved exp17: 0.2220
- [x] MAP@100 > 0.25 (milestone 2) — achieved exp22 BM25+Bo1: 0.2504
- [x] MAP@100 > 0.30 (beat BM25+PRF MAP@100 significantly) — achieved exp30: 0.3275 (RRF fused, no reranker)
- [ ] MAP@100 > 0.40
- [ ] MAP@100 > 0.50
- [ ] Finish exploring all methods in plan find overall best

## Basic methods

- [x] BM25+PRF baseline: exp22 pyterrier BM25(k1=0.9,b=0.4)+Bo1(fbDocs=5,fbTerms=30) → MAP@100=0.2504, MAP@1000=0.2968
- [ ] Utilize query variants to improve query performance (for evaluation, besides standard metrics, also look at good, medium and bad quality title queries and respective results)
  - [x] exp35-qvariants: BM25+Bo1 + paraphrase variants (Qwen3.5-4B, no thinking) + RRF → MAP@100=0.2754 (above BM25 0.2504, +10%). Query2doc strategy crashed (LLM leaked chain-of-thought into variants). Transformers generate() too slow for 249 queries.
  - [x] exp35-simple-qvariants: BM25+Bo1 + all variants (vLLM thinking) + RRF → MAP@100=0.2441 WORSE than baseline (0.2504). Thinking model generated ~15 para variants/topic (expected 5), diluting RRF with 44 runs. nDCG@10 improved 0.466→0.497 but MAP/recall dropped. Need to cap variants at 5.
  - [x] exp34-query-variants: Current best (3-phase HN + BM25 RRF) + query variants (vLLM, 2220 variants) → MAP@100=0.3285 WORSE than baseline 0.3450 (no variants). Multi-variant RRF (18 runs) dilutes fusion. Dense-only MAP@100=0.3085 (vs exp33 0.3152). Conclusion: query variants hurt both BM25-only and hybrid pipelines when added via RRF.
  - [x] exp36-qvariants-rerank: BM25+Bo1 + query variants (para+q2d+decomp) RRF → Qwen3-Reranker top-100 → MAP@100=0.2667. Variants+rerank (0.2667) > variants-only (0.2635) > rerank-only (0.2537) > baseline (0.2504). Variants help recall (0.4527→0.4793), reranker helps precision. Still far below dense+BM25 hybrid (0.3483).

## Priority 1: Cross-encoder reranking

- [x] exp16: e5-base-v2 + MiniLM-L-6-v2 rerank top-100 → MAP=0.1996
- [x] exp17: rerank top-1000 → MAP=0.2220 ← current best
- [x] exp18: MiniLM-L-12-v2 reranker → MAP=0.1971 (worse, discarded)
- [x] exp21: cross-encoder/ms-marco-electra-base → SUSPICIOUS nDCG@10=0.075, model fails on long Robust04 docs despite working on short pairs
- [x] exp23: Qwen3-Reranker-0.6B rerank BM25+Bo1 top-100 → MAP@100=0.2552, nDCG@10=0.5292 ← new best
- [x] exp24: Qwen3-Reranker-0.6B rerank BM25+Bo1 top-1000 → MAP@100=0.2596, MAP@1000=0.3026, nDCG@10=0.5304, recall@100=0.4660 ← new best (marginal gain over exp23, +75min runtime)

## Priority 2: Better backbones

- [ ] Fix Qwen3-Embedding-0.6B (EOS/pooling issue) — zero-shot then fine-tuned
- [x] exp25: `intfloat/e5-large-v2` batch=32 doc_len=220 → dense MAP@100=0.1605 (worse than e5-base 0.1772), +rerank MAP@100=0.2219 (matches e5-base+rerank). Discarded — batch=32 hurts contrastive learning vs batch=64 on e5-base
- [ ] exp26: e5-large-v2 with doc_len=512 for encoding only (train at 220, encode at 512 — NOTE: train/encode length mismatch, model sees longer docs at retrieval than training)
- [ ] `BAAI/bge-base-en-v1.5` as alternative to e5

## Priority 3: Training improvements

- [x] exp33: 3-phase iterative HN mining (200s*3) → dense MAP@100=0.3152 (up from 0.2358), hybrid RRF MAP@100=0.3483 (new best). Extra mining round helps dense +0.08, fused +0.02. Loss spiky on R04 batches (0.3→0.9 on HN batches)
- [x] exp30: Hard negative mining: 2-phase (300s MS-MARCO + 300s mixed MS-MARCO/Robust04 HN) → dense MAP@100=0.236 (up from 0.180), hybrid RRF MAP@100=0.3275 (best w/o reranker), +reranker MAP@100=0.3149 (reranker hurts!)
- [x] exp29: MarginMSE distillation → dense MAP@100=0.156 (worse than 0.180 baseline). Loss flat ~10.5, scaling broken. Discarded
- [x] exp31: Gradient accum (4 steps, eff batch=512) + cosine LR (2e-5→0) → fused MAP@100=0.286 (worse than exp30 0.3275). Too aggressive — discarded

## Priority 4: Hybrid retrieval

- [x] exp27: Dense(e5-base-v2) + BM25+Bo1 RRF fusion → Qwen3-Reranker top-100 → MAP@100=0.2675, nDCG@10=0.5441, recall@100=0.4843 ← new best
- [x] exp28: RRF top-1000 reranking → MAP@100=0.2649, nDCG@10=0.5358 (WORSE than top-100). Reranker degrades with more candidates — top-100 is optimal cutoff
- [ ] Try linear interpolation instead of RRF
- [ ] SPLADE-style sparse augmentation on top of dense

## Priority 5: Advanced methods (from survey)

- [x] exp32: ColBERT late interaction (128d token embs, MaxSim rescore top-200) + 2-phase HN mining + BM25 RRF → fused MAP@100=0.3038 (below exp30 0.3275). ColBERT MaxSim=0.2079 worse than mean-pooled=0.2330. In-batch ColBERT loss unstable with R04 batches. Discarded
- [ ] Listwise reranking with LLM (RankGPT-style)
- [ ] Two-stage curriculum: easy negatives → model-mined hard negatives
- [ ] Document expansion: generate synthetic queries per document (docT5query-style)

## Priority 6: Agentic retrieval (see docs/agentic-retrieval-research.md)

- [ ] Agentic retrieval with Qwen3.5-9B: multi-round iterative retrieval where LLM examines top docs each round, scores relevance, generates new aspect-exploring queries, accumulates ~100 high-quality docs. Inspired by PRISM, SmartSearch, IRCoT. Track eval_dur carefully — heaviest pipeline.

## Memory constraints (L40S, 46GB)

- e5-base-v2: batch=128, encode_batch=512, doc_len≤256 → ~20GB (plenty of headroom)
- e5-large-v2: batch=64, encode_batch=256, doc_len≤220 → estimate ~30GB (fits comfortably)
- Qwen3-0.6B: batch=64, encode_batch=256 → ~20GB
- Cross-encoder reranking: can keep bi-encoder in memory alongside reranker
- ColBERT-style late interaction: feasible with token-level embeddings in memory

## Notes

- Add experiment ideas here as they come up
- Reference papers from `docs/ir-survey-202603.md` by cite_id
