# Experiment Plan

Prioritized list of experiments to try. Check off as completed.
Reference: `docs/ir-survey-202603.md` for paper details and results.

## Current best

- **exp23 (BM25+Bo1 + Qwen3-Reranker-0.6B)**: BM25+Bo1 top-100 reranked with Qwen3-Reranker-0.6B
- MAP@100 = 0.2552 | MAP@1000 = 0.2552 | nDCG@10 = 0.5292 | recall@100 = 0.4527
- **exp22 (BM25+Bo1)**: pyterrier BM25(k1=0.9,b=0.4) + Bo1(fbDocs=5,fbTerms=30)
- MAP@100 = 0.2504 | MAP@1000 = 0.2968 | nDCG@10 = 0.4662 | recall@100 = 0.4527
- **exp17 (best dense)**: e5-base-v2 fine-tuned + cross-encoder/ms-marco-MiniLM-L-6-v2 rerank top-1000
- MAP@100 = 0.2220 | nDCG@10 = 0.4745 | recall@100 = 0.4111

## Targets

- [x] MAP@100 ≥ 0.20 (milestone 1) — achieved exp17: 0.2220
- [x] MAP@100 > 0.25 (milestone 2) — achieved exp22 BM25+Bo1: 0.2504
- [ ] MAP@100 > 0.30 (beat BM25+PRF MAP@100 significantly)

## Basic methods

- [x] BM25+PRF baseline: exp22 pyterrier BM25(k1=0.9,b=0.4)+Bo1(fbDocs=5,fbTerms=30) → MAP@100=0.2504, MAP@1000=0.2968
- [ ] Utilize query variants to improve query performance (for evaluation, besides standard metrics, also look at good, medium and bad quality title queries and respective results)

## Priority 1: Cross-encoder reranking

- [x] exp16: e5-base-v2 + MiniLM-L-6-v2 rerank top-100 → MAP=0.1996
- [x] exp17: rerank top-1000 → MAP=0.2220 ← current best
- [x] exp18: MiniLM-L-12-v2 reranker → MAP=0.1971 (worse, discarded)
- [x] exp21: cross-encoder/ms-marco-electra-base → SUSPICIOUS nDCG@10=0.075, model fails on long Robust04 docs despite working on short pairs
- [x] exp23: Qwen3-Reranker-0.6B rerank BM25+Bo1 top-100 → MAP@100=0.2552, nDCG@10=0.5292 ← new best
- [ ] Try Qwen3-Reranker-0.6B rerank top-200/1000 (more recall headroom)

## Priority 2: Better backbones

- [ ] Fix Qwen3-Embedding-0.6B (EOS/pooling issue) — zero-shot then fine-tuned
- [ ] `intfloat/e5-large-v2` with batch=64 (fits easily in 46GB L40S)
- [ ] `BAAI/bge-base-en-v1.5` as alternative to e5

## Priority 3: Training improvements

- [ ] Hard negative mining: after initial training, use model to mine hard negatives from Robust04 corpus, then retrain
- [ ] Knowledge distillation: use cross-encoder scores as soft labels for bi-encoder training
- [ ] Gradient accumulation (2-4 steps) to simulate larger effective batch
- [ ] Cosine LR decay (no warmup) — simple schedule

## Priority 4: Hybrid retrieval

- [ ] Dense + BM25 score interpolation (requires implementing BM25 scoring)
- [ ] SPLADE-style sparse augmentation on top of dense

## Priority 5: Advanced methods (from survey)

- [ ] ColBERT-style late interaction (feasible with 46GB L40S)
- [ ] Listwise reranking with LLM (RankGPT-style)
- [ ] Two-stage curriculum: easy negatives → model-mined hard negatives
- [ ] Document expansion: generate synthetic queries per document (docT5query-style)

## Memory constraints (L40S, 46GB)

- e5-base-v2: batch=128, encode_batch=512, doc_len≤256 → ~20GB (plenty of headroom)
- e5-large-v2: batch=64, encode_batch=256, doc_len≤220 → estimate ~30GB (fits comfortably)
- Qwen3-0.6B: batch=64, encode_batch=256 → ~20GB
- Cross-encoder reranking: can keep bi-encoder in memory alongside reranker
- ColBERT-style late interaction: feasible with token-level embeddings in memory

## Notes

- Add experiment ideas here as they come up
- Reference papers from `docs/ir-survey-202603.md` by cite_id
