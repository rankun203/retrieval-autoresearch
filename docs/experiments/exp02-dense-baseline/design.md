# exp02-dense-baseline: Zero-shot Dense Retrieval with e5-base-v2

## Goal

Establish a dense retrieval baseline on Robust04 using e5-base-v2 with no fine-tuning.
This gives us the zero-shot dense retrieval floor to compare against BM25 and future
fine-tuned models.

## Hypothesis

Zero-shot e5-base-v2 will underperform BM25+Bo1 PRF (MAP@100=0.2504) on Robust04 since
the model was trained on general-domain data and Robust04 contains newswire text with
keyword-heavy TREC topics. Dense retrievers typically need domain adaptation to match
strong lexical baselines on Robust04. We expect MAP@100 in the 0.15-0.22 range.

## Method

1. Load Robust04 corpus (528K docs) and queries (249)
2. Encode all documents using e5-base-v2 with "passage: " prefix
3. Encode all queries with "query: " prefix
4. Build FAISS flat index (exact search) over document embeddings
5. Retrieve top-1000 for each query
6. Evaluate with pytrec_eval via prepare.py

No training or fine-tuning. Pure inference pipeline.

## Key Parameters

| Parameter        | Value                    |
|------------------|--------------------------|
| encoder_model    | intfloat/e5-base-v2      |
| embedding_dim    | 768                      |
| doc_max_length   | 256 tokens               |
| query_max_length | 64 tokens                |
| encode_batch     | 512                      |
| doc_prefix       | "passage: "              |
| query_prefix     | "query: "                |
| top_k            | 1000                     |
| faiss_index      | FlatIP (exact cosine)    |
| normalize        | Yes (L2 norm before IP)  |
| fp16_encode      | Yes (autocast)           |

## Runs

### Run: `e5-base-v2-zeroshot`
- **Description**: Zero-shot retrieval with e5-base-v2, exact FAISS search
- **Parameter overrides**: None (defaults above)
- **Expected outputs**: `runs/e5-base-v2-zeroshot.run`

## Expected Outcome

- MAP@100: ~0.17-0.22 (below BM25+Bo1's 0.2504)
- nDCG@10: ~0.35-0.42
- recall@100: ~0.35-0.42

Rationale: Zero-shot dense retrievers typically lag behind tuned BM25 on Robust04.
E5-base-v2 is a solid general-purpose model but has not been adapted to newswire/TREC.

## Baseline Comparison

| Method              | MAP@100 |
|---------------------|---------|
| BM25+Bo1 PRF        | 0.2504  |
| e5-base-v2 (expected) | ~0.18 |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training at all)
- [x] Training does NOT use Robust04 qrels (no training at all)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (zero-shot, no weight updates)
