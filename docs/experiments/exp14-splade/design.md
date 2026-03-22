# exp14-splade: SPLADE Learned Sparse Retrieval

## Literature Review

### SPLADE-v3: New Baselines for SPLADE
- **Authors**: Carlos Lassance, Herve Dejean, Thibault Formal, Stephane Clinchant
- **Venue**: arXiv 2024
- **URL**: https://arxiv.org/abs/2403.06789
- **Key takeaways**:
  - SPLADE-v3 achieves MRR@10 40.2 on MS MARCO (vs SPLADE++SelfDistil 37.6)
  - BEIR nDCG@10 51.7 (good zero-shot transfer)
  - TREC DL19 nDCG@10 72.3, DL20 75.4
  - Training uses KL-Div + MarginMSE with 8 hard negatives per query
  - Architecture: BertForMaskedLM with max-pooling and log(1+ReLU(logits)) transformation
  - Output: 30,522-dimensional sparse vector (one per vocab token)

### SPLADE Model Card (naver/splade-v3)
- **URL**: https://huggingface.co/naver/splade-v3
- **Key takeaways**:
  - Base model: naver/splade-cocondenser-selfdistil
  - Max sequence length: 512 (256 for eval in model card)
  - Separate encode_query() and encode_document() in sentence-transformers
  - Dot product similarity for scoring
  - CC-BY-NC-SA-4.0 license

### SPLADE-cocondenser-ensembledistil
- **URL**: https://huggingface.co/naver/splade-cocondenser-ensembledistil
- **Key takeaways**:
  - MRR@10 38.3 on MS MARCO dev, R@1000 98.3
  - BertForMaskedLM architecture, vocab size 30,522
  - Raw forward pass: `torch.max(torch.log(1 + torch.relu(output.logits)) * attention_mask.unsqueeze(-1), dim=1)`

### DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities
- **cite_id**: nguyen2024dyvo (from survey)
- **Key takeaway**: Robust04 nDCG@10 54.39 with entity-augmented SPLADE -- shows SPLADE can work on news domain

### Implementation Approach
- Use sentence-transformers SparseEncoder API (clean, correct query/doc separation)
- SPLADE models are small (~110M params BERT-base), fitting easily in 46GB VRAM
- Sparse vectors scored via dot product; use scipy sparse matrices for efficient storage/retrieval

## Goal

Test SPLADE learned sparse retrieval as an alternative to BM25 and as a complement to both BM25 and dense retrieval. SPLADE learns term importance weights via MLM head, enabling semantic expansion beyond exact term matching while retaining sparse retrieval efficiency.

## Hypothesis

1. SPLADE-v3 should outperform plain BM25 on Robust04 because it learns semantic term expansion (e.g., "cancer" -> "tumor", "oncology") while retaining lexical precision.
2. SPLADE + BM25 fusion should outperform either alone because SPLADE captures different relevance signals (learned expansion vs statistical term weighting with PRF).
3. SPLADE + dense (Qwen3-0.6B) + BM25 triple fusion may further improve recall by combining three complementary retrieval paradigms.
4. The recall@100 bottleneck (~0.49-0.51) may improve because SPLADE retrieves different relevant documents than BM25 or dense alone.

## Method

### Pipeline
1. **BM25+Bo1**: Existing cached Terrier pipeline (top-1000)
2. **SPLADE encoding**: Encode all 528K docs + 249 queries using naver/splade-v3 via sentence-transformers SparseEncoder
3. **SPLADE retrieval**: Store sparse vectors as scipy CSR matrix, dot product for top-1000 retrieval
4. **Dense encoding**: Load cached Qwen3-Embedding-0.6B embeddings
5. **Fusion runs**: Test various combinations

### SPLADE Forward Pass (via sentence-transformers)
```python
from sentence_transformers import SparseEncoder
model = SparseEncoder("naver/splade-v3")
query_embeddings = model.encode_query(queries)  # sparse 30522-dim
doc_embeddings = model.encode_document(documents)  # sparse 30522-dim
```

### Retrieval Strategy
- Store sparse doc vectors as scipy.sparse CSR matrix (528K x 30522)
- Query sparse vectors also as sparse vectors
- Dot product via sparse matrix multiplication for retrieval
- Top-K selection per query

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SPLADE model | naver/splade-v3 | Best SPLADE variant, MRR@10 40.2 |
| Doc max length | 256 | Match model card eval setting |
| Query max length | 256 | Match model card eval setting |
| Encode batch size | 64 | Conservative for 528K docs on BERT-base |
| SPLADE top-K | 1000 | Standard retrieval depth |
| BM25 config | k1=0.9, b=0.4, Bo1 fb_docs=5, fb_terms=30 | Existing baseline |
| Dense model | Qwen/Qwen3-Embedding-0.6B | Cached embeddings available |
| Fusion alphas | Various (see runs) | Sweep over fusion weights |

## Runs

### Run 1: splade-v3-zeroshot
- **Description**: SPLADE-v3 standalone retrieval on Robust04
- **Expected output**: `runs/splade-v3-zeroshot.run`

### Run 2: splade-bm25-fusion-a{03,05,07}
- **Description**: Linear fusion of SPLADE + BM25+Bo1 at alpha=0.3, 0.5, 0.7 (SPLADE weight)
- **Expected output**: `runs/splade-bm25-fusion-a{03,05,07}.run`

### Run 3: triple-fusion-*
- **Description**: Triple fusion of SPLADE + Qwen3-0.6B dense + BM25+Bo1
- **Parameter overrides**: Sweep weights (splade, dense, bm25)
- **Expected output**: `runs/triple-fusion-*.run`

### Run 4: splade-dense-fusion-a05
- **Description**: SPLADE + dense only fusion (no BM25), testing SPLADE as BM25 replacement
- **Expected output**: `runs/splade-dense-fusion-a05.run`

## Expected Outcome

- **SPLADE standalone**: MAP@100 ~0.18-0.24 (domain mismatch: trained on MS MARCO, tested on news)
- **SPLADE + BM25 fusion**: MAP@100 ~0.26-0.28 (complementary signals)
- **Triple fusion**: MAP@100 ~0.28-0.30 (three complementary retrievers)
- **Key question**: Does SPLADE improve recall@100 beyond the ~0.50 plateau?

## Baseline Comparison

| Method | MAP@100 | recall@100 |
|--------|---------|------------|
| BM25+Bo1 | 0.2504 | 0.4527 |
| Qwen3-0.6B + BM25 fusion | 0.2762 | 0.4920 |
| Qwen3-0.6B + BM25 + doc expansion | 0.2903 | 0.5057 |
| Qwen3-8B + BM25 fusion | 0.2929 | 0.5103 |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (zero-shot, no training)
- [x] Training does NOT use Robust04 qrels (zero-shot, no training)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- zero-shot)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (pre-trained model used as-is)
