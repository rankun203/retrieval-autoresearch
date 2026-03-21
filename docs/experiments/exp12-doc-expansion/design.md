# exp12-doc-expansion: Document Expansion via Synthetic Query Generation

## Literature Review

### docTTTTTquery / doc2query (Nogueira & Lin, 2019)
- **Paper**: "From doc2query to docTTTTTquery" / "Document Expansion by Query Prediction"
- **URL**: https://arxiv.org/abs/1904.08375
- **Key idea**: Train a seq2seq model (T5-base) to predict queries a document would answer. Generate ~10 queries per document via top-k sampling, append them to the original text, then index with BM25.
- **Results**: On MS MARCO passage ranking, doc expansion improved MRR@10 from 0.184 (BM25) to 0.272. Expanded index approaches neural re-ranker effectiveness at a fraction of the cost.
- **Implementation**: Feed document text as input to T5, use top-k sampling (k=10) to generate diverse queries. Append all generated queries to doc text before indexing.

### Doc2Query-- (Gospodinov et al., ECIR 2023)
- **Paper**: "Doc2Query--: When Less is More"
- **URL**: https://arxiv.org/abs/2301.03266
- **Key idea**: Add a filtering stage after query generation -- use a cross-encoder to score each (generated_query, document) pair and remove low-quality queries. Improves precision by 16% while reducing index size by 33%.
- **Takeaway for us**: Filtering helps precision but HURTS recall (confirmed by SIGIR 2024 study). Since our bottleneck is recall@100, we should NOT filter.

### Revisiting Document Expansion and Filtering (Mansour et al., SIGIR 2024)
- **Paper**: "Revisiting Document Expansion and Filtering for Effective First-Stage Retrieval"
- **URL**: https://dl.acm.org/doi/10.1145/3626772.3657850
- **Key finding**: Filtering in Doc2Query-- actually HARMS recall-based metrics across test collections. Simple unfiltered expansion matches or exceeds filtered approaches for recall-oriented evaluation.
- **Implication**: We should use unfiltered expansion -- generate queries and append them all.

### Doc2Query++ (2025)
- **Paper**: "Doc2Query++: Topic-Coverage based Document Expansion"
- **URL**: https://arxiv.org/abs/2510.09557
- **Key idea**: Use BERTopic to infer document topics, extract keywords, guide LLM query generation for diversity.
- **Results**: Significant gains in MAP, nDCG@10, and Recall@100 on both sparse and dense retrieval.
- **Takeaway**: Topic-guided generation improves diversity but adds complexity. We use a simpler approach: prompt the LLM to generate diverse queries covering different aspects.

### Key Implementation Decisions from Literature
1. **Number of queries**: 5 per document (balance coverage vs. generation time for 528K docs)
2. **No filtering**: SIGIR 2024 confirms filtering hurts recall
3. **Diverse generation**: Temperature sampling (0.7) for varied queries
4. **Benefits for both sparse AND dense**: Expanded text helps BM25 (vocabulary enrichment) and dense models (richer semantic content)

## Goal

Generate synthetic queries for each of the 528K Robust04 documents using an LLM, append them to the document text, then run the standard BM25+Bo1 + Qwen3-Embedding-0.6B fusion pipeline on the expanded corpus. This should improve both BM25 and dense recall by enriching document representations with query-like language, directly addressing the recall@100 bottleneck (~0.49) identified in exp11.

## Hypothesis

1. **Recall bottleneck**: exp11 showed recall@100 is ~0.49 with the current fusion pipeline. Reranking cannot help documents not in the top-100. Document expansion should bring more relevant documents into the top-100 by adding query-like terms.
2. **Vocabulary mismatch**: Many relevant documents fail to rank because they don't contain the exact query terms. Synthetic queries bridge this gap for BM25.
3. **Dense embedding enrichment**: For dense retrieval, appended queries add explicit query-intent language that shifts document embeddings closer to relevant query embeddings in the shared space.
4. **Unfiltered expansion is best for recall**: Per SIGIR 2024, filtering hurts recall. We use all generated queries.

## Method

### Phase 1: Generate synthetic queries (generate_queries.py)
1. Load the 528K Robust04 corpus
2. Use vLLM with Qwen2.5-3B-Instruct to generate 5 synthetic queries per document
3. Prompt: instruct the model to generate diverse search queries the article would answer
4. Save expanded corpus to cache as JSON lines
5. Free the LLM from GPU

### Phase 2: Retrieval pipeline (train.py)
1. Load the expanded corpus from cache
2. Run BM25+Bo1 on expanded documents (rebuild Terrier index with expanded text)
3. Run Qwen3-Embedding-0.6B dense retrieval on expanded documents
4. Linear fusion (alpha=0.3 and alpha=0.5)
5. Evaluate all runs

### Why two scripts?
vLLM and PyTerrier/Transformers can conflict (CUDA context issues). Separating generation from retrieval avoids these problems, following the pattern from exp11.

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LLM for generation | Qwen/Qwen2.5-3B-Instruct | ~6GB fp16, fast with vLLM |
| Queries per document | 5 | Balance coverage vs. generation time |
| Generation temperature | 0.7 | Encourage diverse queries |
| Generation max_tokens | 200 | ~40 tokens per query x 5 |
| vLLM tensor_parallel | 1 | Single GPU sufficient for 3B |
| Embedding model | Qwen/Qwen3-Embedding-0.6B | 1024-dim, last-token pooling |
| Embedding instruction | "Given a topic query, retrieve relevant news articles that discuss the topic" | Domain-specific for Robust04 |
| Encode batch size | 256 | Fits in 46GB with 0.6B model |
| Doc max length | 512 | Standard |
| Query max length | 512 | Standard |
| BM25 k1/b | 0.9/0.4 | Same as baseline |
| Bo1 fb_docs/fb_terms | 5/30 | Same as baseline |
| Fusion alphas | 0.3, 0.5 | Dense weight; sweep both |
| Dense top-K | 1000 | Standard |

## Runs

### Run 1: `bm25-bo1-expanded`
- BM25+Bo1 on expanded corpus
- Tests whether synthetic queries improve sparse retrieval alone
- Expected: MAP@100 improvement over baseline BM25+Bo1 (0.2504)

### Run 2: `dense-expanded`
- Qwen3-Embedding-0.6B on expanded corpus
- Tests whether richer document text improves dense embeddings
- Expected: MAP@100 improvement over baseline dense-only (0.2105)

### Run 3: `fusion-a03-expanded`
- Linear fusion (alpha=0.3) of expanded BM25+Bo1 and expanded dense
- Expected: MAP@100 > 0.2827 (current 0.6B best with reranker)

### Run 4: `fusion-a05-expanded`
- Linear fusion (alpha=0.5) of expanded runs
- Sweep fusion weight

## Expected Outcome

- **BM25+Bo1 expanded**: MAP@100 ~0.26-0.28 (up from 0.2504)
- **Dense expanded**: MAP@100 ~0.22-0.24 (up from 0.2105)
- **Fusion expanded**: MAP@100 ~0.29-0.31, recall@100 ~0.52-0.55 (up from 0.49)
- Key metric: recall@100 -- if expansion raises it above 0.52, creates headroom for future reranking

## Baseline Comparison

| System | MAP@100 | recall@100 |
|--------|---------|------------|
| BM25+Bo1 (exp01) | 0.2504 | 0.4527 |
| 0.6B dense (exp05) | 0.2105 | 0.3912 |
| 0.6B fusion (exp05) | 0.2762 | 0.4920 |
| 0.6B fusion + reranker (exp05) | 0.2827 | 0.4961 |
| 8B fusion (exp07, deprecated) | 0.2929 | 0.5103 |

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries -- LLM generates queries purely from document text
- [x] Training does NOT use Robust04 qrels -- no relevance judgments used during generation
- [x] Hard negative mining uses MS-MARCO or documented train split only -- N/A (no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights -- Qwen2.5-3B is used as-is, zero-shot
