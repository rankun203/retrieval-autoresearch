# Query Variant Generation Research

## Top Strategies for Robust04

### 1. Query2Doc (Few-shot pseudo-document generation)
Best overall for TREC. Generate a hypothetical relevant passage, append to query.
```
Prompt: "Write a passage that answers the given query:
Query: {query}
Passage:"
```
- BM25: +3-15% on MS-MARCO, +4.0 nDCG@10 on TREC DL 2019
- Works best with capable LLMs

### 2. HyDE (Hypothetical Document Embeddings)
Generate hypothetical answer, use its embedding for dense retrieval.
```
Prompt: "Please write a relevant passage to answer this question: {query}"
```
- Matches fine-tuned dense retrievers without training data

### 3. P-2 Format (Title + Description + Narrative)
Best for TREC topics. Use full topic context for variant generation.
```
Prompt: "You are a generator of search query variants.
Generate 10 diverse keyword queries about {title}.
Context: {description}
Your reply is a numbered list of search queries."
```
- Outperforms title-only approaches

### 4. ThinkQE (Iterative Refinement)
Multi-round: generate expansion → retrieve → refine with retrieved docs → repeat.
```
Round 1: "Given query '{query}', write a passage that answers it."
Round 2: "Given query '{query}' and these passages: {top_docs}, write a better answering passage."
```
- Outperforms single-shot on web search tasks

### 5. Query Decomposition
Break complex queries into sub-queries, retrieve separately, fuse with RRF.
```
Prompt: "Break this query into 3 distinct sub-questions exploring different aspects:
Query: {query}"
```

### 6. CSQE (Corpus-Steered)
Ground expansions in actual retrieved documents to avoid hallucinations.
Two-stage: initial retrieval → extract key sentences → expand query with grounded context.

## Combination Strategy
- **RRF fusion** of variant results beats concatenation
- 3-5 variants optimal (more adds redundancy)
- RRF k=60
- Multi-pass: BM25(expanded) + Dense(original) + Dense(expanded) → RRF

## For Robust04 Specifically
- Title + Description weighting improves variant quality
- Keyword expansion most effective for short title queries
- Pseudo-document generation best for description queries
- Corpus-steered expansion avoids topic drift on hard topics
