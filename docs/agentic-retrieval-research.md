# Agentic Retrieval Research

## Concept

Multi-round iterative retrieval where an LLM agent:
1. Runs first-round retrieval (BM25+dense)
2. Examines top-10 docs, scores relevance, adds best to final list
3. Generates new queries exploring unexplored aspects based on results so far
4. Repeats until ~100 high-quality docs collected (min 10 rounds)
5. Final ranked list evaluated as retrieval output

## Most Related Papers

### PRISM (2025) — closest match
- Three-agent loop: Question Analyzer → Selector (precision) → Adder (recall)
- Iterates Selector/Adder cycle to build comprehensive evidence sets
- HotpotQA, 2WikiMultiHopQA, MuSiQue benchmarks
- [arXiv 2510.14278](https://arxiv.org/abs/2510.14278)

### SmartSearch (2026) — query refinement with rewards
- Process rewards for intermediate search query quality
- Dual-level credit: rule-based novelty + model-based usefulness
- Iteratively refines queries across search rounds
- [arXiv 2601.04888](https://arxiv.org/abs/2601.04888)

### IRCoT (ACL 2023) — interleaved retrieval + reasoning
- Alternates: generate reasoning step → use as retrieval query → fetch more docs
- Up to +21 points retrieval improvement on multi-hop QA
- [arXiv 2212.10509](https://arxiv.org/abs/2212.10509)

### Self-RAG (ICLR 2024) — adaptive retrieval with reflection
- Reflection tokens to evaluate own outputs and decide when to retrieve more
- On-demand retrieval rather than fixed schedule
- [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)

### FLARE (EMNLP 2023) — confidence-triggered retrieval
- Checks confidence during generation, retrieves when uncertain
- [arXiv 2305.06983](https://arxiv.org/abs/2305.06983)

### CRAG (ICLR 2024) — corrective retrieval
- Lightweight evaluator assesses document quality
- Triggers web search fallback if initial retrieval is low quality
- [arXiv 2401.15884](https://arxiv.org/abs/2401.15884)

## Our Approach vs Literature

**Similar to**: PRISM (multi-agent loop), SmartSearch (query refinement), IRCoT (interleaving)

**Unique aspects**:
- Document-centric: explicitly score and accumulate "good enough" docs each round
- Aspect exploration: generate queries covering different facets of the topic
- Clear stopping criterion (~100 docs) vs max-steps
- Final output is a ranked document list (standard IR eval), not QA answers

## Implementation Plan

Using Qwen3.5-9B as the agent LLM:

```
for each query:
    final_docs = {}  # doc_id -> relevance_score
    explored_queries = [original_query]

    for round in range(max_rounds):
        # Retrieve with current query set
        results = retrieve(explored_queries[-1], top_k=10)

        # LLM evaluates each doc
        for doc in results:
            if doc not in final_docs:
                score = llm_score_relevance(query, doc)
                if score > threshold:
                    final_docs[doc] = score

        # Check stopping criterion
        if len(final_docs) >= 100 and round >= 10:
            break

        # LLM generates new query exploring uncovered aspects
        new_query = llm_generate_query(
            original_query,
            explored_queries,
            final_docs
        )
        explored_queries.append(new_query)

    # Rank final_docs by score → TREC run format
```

Track `eval_dur` from queries in to all retrievals done (this will be the heaviest pipeline).
