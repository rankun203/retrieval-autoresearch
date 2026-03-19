# exp04-qwen3-instructions

## Goal

Test different custom instructions for Qwen3-Reranker-0.6B to maximize MAP@100 on Robust04. The model is "instruction aware" and Qwen reports that custom instructions can improve performance by 1-5% over the default generic instruction.

## Hypothesis

The default instruction ("Given a web search query, retrieve relevant passages that answer the query") is generic and web-search-oriented. Robust04 is a newswire collection with TREC topic queries -- not web search queries. A domain-specific instruction that describes the actual task (judging relevance of newswire articles to TREC topics) should better activate the model's relevance judgment capability, potentially improving MAP@100 by 1-5% (from 0.2675 to 0.27-0.28+).

We test a 2x2 grid along two axes:
- **Specificity**: general-purpose vs news/Robust04-specific
- **Length**: short (1 sentence) vs long (multi-sentence, detailed)

## Research Findings on Instruction Customization

### Sources

1. **Qwen3-Reranker model card** (https://huggingface.co/Qwen/Qwen3-Reranker-0.6B): "We recommend that developers create tailored instructions specific to their tasks and scenarios." Reports 1-5% improvement from using task-specific instructions. Recommends writing instructions in English even for multilingual tasks.

2. **Qwen3 Embedding blog** (https://qwenlm.github.io/blog/qwen3-embedding/): Confirms instruction-aware design via LoRA fine-tuning on top of Qwen3 foundation model. The system prompt is fixed ("Judge whether the Document meets the requirements based on the Query and the Instruct provided"), and the customizable part is the `<Instruct>:` field.

3. **Qwen3 Embedding paper** (arXiv:2506.05176, Zhang et al., 2025): Describes training methodology for instruction-aware embedding and reranking. Models trained with diverse task instructions to generalize across scenarios.

4. **E5-Mistral-7B-Instruct** (https://huggingface.co/intfloat/e5-mistral-7b-instruct, Wang et al., 2024): Pioneered the "Instruct: {task}\nQuery: {query}" format. Key insight: "The task definition should be a one-sentence instruction that describes the task." Examples include "Given a claim, find documents that refute the claim" and "Given a web search query, retrieve relevant passages that answer the query."

5. **FollowIR benchmark** (Weller et al., 2024): Measures how well retrieval models follow nuanced instructions. Qwen3-Reranker-0.6B scores 5.41 on FollowIR, confirming it genuinely responds to instruction content rather than ignoring it.

### Key Takeaways for Instruction Design

- Instructions should describe the **task** not the **model behavior** (i.e., what to find, not how to score)
- Specificity helps: describing the document type, query format, and relevance criteria improves results
- Instructions should be in English (training data was primarily English instructions)
- Length matters less than content quality, but overly long instructions may distract
- The instruction interacts with the system prompt "Judge whether the Document meets the requirements" -- so the instruction effectively defines what "requirements" means

## Method

1. Run BM25+Bo1 first-stage retrieval (identical to exp01/exp03b) to get top-1000 candidates.
2. Load Qwen3-Reranker-0.6B with the CORRECT prompt format (from exp03b).
3. For each of 5 instructions (4 experimental + 1 default baseline), rerank top-100 candidates and evaluate.
4. Compare all 5 runs to find the best instruction.

The Qwen3-Reranker implementation is copied exactly from exp03b (prefix/suffix tokens, process_inputs, compute_scores). Only the INSTRUCTION variable changes between runs.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| First-stage retriever | BM25+Bo1 (k1=0.9, b=0.4, fb_docs=5, fb_terms=30) |
| First-stage top-K | 1000 |
| Reranker model | Qwen/Qwen3-Reranker-0.6B |
| Rerank depth | 100 (top-100 only, based on exp03b results) |
| Max input length | 4096 tokens |
| Scoring batch size | 8 |
| Scoring method | log_softmax over [no, yes] token logits at last position |
| dtype | float16 |
| Training | None (zero-shot, instruction variation only) |
| GPU | L40S 46GB |

## Runs

### Run 1: `default` (baseline)
- **Description**: Default instruction from exp03b, serves as control
- **Instruction**: "Given a web search query, retrieve relevant passages that answer the query"
- **Expected output**: `runs/default.run`

### Run 2: `general-short`
- **Description**: General-purpose, concise instruction
- **Instruction**: "Determine if the document is relevant to the query"
- **Rationale**: Simpler, more direct task description without web-search framing
- **Expected output**: `runs/general-short.run`

### Run 3: `general-long`
- **Description**: General-purpose, detailed instruction
- **Instruction**: "Given a keyword query, determine whether the document contains information that is relevant to the topic described by the query. A relevant document should directly address the query topic with substantive information, not merely mention related terms."
- **Rationale**: Provides detailed relevance criteria while staying domain-agnostic
- **Expected output**: `runs/general-long.run`

### Run 4: `news-short`
- **Description**: News-domain-specific, concise instruction
- **Instruction**: "Given a topic query, retrieve relevant news articles that discuss the topic"
- **Rationale**: Frames the task as news retrieval matching Robust04's newswire corpus
- **Expected output**: `runs/news-short.run`

### Run 5: `news-long`
- **Description**: News-domain-specific, detailed instruction
- **Instruction**: "Given a short topic query about a news event or subject, determine whether the news article is relevant. A relevant article should contain substantial information about the query topic, including factual reporting, analysis, or background context. Articles that merely mention the topic in passing are not relevant."
- **Rationale**: Combines domain specificity (news articles) with detailed relevance criteria matching TREC judgment guidelines
- **Expected output**: `runs/news-long.run`

## Expected Outcome

| Run | Expected MAP@100 | Rationale |
|-----|-------------------|-----------|
| default | ~0.2675 | Matches exp03b result (control) |
| general-short | ~0.265-0.270 | Simpler may not help much |
| general-long | ~0.270-0.280 | Detailed criteria should help |
| news-short | ~0.270-0.280 | Domain match should help |
| news-long | ~0.275-0.285 | Best of both: domain + detail |

Based on Qwen's reported 1-5% improvement range, we expect the best instruction to reach MAP@100 of 0.275-0.285, a relative improvement of 3-6% over the default.

## Baseline Comparison

- **exp03b (default instruction)**: MAP@100=0.2675, nDCG@10=0.5301
- **exp01-bm25-baseline**: MAP@100=0.2504 (BM25+Bo1 PRF, current best logged)

## Data Leakage Checklist

- [x] Training does NOT use Robust04 test queries (no training -- zero-shot, instruction only)
- [x] Training does NOT use Robust04 qrels (no training -- zero-shot)
- [x] Hard negative mining uses MS-MARCO or documented train split only (N/A -- no training)
- [x] `evaluate_run()` called only for final evaluation, not during training
- [x] No test-time information flows into model weights (pre-trained model used as-is)
