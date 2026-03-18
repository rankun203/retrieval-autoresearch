# autoresearch — dense retrieval

![progress](./progress.png)

Autonomous retrieval research on Robust04 (TREC 2004). An AI agent iterates on retrieval pipelines — training bi-encoders, fusing dense + sparse retrieval, reranking, and mining hard negatives — to maximize **MAP@100** on 249 test queries over a 528K document corpus.

## Current best

**MAP@100 = 0.3275** (exp30: e5-base-v2 with hard negative mining + BM25+Bo1 RRF fusion)

| Milestone | MAP@100 | Method |
|-----------|---------|--------|
| Baseline | 0.126 | MiniLM-L6-v2 bi-encoder |
| Dense+rerank | 0.222 | e5-base-v2 + MiniLM cross-encoder rerank |
| BM25+PRF | 0.250 | BM25(k1=0.9,b=0.4) + Bo1 query expansion |
| LLM reranker | 0.260 | BM25+Bo1 + Qwen3-Reranker-0.6B |
| Hybrid fusion | 0.268 | Dense + BM25 RRF fusion + Qwen3-Reranker |
| **HN mining + hybrid** | **0.328** | **2-phase hard negative mining + RRF fusion** |

## How it works

- **`prepare.py`** — fixed utilities: Robust04 download, data loading, MS-MARCO training stream, evaluation harness. **Do not modify.**
- **`train.py`** — the file the agent edits. Current pipeline: bi-encoder training with hard negative mining, FAISS indexing, hybrid RRF fusion with BM25, optional LLM reranking.
- **`program.md`** — instructions for the agent: experiment setup, worktree management, result logging, keep/discard decisions.
- **`docs/plan.md`** — prioritized experiment list and current best results.

Each experiment runs in an isolated git worktree. The agent trains for a **fixed 10-minute wall-clock budget**, encodes the full Robust04 corpus (~528K docs), and evaluates. Results are logged to `results.tsv`.

## Quick start

**Requirements:** NVIDIA GPU (46GB+ recommended), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Download Robust04 and verify MS-MARCO stream (one-time, ~600MB)
uv run prepare.py

# Run a single experiment (~30-50 min with full pipeline)
uv run train.py
```

## Running the agent

Point your agent at `program.md`:

```
Have a look at program.md and let's kick off a new experiment.
```

## Project structure

```
prepare.py      — fixed: data download, Robust04 loading, evaluation (do not modify)
train.py        — agent modifies this: model, training, indexing, retrieval
program.md      — agent instructions and experiment loop
docs/plan.md    — prioritized experiment list and results tracker
docs/ir-survey-202603.md — IR paper survey for experiment ideas
results.tsv     — all experiment results (tab-separated)
progress.png    — auto-generated progress chart
analysis.ipynb  — notebook to regenerate progress.png
runs/           — run artifacts (TREC run files, logs) — gitignored
worktrees/      — experiment worktrees — gitignored
```

## License

MIT
