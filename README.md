# autoresearch — retrieval

![progress](./progress.png)

Autonomous retrieval research on Robust04 (TREC 2004). An AI agent iterates on retrieval pipelines — training bi-encoders, fusing dense + sparse retrieval, reranking, and mining hard negatives — to maximize **MAP@100** on 249 test queries over a 528K document corpus.

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
