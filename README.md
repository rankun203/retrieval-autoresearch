# autoresearch — dense retrieval

![progress](progress.png)

Autonomous dense retrieval research on Robust04. An AI agent fine-tunes a bi-encoder on MS-MARCO, indexes Robust04, retrieves top-1000 per query, and evaluates MAP@100. It then iterates: modify `train.py`, commit, run, keep or discard, repeat.

## How it works

Three files:

- **`prepare.py`** — fixed utilities: Robust04 download, data loading, MS-MARCO training stream, evaluation harness. **Do not modify.**
- **`train.py`** — the file the agent edits. Bi-encoder model, training loop, FAISS indexing, retrieval, evaluation. Everything is fair game.
- **`program.md`** — instructions for the agent.

The training script runs for a **fixed 10-minute wall-clock budget**, then encodes the full Robust04 corpus (~500K docs) and evaluates. The primary metric is **nDCG@10** — higher is better.

## Quick start

**Requirements:** NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Download Robust04 and verify MS-MARCO stream (one-time, ~600MB)
uv run prepare.py

# Run a single experiment (~12-15 min)
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
program.md      — agent instructions
pyproject.toml  — dependencies
```

## License

MIT
