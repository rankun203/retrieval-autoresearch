# Dense Retrieval Autoresearch

This is an autonomous research platform for improving dense retrieval methods.
The evaluation target is **Robust04** (TREC Robust 2004, BEIR version).
The primary metric is **nDCG@10** — higher is better.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). Branch `autoresearch/<tag>` must not exist.
2. **Create a worktree**: from the repo root, run:
   ```bash
   git worktree add ./worktrees/<tag> -b autoresearch/<tag>
   ```
   All experiment work happens inside `./worktrees/<tag>/` — never touch the main working directory.
3. **Read the in-scope files** (from inside the worktree): Read these for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed utilities: data loading, evaluation, MS-MARCO training stream. Do NOT modify.
   - `train.py` — the baseline you iterate on.
4. **Verify data exists**: Check that `~/.cache/autoresearch-retrieval/robust04/` exists. If not, tell the human to run `uv run prepare.py` from the repo root.
5. **Initialize results.tsv**: Create `./worktrees/<tag>/results.tsv` with header only.
6. **All commands run from inside the worktree directory** (`./worktrees/<tag>/`).
7. **Confirm and go**.

## Experimentation

Training runs for a **fixed 10-minute (600s) wall-clock budget** (tracked in `total_training_time`).
After training, the model encodes the full Robust04 corpus, builds a FAISS index, retrieves top-1000 per query, and evaluates nDCG@10.

**What you CAN do:**
- Modify `train.py` or any other experiment files. Everything is fair game:
  - Encoder architecture (model name, pooling strategy, projection head)
  - Training objective (InfoNCE, triplet, listwise, distillation from BM25)
  - Hard negative mining (BM25, cross-encoder scores, in-batch)
  - Indexing strategy (flat, IVF, HNSW, product quantization)
  - Query/document representation (late interaction, sparse augmentation)
  - Hybrid approaches (dense + BM25 interpolation)
- Add new `.py` files if the experiment has multiple stages (e.g. `model.py`, `index.py`).
  Just make sure there's a single entry point: `uv run train.py`.

**What you CANNOT do:**
- Modify `prepare.py`. The `evaluate_run()` function is the ground truth metric.
- Install new packages or add dependencies.

**The goal: maximize nDCG@10 on Robust04 test queries.**

## Output format

The script always prints this summary at the end:
```
---
ndcg@10:          0.XXXXXX
map@100:          0.XXXXXX
recall@1000:      0.XXXXXX
training_seconds: 600.1
total_seconds:    NNN.N
peak_vram_mb:     XXXXX.X
num_steps:        NNNN
encoder_model:    MODEL_NAME
num_docs_indexed: NNNNNN
```

Extract the key metrics:
```bash
grep "^ndcg@10:\|^peak_vram_mb:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):
```
commit	ndcg@10	memory_gb	status	description
```
- commit: 7-char short hash
- ndcg@10: e.g. 0.421234 (use 0.000000 for crashes)
- memory_gb: peak_vram_mb / 1024, rounded to .1f (use 0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: short text of what this experiment tried

Do NOT commit results.tsv (leave it untracked).

## The experiment loop

LOOP FOREVER:

1. Look at git state (current branch/commit).
2. Plan an experimental change to `train.py` (or new helper files). Ideas:
   - Different pretrained backbone (larger BERT, E5-small, etc.)
   - Add cross-encoder re-ranking stage
   - BM25 + dense hybrid (interpolate scores)
   - Different pooling (CLS, mean, weighted mean)
   - Larger batch size for more in-batch negatives
   - Curriculum: start with BM25 negatives, switch to model-mined hard negatives mid-training
   - SPLADE-style sparse regularization
   - Knowledge distillation: use BM25 scores as soft labels
3. `git commit` your changes.
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^ndcg@10:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Run `tail -n 50 run.log` for stack trace. Fix if trivial, else skip.
7. Log to results.tsv.
8. If nDCG@10 improved → keep commit. If not → `git reset --hard HEAD~1`.

**Timeout**: Each experiment should take ~12-15 minutes total. If it exceeds 25 minutes, kill and treat as failure.

**NEVER STOP**: Once the loop starts, do not pause for human approval. Run until manually interrupted.

**Simplicity criterion**: A 0.001 improvement that adds 50 lines of complex code is not worth it. A simplification that matches performance? Always keep.
