# Dense Retrieval Autoresearch

This is an autonomous research platform for improving dense retrieval methods.
The evaluation target is **Robust04** (TREC Robust 2004, BEIR version).
The primary metric is **MAP@100** — higher is better. We also track nDCG@10 and recall@100.

## Setup

1. **Name the experiment**: each experiment gets a unique name (e.g. `exp20-bm25-tuning`, `exp21-hybrid-rerank`). Branch `autoresearch/<name>` must not exist.
2. **Create a worktree**: from the repo root, run:
   ```bash
   git worktree add ./worktrees/<name> -b autoresearch/<name>
   ```
   Each experiment gets its own worktree. All work happens inside `./worktrees/<name>/` — never touch the main working directory.
3. **Read the in-scope files** (from inside the worktree): Read these for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed utilities: data loading, evaluation, MS-MARCO training stream. Do NOT modify.
   - `train.py` — the baseline you iterate on.
4. **Verify data exists**: Check that `~/.cache/autoresearch-retrieval/robust04/` exists. If not, tell the human to run `uv run prepare.py` from the repo root.
5. **All commands run from inside the worktree directory** (`./worktrees/<name>/`).
6. **Confirm and go**.

## Experimentation

Training runs for a **fixed 10-minute (600s) wall-clock budget** (tracked in `total_training_time`).
After training, the model encodes the full Robust04 corpus, builds a FAISS index, retrieves top-1000 per query, and evaluates.

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

**The goal: maximize MAP@100 on Robust04 test queries (249 topics, excluding qid 672).**

## Experiment round

An experiment round is one full cycle: commit code, run, evaluate, log, decide keep/discard. Everything below happens inside the worktree.

### 1. Commit before running

Every experiment must have a commit hash for traceability:
```bash
git add -A && git commit -m "exp-name: description of what this tries"
```

### 2. Run the experiment

```bash
PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```
Use `PYTHONUNBUFFERED=1` so progress prints flush immediately to `run.log`.
All print statements in the experiment code should use `flush=True` for real-time progress tracking.

### 3. Read results

The script prints a summary block at the end:
```
---
ndcg@10:          0.XXXXXX
map@100:          0.XXXXXX
recall@100:       0.XXXXXX
training_seconds: 600.1
total_seconds:    NNN.N
peak_vram_mb:     XXXXX.X
num_steps:        NNNN
encoder_model:    MODEL_NAME
num_docs_indexed: NNNNNN
eval_duration:    NNN.NNN
```

Extract key metrics:
```bash
grep "^ndcg@10:\|^map@100:\|^recall@100:\|^peak_vram_mb:\|^loss_curve:\|^budget_assessment:" run.log
```

If empty → crashed. Run `tail -n 50 run.log` for the stack trace. Fix if trivial, else skip.

The summary also includes:
- `loss_curve`: smoothed loss at 0%, 10%, ..., 100% of training time
- `budget_assessment`: one of `OK`, `UNDERTRAINED`, or `OVERFIT/PLATEAU`

**Act on the budget assessment**:
- `UNDERTRAINED`: next experiment should try `TIME_BUDGET = 900` (15 min). If consistently undertrained, raise further.
- `OVERFIT/PLATEAU`: next experiment can try `TIME_BUDGET = 300` (5 min) — wasting time on flat loss.
- `OK`: keep `TIME_BUDGET = 600`.
Only change TIME_BUDGET if you see the same signal 2+ experiments in a row.

### 4. Sanity-check results

If a result is far off from what you'd expect (e.g. near-zero for a known-good model, or much worse than baseline), do NOT just log and move on. Instead:
- Add a note in the results.tsv description like `SUSPICIOUS: expected ~0.4 got 0.01`
- Research online (WebSearch) for the correct way to use that model/method — check the model card, HuggingFace docs, or GitHub issues
- Fix and re-run before moving on

### 5. Log results

Every run (keep, discard, crash) gets logged:

**a) Save run artifacts** inside the worktree (train.py should do this automatically):
- TREC run file: `runs/<worktree_name>/<worktree_name>.run`
  - Format: `qid Q0 docno rank score run_name` (standard trec_eval compatible)
  - Use `write_trec_run()` from `prepare.py` to generate this file after retrieval/reranking, before evaluation
- Run log: `cp run.log runs/<worktree_name>/run.log`

**b) Append to the root results.tsv** at the project root (NOT in the worktree). Use absolute path or `../../results.tsv` from worktree. Tab-separated columns:

```
commit	ndcg@10	map@1000	map@100	recall@100	memory_gb	eval_dur	status	encoder	batch	doc_len	lr	description	worktree
```
- commit: 7-char short hash
- ndcg@10, map@100, recall@100: from summary (use 0.000000 for crashes)
- Use `-1` for any metric that is not a true value (e.g. map@1000 when only top-100 docs were retrieved/reranked — the number would equal map@100 but is not a real map@1000)
- memory_gb: peak_vram_mb / 1024, rounded to .1f (use 0.0 for crashes)
- eval_dur: eval_duration from summary (seconds with 3 decimal places, e.g. 834.123; use N/A for crashes)
- status: `keep`, `discard`, or `crash`
- encoder: short model name (e.g. `e5-base-v2`)
- batch: BATCH_SIZE value
- doc_len: MAX_DOC_LEN value
- lr: LR value (e.g. `1e-5`)
- description: short text of what this experiment tried

### 6. Keep or discard

**If improved** → keep the commit. Cherry-pick onto master:
```bash
git -C /path/to/repo checkout master && git cherry-pick <commit> && git checkout -
```

**If not improved** → reset to discard changes:
```bash
git reset --hard HEAD~1
```
The commit becomes an orphan and will eventually be garbage-collected — that's fine. The results.tsv row preserves the metrics and description for reference.

### 7. Close the worktree

When all runs in a worktree are done (or the experiment direction is exhausted):

1. **Copy run artifacts** to `runs/<worktree_name>/` at the **repo root** (NOT inside the worktree). Save all TREC run files and logs — if multiple runs exist (e.g. from iterations), save all with descriptive names:
   ```bash
   mkdir -p runs/<worktree_name>
   cp worktrees/<worktree_name>/runs/<worktree_name>/*.run runs/<worktree_name>/
   cp worktrees/<worktree_name>/run.log runs/<worktree_name>/run.log
   ```
2. **Verify results.tsv** — every run (keep, discard, crash) must have a row.
3. **Verify kept commits are on master** — cherry-pick if not already done.
4. **Update `docs/plan.md`** — check off completed items, update current best, add notes on findings.
5. **Commit and push master**:
   ```bash
   git add results.tsv docs/plan.md && git commit -m "Close <worktree>: <summary>"
   git push
   ```
6. **Remove the worktree and branch**:
   ```bash
   git worktree remove --force worktrees/<worktree_name>
   git branch -D autoresearch/<worktree_name>
   ```

The `runs/` directory is gitignored (run files are large). Run files can be re-evaluated anytime with:
```bash
uv run evaluate.py --run runs/<name>/<name>.run --output-dir eval_results/
```

## The experiment loop

LOOP FOREVER:

1. Consult `docs/plan.md` for the prioritized experiment list and `docs/ir-survey-202603.md` for paper-backed ideas. Pick the highest-priority unchecked item.
2. **Launch a new Agent** for the experiment. Each experiment runs in its own agent (subagent) so the main conversation stays responsive. The agent prompt must include:
   - The experiment name and goal (from `docs/plan.md`)
   - A directive to follow this `program.md` end-to-end: setup → experiment rounds → close worktree
   - Any relevant context (current best scores, prior findings, model details from `docs/ir-survey-202603.md`)
3. **Wait for the agent to finish**, then review its result summary.
4. Go to 1.

**Agent launch template** (adapt as needed):
```
Read program.md and follow the full experiment loop for experiment "<name>".
Goal: <what this experiment tries>.
Context: current best MAP@100=X.XXXX (<method>). See docs/plan.md and docs/ir-survey-202603.md for details.
Run the experiment end-to-end: set up worktree, write/modify code, commit, run, log results, keep/discard, close worktree.
```

**Timeout**: Different pipelines have different runtimes:
- Bi-encoder only: ~15 min (train + encode + eval)
- Bi-encoder + cross-encoder rerank: ~30-50 min
- BM25 + LLM reranker (0.6B+): ~1-3 hours
- General limit: **12 hours max** unless the method is high-priority and expected to yield big improvements. Kill and treat as failure if exceeded.

**NEVER STOP**: Once the loop starts, do not pause for human approval. Run until manually interrupted.

**Simplicity criterion**: A 0.001 improvement that adds 50 lines of complex code is not worth it. A simplification that matches performance? Always keep.
