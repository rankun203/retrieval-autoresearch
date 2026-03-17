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

**The goal: maximize MAP@100 on Robust04 test queries (249 topics, excluding qid 672).**

## Output format

The script always prints this summary at the end:
```
---
ndcg@10:          0.XXXXXX
map@100:          0.XXXXXX
recall@100:      0.XXXXXX
training_seconds: 600.1
total_seconds:    NNN.N
peak_vram_mb:     XXXXX.X
num_steps:        NNNN
encoder_model:    MODEL_NAME
batch_size:       NNN
max_doc_len:      NNN
max_query_len:    NNN
lr:               X.Xe-XX
temperature:      X.XX
num_docs_indexed: NNNNNN
```

Extract the key metrics:
```bash
grep "^ndcg@10:\|^map@100:\|^recall@100:\|^peak_vram_mb:\|^loss_curve:\|^budget_assessment:" run.log
```

The summary also includes:
- `loss_curve`: smoothed loss at 0%, 10%, ..., 100% of training time
- `budget_assessment`: one of `OK`, `UNDERTRAINED`, or `OVERFIT/PLATEAU`

**Act on the budget assessment**:
- `UNDERTRAINED`: next experiment should try `TIME_BUDGET = 900` (15 min). If consistently undertrained, raise further.
- `OVERFIT/PLATEAU`: next experiment can try `TIME_BUDGET = 300` (5 min) — wasting time on flat loss.
- `OK`: keep `TIME_BUDGET = 600`.
Only change TIME_BUDGET if you see the same signal 2+ experiments in a row.

## Logging results

After each completed run:
1. **Save the log**: `mkdir -p logs && cp run.log logs/$(git rev-parse --short HEAD).log`
2. **Append to the root results.tsv** at the project root (NOT in the worktree). Use absolute path or `../../results.tsv` from worktree. Tab-separated, 12 columns:

```
commit	ndcg@10	map@1000	map@100	recall@100	memory_gb	eval_dur	status	encoder	batch	doc_len	lr	description	worktree
```
- commit: 7-char short hash
- ndcg@10, map@100, recall@100: from summary (use 0.000000 for crashes)
- memory_gb: peak_vram_mb / 1024, rounded to .1f (use 0.0 for crashes)
- eval_dur: eval_duration from summary (seconds with 3 decimal places, e.g. 834.123; use N/A for crashes)
- status: `keep`, `discard`, or `crash`
- encoder: short model name (e.g. `e5-base-v2`)
- batch: BATCH_SIZE value
- doc_len: MAX_DOC_LEN value
- lr: LR value (e.g. `1e-5`)
- description: short text of what this experiment tried

Do NOT commit results.tsv or logs/ (leave them untracked). results.tsv lives at the project root, shared across all worktrees.

## The experiment loop

LOOP FOREVER:

1. Look at git state (current branch/commit).
2. Plan an experimental change to `train.py` (or new helper files). Consult `docs/plan.md` for the prioritized experiment list and `docs/ir-survey-202603.md` for paper-backed ideas. Pick the highest-priority unchecked item from the plan.
3. **Commit changes BEFORE running** — every experiment must have a commit hash:
   ```bash
   git add -A && git commit -m "exp-name: description of what this tries"
   ```
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^ndcg@10:\|^map@1000:\|^map@100:\|^recall@100:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Run `tail -n 50 run.log` for stack trace. Fix if trivial, else skip.
7. **Sanity-check results**: If a result is far off from what you'd expect (e.g. near-zero for a known-good model, or much worse than baseline), do NOT just log and move on. Instead:
   - Add a note in the results.tsv description like `SUSPICIOUS: expected ~0.4 got 0.01`
   - Research online (WebSearch) for the correct way to use that model/method — check the model card, HuggingFace docs, or GitHub issues
   - Fix and re-run before moving on
8. **Log to results.tsv** (at project root, using the commit hash from step 3).
8. **If improved** → keep the commit. Cherry-pick onto master:
   ```bash
   git -C /path/to/repo checkout master && git cherry-pick <commit> && git checkout -
   ```
9. **If not improved** → reset to discard changes (commit becomes orphan but is still reachable by hash):
   ```bash
   git reset --hard HEAD~1
   ```
   The results.tsv row still has the commit hash so you can always `git show <hash>` to see what was tried.

**Timeout**: Different pipelines have different runtimes:
- Bi-encoder only: ~15 min (train + encode + eval)
- Bi-encoder + cross-encoder rerank: ~30-50 min
- BM25 + LLM reranker (0.6B+): ~1-3 hours
- General limit: **12 hours max** unless the method is high-priority and expected to yield big improvements. Kill and treat as failure if exceeded.

**NEVER STOP**: Once the loop starts, do not pause for human approval. Run until manually interrupted.

**Simplicity criterion**: A 0.001 improvement that adds 50 lines of complex code is not worth it. A simplification that matches performance? Always keep.
