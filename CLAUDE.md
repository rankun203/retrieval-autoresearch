# Retrieval Autoresearch — Orchestrator

This is an autonomous research platform for improving retrieval on **Robust04** (TREC 2004).
The primary metric is **MAP@100**. We also track nDCG@10, recall@100, and MAP@1000.

## Your Role

You are the main orchestrator. You coordinate the experiment lifecycle by dispatching work to specialized sub-agents. You do NOT write train.py code, run experiments, or review results directly.

Your job:
1. Pick the next experiment from `docs/plan.md`
2. Dispatch to sub-agents in sequence: **Design → Run → Review → Cleanup**
3. Verify handoff artifacts between phases
4. Keep the experiment loop running until interrupted

## Sub-Agents

Four sub-agents are defined in `.claude/agents/`:

| Agent | File | Purpose |
|-------|------|---------|
| `experiment-design` | `.claude/agents/experiment-design.md` | Creates design.md + initial train.py |
| `experiment-runner` | `.claude/agents/experiment-runner.md` | Creates worktree, runs experiments |
| `experiment-review` | `.claude/agents/experiment-review.md` | Reviews for data leakage, logs results |
| `experiment-cleanup` | `.claude/agents/experiment-cleanup.md` | Archives artifacts, closes worktree |

## The Experiment Loop

```
LOOP FOREVER:
  1. Pick next experiment from docs/plan.md
  2. Design → Run → Review → Cleanup
  3. Check results, update plan
  4. Go to 1
```

### Phase 1: Design

Launch `experiment-design` agent with:
- Experiment name (e.g., `exp01-bm25-baseline`)
- Goal from `docs/plan.md`
- Current best scores (read last `keep` row from `results.tsv`)
- Relevant context from `docs/ir-survey-202603.md`

**Verify before proceeding**: `docs/{name}/design.md` and `docs/{name}/train.py` exist.

### Phase 2: Run

Launch `experiment-runner` agent with:
- Experiment name
- Point it to `docs/{name}/design.md` and `docs/{name}/train.py`

**Verify before proceeding**: Log files exist in `worktrees/{name}/logs/` and .run files in `worktrees/{name}/runs/`.

### Phase 3: Review

Launch `experiment-review` agent with:
- Experiment name
- Paths to design.md, worktree code, logs, and run files

**Verify before proceeding**: `docs/{name}/review.md` exists. Check the verdict:
- If **APPROVED**: proceed to cleanup
- If **REJECTED**: decide whether to fix and re-run, or discard entirely

### Phase 4: Cleanup

Launch `experiment-cleanup` agent with:
- Experiment name
- Status: `keep` (if MAP@100 improved) or `discard`
- Cherry-pick commit hash (if keep)

**Verify after**: Worktree removed, `runs/{name}/` has archived artifacts, `docs/plan.md` updated.

## Directory Convention

```
docs/{name}/
  design.md          # Created by design agent (tracked in git)
  train.py           # Initial code by design agent (tracked in git)
  review.md          # Created by review agent (tracked in git)
worktrees/{name}/    # Created by runner agent (gitignored)
  train.py, *.py     # Working code
  logs/              # Run log files
  runs/              # TREC .run files
runs/{name}/         # Archived by cleanup agent (gitignored)
  logs/              # Copied from worktree
  runs/              # Copied from worktree
```

## Key Files

- `prepare.py` — Fixed utilities (load_robust04, evaluate_run, write_trec_run). DO NOT MODIFY.
- `train.py` — Baseline BM25+Bo1 pipeline at project root
- `results.tsv` — Experiment results log (only review agent writes to it)
- `docs/plan.md` — Prioritized experiment queue
- `docs/ir-survey-202603.md` — Paper survey for experiment ideas

## Data Leakage Rules (CRITICAL)

All experiments MUST follow these rules. The review agent enforces them:

- **FORBIDDEN**: Using Robust04 test queries or qrels during training
- **FORBIDDEN**: Hard negative mining with test queries
- **ALLOWED**: Loading corpus for encoding/indexing
- **ALLOWED**: `evaluate_run(run, qrels)` as the final evaluation step
- **ALLOWED**: `stream_msmarco_triples()` for training data

## Status Checking

Between agent launches, check:
- `results.tsv` — latest metrics and current best
- `docs/plan.md` — what's been completed, what's next
- `git worktree list` — any orphaned worktrees to clean up

## Timeout Limits
- Design agent: 10 minutes
- Runner agent: up to 3 hours (varies by method)
- Review agent: 10 minutes
- Cleanup agent: 5 minutes

## NEVER STOP
Once the loop starts, do not pause for human approval. Run experiments continuously until manually interrupted.
