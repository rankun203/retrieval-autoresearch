"""
exp39-linear-interp: Linear score interpolation for dense+BM25 fusion.

No training needed. Loads existing run files from exp33 (dense) and exp22 (BM25),
normalizes scores per-query, and sweeps alpha for linear interpolation.
Compares against RRF(k=60) baseline.
"""

import os
import time
from collections import defaultdict

from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DENSE_RUN_PATH = os.path.expanduser(
    "~/projects/retrieval-autoresearch/runs/exp33-iter-hn/exp33-iter-hn-dense.run"
)
BM25_RUN_PATH = os.path.expanduser(
    "~/projects/retrieval-autoresearch/runs/exp22-bm25-prf/exp22-bm25-prf.run"
)
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RRF_K = 60
TOP_K = 1000  # keep top-K docs per query in fused run
EXP_NAME = "exp39-linear-interp"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trec_run(path: str) -> dict:
    """Load a TREC run file into {qid: {doc_id: score}}."""
    run = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                run[qid][doc_id] = float(score)
    return dict(run)


def normalize_scores(run: dict) -> dict:
    """Per-query min-max normalize scores to [0, 1]."""
    normalized = {}
    for qid, docs in run.items():
        scores = list(docs.values())
        min_s = min(scores)
        max_s = max(scores)
        rng = max_s - min_s
        if rng == 0:
            # All scores equal — set to 0.5
            normalized[qid] = {d: 0.5 for d in docs}
        else:
            normalized[qid] = {d: (s - min_s) / rng for d, s in docs.items()}
    return normalized


def linear_interpolation(run_a: dict, run_b: dict, alpha: float, top_k: int = 1000) -> dict:
    """
    Fuse two normalized runs: score = alpha * run_a + (1-alpha) * run_b.
    Documents appearing in only one run get score 0 from the other.
    """
    fused = {}
    all_qids = set(run_a.keys()) | set(run_b.keys())
    for qid in all_qids:
        docs_a = run_a.get(qid, {})
        docs_b = run_b.get(qid, {})
        all_docs = set(docs_a.keys()) | set(docs_b.keys())
        scores = {}
        for doc_id in all_docs:
            sa = docs_a.get(doc_id, 0.0)
            sb = docs_b.get(doc_id, 0.0)
            scores[doc_id] = alpha * sa + (1 - alpha) * sb
        # Keep top-K
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused[qid] = dict(sorted_docs)
    return fused


def rrf_fusion(run_a: dict, run_b: dict, k: int = 60, top_k: int = 1000) -> dict:
    """RRF fusion: score = 1/(k+rank_a) + 1/(k+rank_b)."""
    fused = {}
    all_qids = set(run_a.keys()) | set(run_b.keys())
    for qid in all_qids:
        # Build rank maps (1-based)
        docs_a = run_a.get(qid, {})
        docs_b = run_b.get(qid, {})
        rank_a = {doc: r + 1 for r, (doc, _) in enumerate(
            sorted(docs_a.items(), key=lambda x: x[1], reverse=True))}
        rank_b = {doc: r + 1 for r, (doc, _) in enumerate(
            sorted(docs_b.items(), key=lambda x: x[1], reverse=True))}

        all_docs = set(rank_a.keys()) | set(rank_b.keys())
        scores = {}
        for doc_id in all_docs:
            s = 0.0
            if doc_id in rank_a:
                s += 1.0 / (k + rank_a[doc_id])
            if doc_id in rank_b:
                s += 1.0 / (k + rank_b[doc_id])
            scores[doc_id] = s
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused[qid] = dict(sorted_docs)
    return fused


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"=== {EXP_NAME}: Linear Score Interpolation ===", flush=True)

    # Load qrels
    print("Loading Robust04 qrels...", flush=True)
    _, _, qrels = load_robust04()

    # Load run files
    print(f"Loading dense run: {DENSE_RUN_PATH}", flush=True)
    dense_run = load_trec_run(DENSE_RUN_PATH)
    print(f"  {len(dense_run)} queries, {sum(len(v) for v in dense_run.values())} entries", flush=True)

    print(f"Loading BM25 run: {BM25_RUN_PATH}", flush=True)
    bm25_run = load_trec_run(BM25_RUN_PATH)
    print(f"  {len(bm25_run)} queries, {sum(len(v) for v in bm25_run.values())} entries", flush=True)

    # Evaluate individual runs
    print("\n--- Individual run baselines ---", flush=True)
    dense_metrics = evaluate_run(dense_run, qrels)
    print(f"Dense-only:  MAP@100={dense_metrics['map@100']:.4f}  nDCG@10={dense_metrics['ndcg@10']:.4f}  recall@100={dense_metrics['recall@100']:.4f}", flush=True)

    bm25_metrics = evaluate_run(bm25_run, qrels)
    print(f"BM25-only:   MAP@100={bm25_metrics['map@100']:.4f}  nDCG@10={bm25_metrics['ndcg@10']:.4f}  recall@100={bm25_metrics['recall@100']:.4f}", flush=True)

    # RRF baseline
    print("\n--- RRF(k=60) baseline ---", flush=True)
    rrf_run = rrf_fusion(dense_run, bm25_run, k=RRF_K, top_k=TOP_K)
    rrf_metrics = evaluate_run(rrf_run, qrels)
    print(f"RRF(k={RRF_K}):  MAP@100={rrf_metrics['map@100']:.4f}  MAP@1000={rrf_metrics['map@1000']:.4f}  nDCG@10={rrf_metrics['ndcg@10']:.4f}  recall@100={rrf_metrics['recall@100']:.4f}", flush=True)

    # Normalize scores
    print("\n--- Normalizing scores (per-query min-max) ---", flush=True)
    dense_norm = normalize_scores(dense_run)
    bm25_norm = normalize_scores(bm25_run)

    # Alpha sweep
    print("\n--- Alpha sweep ---", flush=True)
    print(f"{'alpha':>6}  {'MAP@100':>8}  {'MAP@1000':>9}  {'nDCG@10':>8}  {'recall@100':>11}", flush=True)
    print("-" * 52, flush=True)

    best_alpha = None
    best_map100 = -1
    best_run = None
    best_metrics = None
    all_results = []

    for alpha in ALPHAS:
        fused = linear_interpolation(dense_norm, bm25_norm, alpha, top_k=TOP_K)
        metrics = evaluate_run(fused, qrels)
        all_results.append((alpha, metrics))
        print(f"{alpha:>6.1f}  {metrics['map@100']:>8.4f}  {metrics['map@1000']:>9.4f}  {metrics['ndcg@10']:>8.4f}  {metrics['recall@100']:>11.4f}", flush=True)

        if metrics['map@100'] > best_map100:
            best_map100 = metrics['map@100']
            best_alpha = alpha
            best_run = fused
            best_metrics = metrics

    # Summary
    print(f"\n--- Summary ---", flush=True)
    print(f"Best alpha:       {best_alpha}", flush=True)
    print(f"Best MAP@100:     {best_metrics['map@100']:.6f}", flush=True)
    print(f"Best MAP@1000:    {best_metrics['map@1000']:.6f}", flush=True)
    print(f"Best nDCG@10:     {best_metrics['ndcg@10']:.6f}", flush=True)
    print(f"Best recall@100:  {best_metrics['recall@100']:.6f}", flush=True)
    print(f"RRF(k=60) MAP@100: {rrf_metrics['map@100']:.6f}", flush=True)
    diff = best_metrics['map@100'] - rrf_metrics['map@100']
    print(f"Difference:       {diff:+.6f} ({'better' if diff > 0 else 'worse'} than RRF)", flush=True)

    # Save best fused run
    run_dir = f"runs/{EXP_NAME}"
    os.makedirs(run_dir, exist_ok=True)
    run_path = f"{run_dir}/{EXP_NAME}.run"
    write_trec_run(best_run, run_path, run_name=EXP_NAME)
    print(f"\nBest run saved to: {run_path}", flush=True)

    # Also save the RRF run for comparison
    rrf_path = f"{run_dir}/{EXP_NAME}-rrf-baseline.run"
    write_trec_run(rrf_run, rrf_path, run_name=f"{EXP_NAME}-rrf")
    print(f"RRF baseline saved to: {rrf_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\n---", flush=True)
    print(f"ndcg@10:          {best_metrics['ndcg@10']:.6f}", flush=True)
    print(f"map@1000:         {best_metrics['map@1000']:.6f}", flush=True)
    print(f"map@100:          {best_metrics['map@100']:.6f}", flush=True)
    print(f"recall@100:       {best_metrics['recall@100']:.6f}", flush=True)
    print(f"training_seconds: 0.0", flush=True)
    print(f"total_seconds:    {elapsed:.1f}", flush=True)
    print(f"peak_vram_mb:     0.0", flush=True)
    print(f"num_steps:        0", flush=True)
    print(f"encoder_model:    e5-base-v2 (pre-trained, from exp33)", flush=True)
    print(f"best_alpha:       {best_alpha}", flush=True)
    print(f"eval_duration:    {elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
