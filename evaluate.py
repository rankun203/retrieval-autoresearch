"""
Evaluate TREC run files on Robust04.

Reads a standard TREC run file and computes:
- MAP@1000 and MAP@100
- nDCG@10
- Recall@100
- RBP (with configurable p)

Usage:
  uv run evaluate.py --run runs/exp17/exp17.run
  uv run evaluate.py --run runs/exp17/exp17.run --output-dir eval_results/

Produces per-query and summary TSV files.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from prepare import load_robust04, ROBUST04_DIR


def parse_trec_run(run_path: str) -> dict:
    """Parse a TREC run file into {qid: {docno: score}}."""
    run = {}
    with open(run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docno, rank, score, name = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            if qid not in run:
                run[qid] = {}
            run[qid][docno] = float(score)
    return run


def evaluate_with_pytrec(run: dict, qrels: dict) -> dict:
    """Evaluate using pytrec_eval. Returns per-query and aggregate metrics."""
    import pytrec_eval

    measures = {"ndcg_cut_10", "map_cut_1000", "map_cut_100", "recall_100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)

    # Aggregate
    agg = {}
    for m_key, m_name in [("ndcg_cut_10", "ndcg@10"), ("map_cut_1000", "map@1000"),
                           ("map_cut_100", "map@100"), ("recall_100", "recall@100")]:
        vals = [v[m_key] for v in per_query.values() if m_key in v]
        agg[m_name] = float(np.mean(vals)) if vals else 0.0

    return per_query, agg


def evaluate_rbp(run_path: str, qrels_path: str, p: float = 0.8, binary: bool = True):
    """Evaluate RBP using trectools (optional, installed via uv run --with)."""
    try:
        from trectools import TrecEval, TrecQrel, TrecRun
    except ImportError:
        return None, None

    qrel = TrecQrel(str(qrels_path))
    evaluator = TrecEval(TrecRun(str(run_path)), qrel)

    rbp_df, residual_df = evaluator.get_rbp(
        p=p, per_query=True, binary_topical_relevance=binary
    )
    rbp_col = f"RBP({p:.2f})@1000"
    rbp_df = rbp_df.reset_index().rename(columns={rbp_col: "rbp"})
    residual_df = residual_df.reset_index().rename(columns={rbp_col: "rbp_residual"})

    merged = rbp_df.merge(residual_df, on="query", how="outer")
    agg_rbp = float(merged["rbp"].mean())
    agg_residual = float(merged["rbp_residual"].mean())

    return merged, {"rbp": agg_rbp, "rbp_residual": agg_residual}


def run_name_from_path(path: str) -> str:
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(description="Evaluate TREC run files on Robust04")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--output-dir", default=None, help="Directory for per-query/summary output TSVs")
    parser.add_argument("--rbp-p", type=float, default=0.8, help="RBP persistence parameter (default: 0.8)")
    parser.add_argument("--binary-relevance", type=bool, default=True, help="Binary topical relevance for RBP")
    args = parser.parse_args()

    run_name = run_name_from_path(args.run)
    print(f"Evaluating: {run_name}")
    print(f"Run file: {args.run}")

    # Load data
    _, queries, qrels = load_robust04()
    run = parse_trec_run(args.run)
    print(f"Queries in run: {len(run)}, Queries with qrels: {len(qrels)}")

    # pytrec_eval metrics
    per_query, agg = evaluate_with_pytrec(run, qrels)

    # RBP (optional, needs trectools)
    qrels_path = ROBUST04_DIR / "qrels" / "test.tsv"
    # trectools needs the original TREC qrels format, not our TSV
    # Convert our qrels to TREC format for trectools
    trec_qrels_path = ROBUST04_DIR / "qrels" / "trec_format.qrels"
    if not trec_qrels_path.exists():
        with open(trec_qrels_path, "w") as f:
            for qid, docs in qrels.items():
                for doc_id, rel in docs.items():
                    f.write(f"{qid} 0 {doc_id} {rel}\n")

    rbp_per_query, rbp_agg = evaluate_rbp(args.run, str(trec_qrels_path), args.rbp_p, args.binary_relevance)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  EVALUATION: {run_name}")
    print("=" * 60)
    print(f"  query_count:    {len(run)}")
    print(f"  ndcg@10:        {agg['ndcg@10']:.6f}")
    print(f"  map@1000:       {agg['map@1000']:.6f}")
    print(f"  map@100:        {agg['map@100']:.6f}")
    print(f"  recall@100:     {agg['recall@100']:.6f}")
    if rbp_agg:
        print(f"  rbp(p={args.rbp_p}):    {rbp_agg['rbp']:.6f}")
        print(f"  rbp_residual:   {rbp_agg['rbp_residual']:.6f}")
    print("=" * 60)

    # Save outputs
    if args.output_dir:
        import pandas as pd

        out_dir = Path(args.output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-query metrics
        rows = []
        for qid in sorted(per_query.keys(), key=lambda x: int(x) if x.isdigit() else x):
            row = {"qid": qid}
            row["ndcg@10"] = per_query[qid].get("ndcg_cut_10", 0.0)
            row["map@1000"] = per_query[qid].get("map_cut_1000", 0.0)
            row["map@100"] = per_query[qid].get("map_cut_100", 0.0)
            row["recall@100"] = per_query[qid].get("recall_100", 0.0)
            rows.append(row)
        pq_df = pd.DataFrame(rows)

        # Merge RBP if available
        if rbp_per_query is not None:
            rbp_per_query["qid"] = rbp_per_query["query"].astype(str)
            pq_df = pq_df.merge(rbp_per_query[["qid", "rbp", "rbp_residual"]], on="qid", how="left")

        pq_path = out_dir / "per-query.tsv"
        pq_df.to_csv(pq_path, sep="\t", index=False)
        print(f"Saved per-query: {pq_path}")

        # Summary
        summary = {"run": run_name, "query_count": len(run), **agg}
        if rbp_agg:
            summary.update(rbp_agg)
        summary_df = pd.DataFrame([summary])
        summary_path = out_dir / "summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
