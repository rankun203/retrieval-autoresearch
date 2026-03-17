"""
BM25+Bo1 PRF baseline for Robust04 using PyTerrier.

Uses Terrier's native BM25 + Bo1 query expansion with proper Snowball stemming.
CPU only, no GPU needed.

Usage: uv run --with python-terrier train.py
"""

import os
import sys
import time

sys.path.insert(0, ".")
from prepare import load_robust04, evaluate_run, write_trec_run

import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import pandas as pd

t_start = time.time()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Robust04...", flush=True)
corpus, queries, qrels = load_robust04()
print(f"  {len(corpus):,} docs, {len(queries)} queries", flush=True)

# ---------------------------------------------------------------------------
# Build Terrier index
# ---------------------------------------------------------------------------

index_path = os.path.expanduser("~/.cache/autoresearch-retrieval/terrier_index")
if not os.path.exists(os.path.join(index_path, "data.properties")):
    print("Building Terrier index...", flush=True)
    os.makedirs(index_path, exist_ok=True)
    docs = [
        {"docno": did, "text": (corpus[did]["title"] + " " + corpus[did]["text"]).strip()}
        for did in corpus
    ]
    indexer = pt.IterDictIndexer(index_path, overwrite=True)
    indexer.index(docs)
    print("  Index built.", flush=True)
else:
    print("  Using cached Terrier index.", flush=True)

print("Loading index...", flush=True)
index = pt.IndexFactory.of(os.path.abspath(index_path), memory=["meta", "lexicon", "inverted"])

# ---------------------------------------------------------------------------
# BM25 + Bo1 pipeline
# ---------------------------------------------------------------------------

bm25 = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.k_1": "0.9", "bm25.b": "0.4"})
bo1 = pt.rewrite.Bo1QueryExpansion(index, fb_docs=5, fb_terms=30)
pipeline = bm25 >> bo1 >> bm25

queries_df = pd.DataFrame([{"qid": qid, "query": text} for qid, text in queries.items()])

# ---------------------------------------------------------------------------
# Run retrieval
# ---------------------------------------------------------------------------

t_eval_start = time.time()
print("Running BM25 + Bo1 PRF...", flush=True)
results = pipeline.transform(queries_df)
eval_dur = time.time() - t_eval_start
print(f"  Retrieval done in {eval_dur:.1f}s", flush=True)

# Convert to run dict
run = {}
for _, row in results.iterrows():
    qid = str(row["qid"])
    if qid not in run:
        run[qid] = {}
    run[qid][str(row["docno"])] = float(row["score"])

# ---------------------------------------------------------------------------
# Write TREC run file
# ---------------------------------------------------------------------------

import subprocess
worktree_name = subprocess.check_output(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
).strip().replace("autoresearch/", "")
run_path = f"runs/{worktree_name}/{worktree_name}.run"
write_trec_run(run, run_path, run_name=worktree_name)
print(f"TREC run written: {run_path}", flush=True)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

metrics = evaluate_run(run, qrels)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()

print("---")
print(f"ndcg@10:          {metrics['ndcg@10']:.6f}")
print(f"map@1000:         {metrics['map@1000']:.6f}")
print(f"map@100:          {metrics['map@100']:.6f}")
print(f"recall@100:       {metrics['recall@100']:.6f}")
print(f"run_file:         {run_path}")
print(f"training_seconds: 0.0")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     0.0")
print(f"num_steps:        0")
print(f"encoder_model:    pyterrier-BM25+Bo1")
print(f"num_docs_indexed: {len(corpus)}")
print(f"eval_duration:    {eval_dur:.3f}")
print(f"method:           BM25(k1=0.9,b=0.4)+Bo1(fbDocs=5,fbTerms=30) via pyterrier")
