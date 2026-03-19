"""
Baseline: BM25+Bo1 retrieval on Robust04 via PyTerrier.

Usage: uv run --with python-terrier train.py
Env: JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
     JVM_PATH=/usr/lib/jvm/java-21-amazon-corretto/lib/server/libjvm.so
"""

import os
import time

import pyterrier as pt
import pandas as pd

from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKTREE_NAME = os.path.basename(os.getcwd())
BM25_K1 = 0.9
BM25_B = 0.4
BO1_FB_DOCS = 5
BO1_FB_TERMS = 30
TOP_K = 1000

t_start = time.time()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Robust04...", flush=True)
corpus, queries, qrels = load_robust04()
print(f"  {len(corpus):,} docs, {len(queries)} queries", flush=True)

# ---------------------------------------------------------------------------
# Build/load Terrier index
# ---------------------------------------------------------------------------
if not pt.java.started():
    pt.java.init()

index_path = os.path.expanduser("~/.cache/autoresearch-retrieval/terrier_index")
if not os.path.exists(os.path.join(index_path, "data.properties")):
    print("Building Terrier index...", flush=True)
    indexer = pt.IterDictIndexer(index_path, meta={"docno": 26})
    def doc_iter():
        for doc_id, doc in corpus.items():
            text = (doc.get("title", "") + " " + doc["text"]).strip()
            yield {"docno": doc_id, "text": text}
    indexer.index(doc_iter())
    print("  Index built.", flush=True)
else:
    print("  Using cached Terrier index.", flush=True)

index_ref = pt.IndexRef.of(os.path.join(index_path, "data.properties"))

# ---------------------------------------------------------------------------
# BM25 + Bo1 pipeline
# ---------------------------------------------------------------------------
bm25 = pt.terrier.Retriever(
    index_ref, wmodel="BM25",
    controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
    num_results=TOP_K,
)
bo1 = pt.rewrite.Bo1QueryExpansion(index_ref, fb_docs=BO1_FB_DOCS, fb_terms=BO1_FB_TERMS)
pipeline = bm25 >> bo1 >> bm25

# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
topics_df = pd.DataFrame([{"qid": qid, "query": text} for qid, text in queries.items()])

print("Running BM25+Bo1...", flush=True)
t_eval_start = time.time()
results = pipeline.transform(topics_df)
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
# Save run file
# ---------------------------------------------------------------------------
run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}.run"
write_trec_run(run, run_path, run_name=WORKTREE_NAME)
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
print(f"training_seconds: 0.0")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     0.0")
print(f"num_steps:        0")
print(f"encoder_model:    pyterrier-BM25+Bo1")
print(f"num_docs_indexed: {len(corpus)}")
print(f"eval_duration:    {eval_dur:.3f}")
print(f"method:           BM25(k1={BM25_K1},b={BM25_B})+Bo1(fbDocs={BO1_FB_DOCS},fbTerms={BO1_FB_TERMS})")
