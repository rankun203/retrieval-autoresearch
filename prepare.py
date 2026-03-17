"""
Dense retrieval autoresearch - fixed utilities. Do not modify.
One-time setup: uv run prepare.py
"""

import os
import re
import io
import json
import csv
import gzip
from pathlib import Path
from typing import Iterator

import requests

import numpy as np

DATA_DIR = Path(os.path.expanduser("~/.cache/autoresearch-retrieval"))
ROBUST04_DIR = DATA_DIR / "robust04"
TIME_BUDGET = 600  # seconds of training wall-clock time per experiment

# Publicly available from NIST (no license required for topics + qrels)
TREC_TOPICS_URL = "https://trec.nist.gov/data/robust/04.testset.gz"
TREC_QRELS_URL  = "https://trec.nist.gov/data/robust/qrels.robust2004.txt"
# Full Robust04 corpus via BEIR/HuggingFace (deduped from BeIR/robust04-generated-queries)
CORPUS_HF_DATASET = "BeIR/robust04-generated-queries"


_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _download_bytes(url: str) -> bytes:
    """Download URL content, using requests with a browser User-Agent."""
    r = requests.get(url, headers=_HEADERS, timeout=120)
    r.raise_for_status()
    return r.content


def _parse_trec_topics(text: str) -> dict:
    """Parse TREC-format topics file. Returns {qid: title_text}."""
    queries = {}
    for top in re.finditer(r"<top>(.*?)</top>", text, re.DOTALL):
        block = top.group(1)
        num_m = re.search(r"<num>\s*Number:\s*(\d+)", block)
        title_m = re.search(r"<title>\s*(.*?)(?=\n<|\Z)", block, re.DOTALL)
        if num_m and title_m:
            qid = num_m.group(1).strip()
            title = title_m.group(1).strip()
            queries[qid] = title
    return queries


def download_robust04():
    """Build Robust04 corpus, queries, and qrels under ROBUST04_DIR."""
    corpus_path = ROBUST04_DIR / "corpus.jsonl"
    if corpus_path.exists():
        print("Robust04 already prepared.")
        return

    ROBUST04_DIR.mkdir(parents=True, exist_ok=True)
    (ROBUST04_DIR / "qrels").mkdir(exist_ok=True)

    # --- 1. Queries (TREC topics) ---
    print("Downloading TREC 2004 Robust topics...")
    topics_gz_bytes = _download_bytes(TREC_TOPICS_URL)
    with gzip.open(io.BytesIO(topics_gz_bytes), "rt", encoding="latin-1") as f:
        topics_text = f.read()
    queries = _parse_trec_topics(topics_text)
    with open(ROBUST04_DIR / "queries.jsonl", "w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    print(f"  {len(queries)} queries saved.")

    # --- 2. Qrels ---
    print("Downloading TREC 2004 Robust qrels...")
    qrels_path = ROBUST04_DIR / "qrels" / "test.tsv"
    qrels_raw = _download_bytes(TREC_QRELS_URL).decode("latin-1")
    with open(qrels_path, "w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "doc-id", "score"])
        for line in qrels_raw.splitlines():
            parts = line.strip().split()
            if len(parts) >= 4:
                writer.writerow([parts[0], parts[1], parts[2], parts[3]])
    print(f"  Qrels saved to {qrels_path}")

    # --- 3. Corpus (BeIR/robust04-generated-queries, deduped) ---
    print("Building corpus from BeIR/robust04-generated-queries (~528K docs, streaming)...")
    from datasets import load_dataset
    ds = load_dataset(CORPUS_HF_DATASET, streaming=True, split="train")
    seen = set()
    n = 0
    with open(corpus_path, "w") as f:
        for ex in ds:
            doc_id = ex["_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            f.write(json.dumps({
                "_id": doc_id,
                "title": ex.get("title", ""),
                "text": ex.get("text", ""),
            }) + "\n")
            n += 1
            if n % 50000 == 0:
                print(f"  {n:,} docs...", end="\r", flush=True)
    print(f"  {n:,} unique docs saved to {corpus_path}")
    print(f"Robust04 ready at {ROBUST04_DIR}")


def load_robust04():
    """
    Returns (corpus, queries, qrels).
      corpus:  {doc_id: {"title": str, "text": str}}
      queries: {qid: str}
      qrels:   {qid: {doc_id: int}}  — only relevant docs (rel > 0)
    """
    corpus = {}
    with open(ROBUST04_DIR / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}

    queries = {}
    with open(ROBUST04_DIR / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    queries.pop("672", None)  # qid 672 has no relevance judgments, excluded by convention

    qrels = {}
    qrels_path = ROBUST04_DIR / "qrels" / "test.tsv"
    with open(qrels_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            qid, _, doc_id, rel = row[0], row[1], row[2], row[3]
            rel = int(rel)
            if rel > 0:
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel

    return corpus, queries, qrels


def evaluate_run(run: dict, qrels: dict) -> dict:
    """
    Evaluate a retrieval run against qrels.
    run:   {qid: {doc_id: score}}   (higher score = more relevant)
    qrels: {qid: {doc_id: int}}

    Returns dict with keys: ndcg@10, map@100, recall@1000
    The primary metric is ndcg@10 — higher is better.
    """
    import pytrec_eval
    measures = {"ndcg_cut_10", "map_cut_100", "recall_100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)
    out = {}
    for m_key, m_name in [("ndcg_cut_10", "ndcg@10"), ("map_cut_100", "map@100"), ("recall_100", "recall@100")]:
        vals = [v[m_key] for v in results.values() if m_key in v]
        out[m_name] = float(np.mean(vals)) if vals else 0.0
    return out


def stream_msmarco_triples() -> Iterator:
    """
    Streams (query_text, pos_text, neg_text) training triples from MS-MARCO passage.
    Uses HuggingFace datasets streaming — no full download required.
    Uses sentence-transformers/msmarco-bm25 (triplet config): query/positive/negative text fields.
    Loops forever (re-shuffles on each epoch).
    """
    from datasets import load_dataset
    while True:
        ds = load_dataset(
            "sentence-transformers/msmarco-bm25",
            name="triplet",
            split="train",
            streaming=True,
        )
        for ex in ds:
            query = ex.get("query", "")
            positive = ex.get("positive", "")
            negative = ex.get("negative", "")
            if not query or not positive or not negative:
                continue
            yield query, positive, negative


if __name__ == "__main__":
    print("=== Dense Retrieval Autoresearch Setup ===")
    download_robust04()
    corpus, queries, qrels = load_robust04()
    print(f"Robust04: {len(corpus):,} docs | {len(queries)} queries | "
          f"{sum(len(v) for v in qrels.values())} relevant judgments")
    # Quick sanity check on the stream (just peek 3 examples)
    print("Checking MS-MARCO stream (first 3 examples)...")
    stream = stream_msmarco_triples()
    for i, (q, p, n) in enumerate(stream):
        print(f"  [{i}] query={q[:60]!r}")
        if i >= 2:
            break
    print("Setup complete. Ready for experiments.")
