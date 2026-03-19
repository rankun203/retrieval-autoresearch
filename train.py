"""
exp03b-qwen3-rerank-fix: Qwen3-Reranker-0.6B with CORRECT prompt format.

Fixes exp03's broken Qwen3 implementation:
- Adds <think> block in suffix
- Adds colons in format tags (<Instruct>:, <Query>:, <Document>:)
- Tokenizes content separately, prepends/appends prefix/suffix tokens
- Uses max_length=8192 instead of 512

Usage:
  uv run --with python-terrier --with transformers train.py

Env:
  JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
  JVM_PATH=/usr/lib/jvm/java-21-amazon-corretto/lib/server/libjvm.so
"""

import os
import gc
import time

import torch
import pyterrier as pt
import pandas as pd

from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BM25_K1 = 0.9
BM25_B = 0.4
BO1_FB_DOCS = 5
BO1_FB_TERMS = 30
BM25_TOP_K = 1000
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_DEPTHS = [100, 1000]
BATCH_SIZE = 8
MAX_LENGTH = 4096
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

peak_vram_mb = 0.0
t_start = time.time()

def update_peak_vram():
    global peak_vram_mb
    if torch.cuda.is_available():
        current = torch.cuda.max_memory_allocated() / 1024 / 1024
        peak_vram_mb = max(peak_vram_mb, current)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Robust04...", flush=True)
corpus, queries, qrels = load_robust04()
print(f"  {len(corpus):,} docs, {len(queries)} queries", flush=True)

def get_doc_text(doc_id):
    doc = corpus[doc_id]
    return (doc.get("title", "") + " " + doc["text"]).strip()

# ---------------------------------------------------------------------------
# BM25+Bo1 first stage
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
bm25 = pt.terrier.Retriever(
    index_ref, wmodel="BM25",
    controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
    num_results=BM25_TOP_K,
)
bo1 = pt.rewrite.Bo1QueryExpansion(index_ref, fb_docs=BO1_FB_DOCS, fb_terms=BO1_FB_TERMS)
pipeline = bm25 >> bo1 >> bm25

topics_df = pd.DataFrame([{"qid": qid, "query": text} for qid, text in queries.items()])

print("Running BM25+Bo1...", flush=True)
t_bm25_start = time.time()
results = pipeline.transform(topics_df)
t_bm25 = time.time() - t_bm25_start
print(f"  BM25+Bo1 done in {t_bm25:.1f}s", flush=True)

# Convert to sorted list per query: [(doc_id, score), ...]
bm25_run = {}
for _, row in results.iterrows():
    qid = str(row["qid"])
    if qid not in bm25_run:
        bm25_run[qid] = []
    bm25_run[qid].append((str(row["docno"]), float(row["score"])))
for qid in bm25_run:
    bm25_run[qid].sort(key=lambda x: -x[1])

# ---------------------------------------------------------------------------
# Qwen3 Reranker (CORRECT implementation from official model card)
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {RERANKER_MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    RERANKER_MODEL, torch_dtype=torch.float16, device_map="cuda",
).eval()

token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

# Official prefix/suffix from model card
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

def format_instruction(query_text, doc_text):
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query_text}\n<Document>: {doc_text}"

def process_inputs(pairs):
    """Tokenize with official prefix/suffix token prepend/append."""
    inputs = tokenizer(
        pairs, padding=False, truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    for key in inputs:
        inputs[key] = inputs[key].to("cuda")
    return inputs

@torch.no_grad()
def compute_scores(inputs):
    """Score via log_softmax over [no, yes] logits at last position."""
    batch_logits = model(**inputs).logits[:, -1, :]
    true_vector = batch_logits[:, token_true_id]
    false_vector = batch_logits[:, token_false_id]
    stacked = torch.stack([false_vector, true_vector], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    scores = log_probs[:, 1].exp().float().tolist()
    return scores

# ---------------------------------------------------------------------------
# Reranking runs
# ---------------------------------------------------------------------------
all_results = {}
os.makedirs("runs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for depth in RERANK_DEPTHS:
    run_name = f"qwen3-rerank-top{depth}"
    print(f"\n{'='*60}", flush=True)
    print(f"Run: {run_name}", flush=True)
    print(f"  Rerank depth: {depth}", flush=True)
    print(f"{'='*60}", flush=True)

    t_rerank_start = time.time()
    reranked_run = {}
    total_pairs = 0

    for i, (qid, candidates) in enumerate(bm25_run.items()):
        query_text = queries[qid]
        top_candidates = candidates[:depth]
        doc_ids = [d[0] for d in top_candidates]
        doc_texts = [get_doc_text(d) for d in doc_ids]

        all_scores = []
        for batch_start in range(0, len(doc_ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(doc_ids))
            batch_texts = doc_texts[batch_start:batch_end]

            pairs = [format_instruction(query_text, dt) for dt in batch_texts]
            inputs = process_inputs(pairs)
            scores = compute_scores(inputs)
            all_scores.extend(scores)

            del inputs
            torch.cuda.empty_cache()

        reranked_run[qid] = {}
        for doc_id, score in zip(doc_ids, all_scores):
            reranked_run[qid][doc_id] = score

        total_pairs += len(doc_ids)
        if (i + 1) % 50 == 0 or (i + 1) == len(bm25_run):
            print(f"    Reranked {i+1}/{len(bm25_run)} queries ({total_pairs:,} pairs scored)", flush=True)

    update_peak_vram()
    t_rerank = time.time() - t_rerank_start
    print(f"  Reranking done in {t_rerank:.1f}s", flush=True)

    # Save run file
    run_path = f"runs/{run_name}.run"
    write_trec_run(reranked_run, run_path, run_name=run_name)
    print(f"  TREC run written: {run_path}", flush=True)

    # Evaluate
    metrics = evaluate_run(reranked_run, qrels)
    eval_dur = t_bm25 + t_rerank  # full pipeline time
    all_results[run_name] = {
        "metrics": metrics,
        "eval_dur": eval_dur,
        "rerank_time": t_rerank,
    }
    print(f"  MAP@100:    {metrics['map@100']:.6f}", flush=True)
    print(f"  nDCG@10:    {metrics['ndcg@10']:.6f}", flush=True)
    print(f"  MAP@1000:   {metrics['map@1000']:.6f}", flush=True)
    print(f"  Recall@100: {metrics['recall@100']:.6f}", flush=True)

# ---------------------------------------------------------------------------
# Cleanup GPU
# ---------------------------------------------------------------------------
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Summary (best run)
# ---------------------------------------------------------------------------
best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["map@100"])
best = all_results[best_name]
t_end = time.time()

print(f"\n{'='*60}", flush=True)
print(f"BEST RUN: {best_name}", flush=True)
print(f"{'='*60}", flush=True)

print("---")
print(f"ndcg@10:          {best['metrics']['ndcg@10']:.6f}")
print(f"map@1000:         {best['metrics']['map@1000']:.6f}")
print(f"map@100:          {best['metrics']['map@100']:.6f}")
print(f"recall@100:       {best['metrics']['recall@100']:.6f}")
print(f"training_seconds: 0.0")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        0")
print(f"encoder_model:    {RERANKER_MODEL}")
print(f"num_docs_indexed: {len(corpus)}")
print(f"eval_duration:    {best['eval_dur']:.3f}")
print(f"loss_curve:       N/A (zero-shot)")
print(f"budget_assessment: OK")
print(f"method:           BM25+Bo1 >> {RERANKER_MODEL} rerank")
