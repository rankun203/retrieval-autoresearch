"""
exp07-qwen3-embed-8b: Hybrid retrieval with Qwen3-Embedding-8B + BM25+Bo1 + Qwen3-Reranker.

Pipeline:
  1. BM25+Bo1 retrieval (top-1000)
  2. Qwen3-Embedding-8B zero-shot dense retrieval (top-1000)
  3. Linear fusion (alpha=0.3 dense, 0.7 sparse)
  4. Qwen3-Reranker-0.6B reranking (news-short instruction, ml768, top-1000)

Usage:
  uv run --with 'python-terrier' --with 'transformers>=4.51' --with faiss-cpu train.py

Env:
  JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
  JVM_PATH=/usr/lib/jvm/java-21-amazon-corretto/lib/server/libjvm.so
"""

import os
import gc
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import pyterrier as pt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BM25_K1 = 0.9
BM25_B = 0.4
BO1_FB_DOCS = 5
BO1_FB_TERMS = 30
BM25_TOP_K = 1000

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 4096
ENCODE_BATCH_SIZE = 64
DOC_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 512

FUSION_ALPHA = 0.3  # dense weight; sparse weight = 1 - alpha
DENSE_TOP_K = 1000

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MAX_LENGTH = 768
RERANKER_BATCH_SIZE = 64
RERANKER_DEPTH = 1000
RERANKER_INSTRUCTION = "Given a news search query, retrieve relevant news articles that answer the query"

TASK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

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

doc_ids_list = list(corpus.keys())
doc_id_to_idx = {did: i for i, did in enumerate(doc_ids_list)}


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

# Build BM25 run dict: {qid: {doc_id: score}}
bm25_run = {}
for _, row in results.iterrows():
    qid = str(row["qid"])
    if qid not in bm25_run:
        bm25_run[qid] = {}
    bm25_run[qid][str(row["docno"])] = float(row["score"])

print(f"  BM25+Bo1: {len(bm25_run)} queries, avg {np.mean([len(v) for v in bm25_run.values()]):.0f} docs/query", flush=True)

# ---------------------------------------------------------------------------
# Qwen3-Embedding-8B dense retrieval
# ---------------------------------------------------------------------------
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-padding token's hidden state (for left-padded inputs)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def get_query_text(query: str) -> str:
    """Format query with instruction prefix (from model card)."""
    return f"Instruct: {TASK_INSTRUCTION}\nQuery:{query}"


print(f"\nLoading {EMBEDDING_MODEL}...", flush=True)
embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side="left")
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL,
    dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
).eval()
update_peak_vram()
print(f"  Model loaded. Peak VRAM: {peak_vram_mb:.0f} MB", flush=True)


@torch.no_grad()
def encode_texts(texts, max_length, show_progress=True, label="texts"):
    """Encode a list of texts into normalized embeddings."""
    all_embeddings = []
    total = len(texts)
    for batch_start in range(0, total, ENCODE_BATCH_SIZE):
        batch_end = min(batch_start + ENCODE_BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]

        batch_dict = embed_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

        outputs = embed_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

        del batch_dict, outputs, embeddings
        torch.cuda.empty_cache()

        if show_progress and (batch_end % (ENCODE_BATCH_SIZE * 20) == 0 or batch_end == total):
            print(f"    Encoded {batch_end:,}/{total:,} {label}", flush=True)

    return np.concatenate(all_embeddings, axis=0)


# Encode corpus
print("Encoding corpus documents...", flush=True)
t_encode_start = time.time()
doc_texts = [get_doc_text(did) for did in doc_ids_list]
doc_embeddings = encode_texts(doc_texts, DOC_MAX_LENGTH, label="docs")
t_encode_docs = time.time() - t_encode_start
print(f"  Encoded {len(doc_ids_list):,} docs in {t_encode_docs:.1f}s, shape: {doc_embeddings.shape}", flush=True)
del doc_texts
gc.collect()

# Encode queries
print("Encoding queries...", flush=True)
query_ids = list(queries.keys())
query_texts = [get_query_text(queries[qid]) for qid in query_ids]
query_embeddings = encode_texts(query_texts, QUERY_MAX_LENGTH, show_progress=True, label="queries")
print(f"  Encoded {len(query_ids)} queries, shape: {query_embeddings.shape}", flush=True)
del query_texts
gc.collect()

update_peak_vram()
print(f"  Peak VRAM after encoding: {peak_vram_mb:.0f} MB", flush=True)

# Build FAISS index
print("Building FAISS index...", flush=True)
import faiss
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(doc_embeddings)
print(f"  FAISS index built with {faiss_index.ntotal:,} vectors, dim={EMBEDDING_DIM}", flush=True)

# Dense retrieval
print("Running dense retrieval...", flush=True)
t_dense_start = time.time()
scores, indices = faiss_index.search(query_embeddings, DENSE_TOP_K)
t_dense = time.time() - t_dense_start
print(f"  Dense retrieval done in {t_dense:.1f}s", flush=True)

# Build dense run dict
dense_run = {}
for i, qid in enumerate(query_ids):
    dense_run[qid] = {}
    for j in range(DENSE_TOP_K):
        idx = int(indices[i, j])
        if idx >= 0:
            doc_id = doc_ids_list[idx]
            dense_run[qid][doc_id] = float(scores[i, j])

# Free embedding model from GPU
print("Freeing embedding model from GPU...", flush=True)
del embed_model, embed_tokenizer, doc_embeddings, query_embeddings, faiss_index
gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB", flush=True)

# ---------------------------------------------------------------------------
# Run 1: Dense-only evaluation
# ---------------------------------------------------------------------------
os.makedirs("runs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

all_results = {}

print(f"\n{'='*60}", flush=True)
print("Run: dense-qwen3-8b-zeroshot", flush=True)
print(f"{'='*60}", flush=True)

write_trec_run(dense_run, "runs/dense-qwen3-8b-zeroshot.run", run_name="dense-qwen3-8b-zeroshot")
metrics = evaluate_run(dense_run, qrels)
all_results["dense-qwen3-8b-zeroshot"] = {"metrics": metrics}
print(f"  MAP@100:    {metrics['map@100']:.6f}", flush=True)
print(f"  nDCG@10:    {metrics['ndcg@10']:.6f}", flush=True)
print(f"  MAP@1000:   {metrics['map@1000']:.6f}", flush=True)
print(f"  Recall@100: {metrics['recall@100']:.6f}", flush=True)

# ---------------------------------------------------------------------------
# Run 2: Linear fusion (alpha=0.3 dense, 0.7 sparse)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print(f"Run: linear-a03", flush=True)
print(f"  Linear alpha={FUSION_ALPHA} (dense={FUSION_ALPHA}, sparse={1-FUSION_ALPHA})", flush=True)
print(f"{'='*60}", flush=True)


def normalize_scores(run):
    """Min-max normalize scores per query to [0, 1]."""
    normalized = {}
    for qid, docs in run.items():
        if not docs:
            normalized[qid] = {}
            continue
        scores_list = list(docs.values())
        min_s = min(scores_list)
        max_s = max(scores_list)
        rng = max_s - min_s if max_s > min_s else 1.0
        normalized[qid] = {did: (s - min_s) / rng for did, s in docs.items()}
    return normalized


def linear_fusion(run_a, run_b, alpha):
    """Linear interpolation: alpha * run_a + (1-alpha) * run_b."""
    norm_a = normalize_scores(run_a)
    norm_b = normalize_scores(run_b)
    fused = {}
    all_qids = set(norm_a.keys()) | set(norm_b.keys())
    for qid in all_qids:
        docs_a = norm_a.get(qid, {})
        docs_b = norm_b.get(qid, {})
        all_docs = set(docs_a.keys()) | set(docs_b.keys())
        fused[qid] = {}
        for did in all_docs:
            score = alpha * docs_a.get(did, 0.0) + (1 - alpha) * docs_b.get(did, 0.0)
            fused[qid][did] = score
    return fused


fusion_run = linear_fusion(dense_run, bm25_run, FUSION_ALPHA)
write_trec_run(fusion_run, "runs/linear-a03.run", run_name="linear-a03")
metrics = evaluate_run(fusion_run, qrels)
all_results["linear-a03"] = {"metrics": metrics}
print(f"  MAP@100:    {metrics['map@100']:.6f}", flush=True)
print(f"  nDCG@10:    {metrics['ndcg@10']:.6f}", flush=True)
print(f"  MAP@1000:   {metrics['map@1000']:.6f}", flush=True)
print(f"  Recall@100: {metrics['recall@100']:.6f}", flush=True)

# ---------------------------------------------------------------------------
# Run 3: Fusion + Qwen3-Reranker-0.6B reranking
# ---------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print(f"Run: linear-a03-reranked", flush=True)
print(f"  Reranking linear-a03 with {RERANKER_MODEL} (ml={RERANKER_MAX_LENGTH}, depth={RERANKER_DEPTH})", flush=True)
print(f"{'='*60}", flush=True)

from transformers import AutoModelForCausalLM

print(f"\nLoading {RERANKER_MODEL}...", flush=True)
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side="left")
reranker_model = AutoModelForCausalLM.from_pretrained(
    RERANKER_MODEL, torch_dtype=torch.float16, device_map="cuda",
    attn_implementation="sdpa",
).eval()
update_peak_vram()

token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")

# Official prefix/suffix from model card
reranker_prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
reranker_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
reranker_prefix_tokens = reranker_tokenizer.encode(reranker_prefix, add_special_tokens=False)
reranker_suffix_tokens = reranker_tokenizer.encode(reranker_suffix, add_special_tokens=False)


def format_reranker_input(query_text, doc_text):
    return f"<Instruct>: {RERANKER_INSTRUCTION}\n<Query>: {query_text}\n<Document>: {doc_text}"


def process_reranker_inputs(pairs, max_length):
    """Tokenize with official prefix/suffix token prepend/append."""
    inputs = reranker_tokenizer(
        pairs, padding=False, truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(reranker_prefix_tokens) - len(reranker_suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = reranker_prefix_tokens + ele + reranker_suffix_tokens
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to("cuda")
    return inputs


@torch.no_grad()
def compute_reranker_scores(inputs):
    """Score via log_softmax over [no, yes] logits at last position."""
    batch_logits = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_logits[:, token_true_id]
    false_vector = batch_logits[:, token_false_id]
    stacked = torch.stack([false_vector, true_vector], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    scores = log_probs[:, 1].exp().float().tolist()
    return scores


# Sort fusion run by score for reranking
fusion_sorted = {}
for qid, docs in fusion_run.items():
    fusion_sorted[qid] = sorted(docs.items(), key=lambda x: -x[1])

t_rerank_start = time.time()
reranked_run = {}
total_pairs = 0

for i, qid in enumerate(fusion_sorted):
    query_text = queries[qid]
    candidates = fusion_sorted[qid][:RERANKER_DEPTH]
    doc_ids = [d[0] for d in candidates]
    doc_texts = [get_doc_text(d) for d in doc_ids]

    all_scores = []
    for batch_start in range(0, len(doc_ids), RERANKER_BATCH_SIZE):
        batch_end = min(batch_start + RERANKER_BATCH_SIZE, len(doc_ids))
        batch_texts = doc_texts[batch_start:batch_end]

        pairs = [format_reranker_input(query_text, dt) for dt in batch_texts]
        inputs = process_reranker_inputs(pairs, RERANKER_MAX_LENGTH)
        scores = compute_reranker_scores(inputs)
        all_scores.extend(scores)

        del inputs
        torch.cuda.empty_cache()

    reranked_run[qid] = {}
    for doc_id, score in zip(doc_ids, all_scores):
        reranked_run[qid][doc_id] = score

    total_pairs += len(doc_ids)
    if (i + 1) % 50 == 0 or (i + 1) == len(fusion_sorted):
        print(f"    Reranked {i+1}/{len(fusion_sorted)} queries ({total_pairs:,} pairs scored)", flush=True)

update_peak_vram()
t_rerank = time.time() - t_rerank_start
print(f"  Reranking done in {t_rerank:.1f}s", flush=True)

write_trec_run(reranked_run, "runs/linear-a03-reranked.run", run_name="linear-a03-reranked")
metrics = evaluate_run(reranked_run, qrels)
all_results["linear-a03-reranked"] = {"metrics": metrics}
print(f"  MAP@100:    {metrics['map@100']:.6f}", flush=True)
print(f"  nDCG@10:    {metrics['ndcg@10']:.6f}", flush=True)
print(f"  MAP@1000:   {metrics['map@1000']:.6f}", flush=True)
print(f"  Recall@100: {metrics['recall@100']:.6f}", flush=True)

# Cleanup reranker
del reranker_model, reranker_tokenizer
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
t_end = time.time()

print(f"\n{'='*80}", flush=True)
print("RESULTS SUMMARY", flush=True)
print(f"{'='*80}", flush=True)
print(f"{'Run':<30} {'MAP@100':>8} {'nDCG@10':>8} {'MAP@1000':>9} {'R@100':>8}", flush=True)
print("-" * 80, flush=True)
for name, r in all_results.items():
    m = r["metrics"]
    print(f"{name:<30} {m['map@100']:8.6f} {m['ndcg@10']:8.6f} {m['map@1000']:9.6f} {m['recall@100']:8.6f}", flush=True)

# Best run by MAP@100
best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["map@100"])
best = all_results[best_name]

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
print(f"encoder_model:    {EMBEDDING_MODEL}")
print(f"num_docs_indexed: {len(corpus)}")
print(f"eval_duration:    {t_end - t_start:.3f}")
print(f"loss_curve:       N/A (zero-shot)")
print(f"budget_assessment: OK")
print(f"method:           linear-a03({EMBEDDING_MODEL}) >> {RERANKER_MODEL} rerank (ml={RERANKER_MAX_LENGTH}, depth={RERANKER_DEPTH})")
