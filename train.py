"""
Dense retrieval autoresearch — baseline bi-encoder.
Fine-tunes a small encoder on MS-MARCO, evaluates on Robust04 (nDCG@10).

What the agent modifies: anything in this file.
What the agent must NOT modify: prepare.py and its evaluate_run() function.

Usage: uv run train.py
"""

import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from prepare import (
    DATA_DIR, TIME_BUDGET, load_robust04, evaluate_run, write_trec_run, stream_msmarco_triples
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these
# ---------------------------------------------------------------------------

ENCODER_MODEL = "intfloat/e5-base-v2"   # larger backbone: 109M params, 768-dim
MAX_QUERY_LEN = 96    # longer queries for verbose TREC topics (Robust04)
MAX_DOC_LEN = 220     # max tokens for documents (longer for more context)
BATCH_SIZE = 64       # reduced from 128 to fit larger model in VRAM
LR = 1e-5             # learning rate (conservative fine-tuning)
TEMPERATURE = 0.05    # InfoNCE temperature (softer distribution)
ENCODE_BATCH = 256    # reduced from 512 to fit larger model in VRAM

# E5 models require task prefixes at both train and eval time
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, q_ids, q_mask, p_ids, p_mask, n_ids, n_mask):
        q_emb = self.encode(q_ids, q_mask)
        p_emb = self.encode(p_ids, p_mask)
        n_emb = self.encode(n_ids, n_mask)
        return q_emb, p_emb, n_emb


def infonce_loss(q, p, n, temperature=0.02):
    """Symmetric in-batch negatives InfoNCE + 1 hard negative per sample.

    Computes both q→p and p→q directions and averages them, doubling
    the effective training signal without extra memory cost.
    """
    # q: [B, D], p: [B, D], n: [B, D]
    q = F.normalize(q, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    # Combine positives and negatives as candidates
    docs = torch.cat([p, n], dim=0)  # [2B, D]
    labels = torch.arange(len(q), device=q.device)
    # q → docs (standard direction)
    loss_qp = F.cross_entropy((q @ docs.T) / temperature, labels)
    # p → queries (symmetric direction, using only positive passages)
    loss_pq = F.cross_entropy((p @ q.T) / temperature, labels)
    return 0.5 * (loss_qp + loss_pq)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
model = BiEncoder(ENCODER_MODEL).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

train_stream = stream_msmarco_triples()

# ---------------------------------------------------------------------------
# Training loop (fixed TIME_BUDGET seconds)
# ---------------------------------------------------------------------------

t_train_start = time.time()
total_training_time = 0.0
step = 0
smooth_loss = 0.0
loss_curve = []
_next_curve_pct = 0
debiased = 0.0

print(f"Training for {TIME_BUDGET}s on MS-MARCO...")

model.train()
while True:
    t0 = time.time()

    # Build a batch
    queries, positives, negatives = [], [], []
    for _ in range(BATCH_SIZE):
        q, p, n = next(train_stream)
        queries.append(QUERY_PREFIX + q)
        positives.append(PASSAGE_PREFIX + p)
        negatives.append(PASSAGE_PREFIX + n)

    def tokenize(texts, max_len):
        return tokenizer(
            texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt"
        )

    q_enc = tokenize(queries, MAX_QUERY_LEN)
    p_enc = tokenize(positives, MAX_DOC_LEN)
    n_enc = tokenize(negatives, MAX_DOC_LEN)

    q_ids = q_enc["input_ids"].to(device)
    q_mask = q_enc["attention_mask"].to(device)
    p_ids = p_enc["input_ids"].to(device)
    p_mask = p_enc["attention_mask"].to(device)
    n_ids = n_enc["input_ids"].to(device)
    n_mask = n_enc["attention_mask"].to(device)

    optimizer.zero_grad()
    q_emb, p_emb, n_emb = model(q_ids, q_mask, p_ids, p_mask, n_ids, n_mask)
    loss = infonce_loss(q_emb, p_emb, n_emb, TEMPERATURE)
    loss.backward()
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    if step > 2:  # skip warmup steps
        total_training_time += dt

    loss_val = loss.item()
    if math.isnan(loss_val):
        print(f"\nNaN loss at step {step}, aborting.")
        raise RuntimeError("NaN loss")
    smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
    debiased = smooth_loss / (1 - 0.9 ** (step + 1))
    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\rstep {step:05d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    pct_done = int(100 * total_training_time / TIME_BUDGET)
    while _next_curve_pct <= pct_done and _next_curve_pct <= 100:
        loss_curve.append((_next_curve_pct, round(debiased, 4)))
        _next_curve_pct += 10

    step += 1
    if step > 2 and total_training_time >= TIME_BUDGET:
        break

print()
print(f"Training done: {step} steps, {total_training_time:.1f}s")

# Free optimizer and gradient state to reclaim VRAM before corpus encoding
del optimizer
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluation timer (covers encoding + indexing + retrieval + reranking + eval)
t_eval_start = time.time()

# ---------------------------------------------------------------------------
# Index: encode Robust04 corpus
# ---------------------------------------------------------------------------

print("Loading Robust04...")
corpus, queries_dict, qrels = load_robust04()

doc_ids = list(corpus.keys())
doc_texts = [
    PASSAGE_PREFIX + (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
    for d in doc_ids
]

print(f"Encoding {len(doc_ids):,} documents...")
model.eval()
all_doc_embs = []
with torch.no_grad():
    for i in range(0, len(doc_texts), ENCODE_BATCH):
        batch = doc_texts[i : i + ENCODE_BATCH]
        enc = tokenizer(batch, max_length=MAX_DOC_LEN, padding=True, truncation=True, return_tensors="pt")
        embs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_doc_embs.append(F.normalize(embs, dim=-1).cpu().float().numpy())
        if (i // ENCODE_BATCH) % 20 == 0:
            print(f"  {i:,}/{len(doc_texts):,}", end="\r", flush=True)

doc_matrix = np.vstack(all_doc_embs)  # [N, D]
print(f"  Encoded {len(doc_matrix):,} docs, shape {doc_matrix.shape}")

# Build FAISS index on GPU
import faiss
dim = doc_matrix.shape[1]
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatIP(dim)  # inner product (cosine after normalization)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(doc_matrix)
print(f"FAISS index built on GPU: {index.ntotal:,} vectors")

# ---------------------------------------------------------------------------
# Retrieve: encode queries and search
# ---------------------------------------------------------------------------

query_ids = list(queries_dict.keys())
query_texts = [QUERY_PREFIX + queries_dict[qid] for qid in query_ids]

print(f"Encoding {len(query_ids)} queries...")
with torch.no_grad():
    q_enc = tokenizer(query_texts, max_length=MAX_QUERY_LEN, padding=True, truncation=True, return_tensors="pt")
    q_embs = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    q_matrix = F.normalize(q_embs, dim=-1).cpu().float().numpy()

TOP_K = 1000
scores_matrix, indices_matrix = index.search(q_matrix, TOP_K)

run = {}
for qi, qid in enumerate(query_ids):
    run[qid] = {doc_ids[idx]: float(scores_matrix[qi, rank]) for rank, idx in enumerate(indices_matrix[qi])}

# ---------------------------------------------------------------------------
# Rerank: cross-encoder rescoring of top-100 per query
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 1000
RERANK_BATCH = 16

print(f"Loading reranker: {RERANKER_MODEL}")
# Aggressively free bi-encoder, FAISS, and all encoding artifacts
del model, index, res, index_cpu
del all_doc_embs, doc_matrix, q_matrix, q_embs, scores_matrix, indices_matrix
gc.collect()
torch.cuda.empty_cache()

reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(device)
reranker.eval()

reranked_run = {}
for qi, qid in enumerate(query_ids):
    query_text = queries_dict[qid]
    # Get top-K candidates from first-stage retrieval
    sorted_docs = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:RERANK_TOP_K]
    candidate_ids = [did for did, _ in sorted_docs]
    candidate_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]).strip() for did in candidate_ids]

    # Score with cross-encoder in batches
    all_scores = []
    for j in range(0, len(candidate_texts), RERANK_BATCH):
        batch_texts = candidate_texts[j:j + RERANK_BATCH]
        pairs = [[query_text, doc_text] for doc_text in batch_texts]
        enc = reranker_tokenizer(pairs, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            scores = reranker(**enc).logits.squeeze(-1).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        all_scores.extend(scores)

    reranked_run[qid] = {did: score for did, score in zip(candidate_ids, all_scores)}
    if (qi + 1) % 50 == 0:
        print(f"  Reranked {qi + 1}/{len(query_ids)} queries")

print(f"Reranking done: {len(query_ids)} queries × top-{RERANK_TOP_K}")
run = reranked_run

# ---------------------------------------------------------------------------
# Write TREC run file
# ---------------------------------------------------------------------------

import subprocess
worktree_name = subprocess.check_output(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
).strip().replace("autoresearch/", "")
run_path = f"runs/{worktree_name}/{worktree_name}.run"
write_trec_run(run, run_path, run_name=worktree_name)
print(f"TREC run written: {run_path}")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

metrics = evaluate_run(run, qrels)
t_eval_end = time.time()
eval_duration = t_eval_end - t_eval_start

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

print("---")
print(f"ndcg@10:          {metrics['ndcg@10']:.6f}")
print(f"map@1000:         {metrics['map@1000']:.6f}")
print(f"map@100:          {metrics['map@100']:.6f}")
print(f"recall@100:       {metrics['recall@100']:.6f}")
print(f"run_file:         {run_path}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"encoder_model:    {ENCODER_MODEL}")
print(f"num_docs_indexed: {len(doc_ids)}")
print(f"eval_duration:    {eval_duration:.3f}")
if loss_curve:
    if not loss_curve or loss_curve[-1][0] < 100:
        loss_curve.append((100, round(debiased, 4)))
    curve_str = "  ".join(f"{p}%:{l}" for p, l in loss_curve)
    print(f"loss_curve:       {curve_str}")
    if len(loss_curve) >= 2:
        drop_first_half = loss_curve[0][1] - loss_curve[len(loss_curve)//2][1]
        drop_second_half = loss_curve[len(loss_curve)//2][1] - loss_curve[-1][1]
        if drop_second_half > 0.5 * drop_first_half:
            print("budget_assessment: UNDERTRAINED — loss still dropping, consider longer budget")
        elif drop_second_half < 0.05 * drop_first_half:
            print("budget_assessment: OVERFIT/PLATEAU — loss flat in second half, budget may be too long")
        else:
            print("budget_assessment: OK — loss curve looks healthy")
