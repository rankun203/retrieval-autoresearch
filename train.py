"""
exp30-hard-negatives: Two-phase training with hard negative mining.

Phase 1: Train e5-base-v2 on MS-MARCO with InfoNCE (300s)
Mine: Use phase-1 model to retrieve top-100 from Robust04, collect non-relevant as hard negatives
Phase 2: Continue training on mix of MS-MARCO + Robust04 hard negatives (300s)
Evaluate: Encode, fuse with BM25, rerank with Qwen3-Reranker top-100
"""

import gc
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from prepare import (
    DATA_DIR, load_robust04, evaluate_run, write_trec_run,
    stream_msmarco_triples
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENCODER_MODEL = "intfloat/e5-base-v2"
MAX_QUERY_LEN = 96
MAX_DOC_LEN = 220
BATCH_SIZE = 64
LR = 1e-5
TEMPERATURE = 0.05
ENCODE_BATCH = 512

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Two-phase time budget
PHASE1_BUDGET = 300  # seconds
PHASE2_BUDGET = 300  # seconds
TOTAL_TIME_BUDGET = PHASE1_BUDGET + PHASE2_BUDGET

# Hard negative mining config
MINE_TOP_K = 100  # retrieve top-K per query for mining
HN_PER_QUERY = 10  # hard negatives to keep per query
ROBUST04_BATCH_RATIO = 0.3  # fraction of phase-2 batches from Robust04 hard negatives

BM25_RUN_FILE = "../../runs/exp22-bm25-prf/exp22-bm25-prf.run"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_TOP_K = 100
RERANK_BATCH = 4
MAX_CONTENT_TOKENS = 512
RRF_K = 60
TASK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

WORKTREE_NAME = "exp30-hard-negatives"

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ===========================================================================
# BiEncoder model
# ===========================================================================

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
    q = F.normalize(q, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    docs = torch.cat([p, n], dim=0)
    labels = torch.arange(len(q), device=q.device)
    loss_qp = F.cross_entropy((q @ docs.T) / temperature, labels)
    loss_pq = F.cross_entropy((p @ q.T) / temperature, labels)
    return 0.5 * (loss_qp + loss_pq)


# ===========================================================================
# PHASE 1: Train on MS-MARCO (300s)
# ===========================================================================

tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
model = BiEncoder(ENCODER_MODEL).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_stream = stream_msmarco_triples()

t_train_start = time.time()
total_training_time = 0.0
step = 0
smooth_loss = 0.0
loss_curve = []
_next_curve_pct = 0
debiased = 0.0

print(f"=== PHASE 1: Training e5-base-v2 for {PHASE1_BUDGET}s on MS-MARCO ===", flush=True)

model.train()
while True:
    t0 = time.time()

    queries_batch, positives, negatives = [], [], []
    for _ in range(BATCH_SIZE):
        q, p, n = next(train_stream)
        queries_batch.append(QUERY_PREFIX + q)
        positives.append(PASSAGE_PREFIX + p)
        negatives.append(PASSAGE_PREFIX + n)

    def tokenize(texts, max_len):
        return tokenizer(
            texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt"
        )

    q_enc = tokenize(queries_batch, MAX_QUERY_LEN)
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
    if step > 2:
        total_training_time += dt

    loss_val = loss.item()
    if math.isnan(loss_val):
        print(f"\nNaN loss at step {step}, aborting.", flush=True)
        raise RuntimeError("NaN loss")
    smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
    debiased = smooth_loss / (1 - 0.9 ** (step + 1))
    remaining = max(0, PHASE1_BUDGET - total_training_time)
    print(f"\rP1 step {step:05d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    # Track loss curve across full training (both phases)
    pct_done = int(100 * total_training_time / TOTAL_TIME_BUDGET)
    while _next_curve_pct <= pct_done and _next_curve_pct <= 100:
        loss_curve.append((_next_curve_pct, round(debiased, 4)))
        _next_curve_pct += 10

    step += 1
    if step > 2 and total_training_time >= PHASE1_BUDGET:
        break

phase1_steps = step
print(flush=True)
print(f"Phase 1 done: {phase1_steps} steps, {total_training_time:.1f}s, loss={debiased:.4f}", flush=True)

# ===========================================================================
# MINE HARD NEGATIVES from Robust04
# ===========================================================================

print("\n=== MINING HARD NEGATIVES ===", flush=True)
t_mine_start = time.time()

print("Loading Robust04...", flush=True)
corpus, queries_dict, qrels = load_robust04()

doc_ids = list(corpus.keys())
doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
doc_texts = [
    PASSAGE_PREFIX + (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
    for d in doc_ids
]

# Encode corpus with phase-1 model
print(f"Encoding {len(doc_ids):,} documents for mining...", flush=True)
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

doc_matrix = np.vstack(all_doc_embs)
print(f"  Encoded {len(doc_matrix):,} docs, shape {doc_matrix.shape}", flush=True)

# Build FAISS index for mining
import faiss
dim = doc_matrix.shape[1]
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatIP(dim)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(doc_matrix)
print(f"FAISS index built: {index.ntotal:,} vectors", flush=True)

# Encode queries and search
query_ids = list(queries_dict.keys())
query_texts = [QUERY_PREFIX + queries_dict[qid] for qid in query_ids]

print(f"Encoding {len(query_ids)} queries...", flush=True)
with torch.no_grad():
    q_enc = tokenizer(query_texts, max_length=MAX_QUERY_LEN, padding=True, truncation=True, return_tensors="pt")
    q_embs = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    q_matrix = F.normalize(q_embs, dim=-1).cpu().float().numpy()

scores_matrix, indices_matrix = index.search(q_matrix, MINE_TOP_K)

# Collect hard negatives: top-ranked docs that are NOT relevant
hard_negatives = {}  # qid -> list of doc_ids
positive_docs = {}   # qid -> list of doc_ids (from qrels)
total_hn = 0
total_pos = 0

for qi, qid in enumerate(query_ids):
    relevant = set(qrels.get(qid, {}).keys())
    positives_for_q = []
    negatives_for_q = []

    for rank in range(MINE_TOP_K):
        doc_idx = indices_matrix[qi, rank]
        did = doc_ids[doc_idx]

        if did in relevant:
            positives_for_q.append(did)
        else:
            if len(negatives_for_q) < HN_PER_QUERY:
                negatives_for_q.append(did)

    hard_negatives[qid] = negatives_for_q
    positive_docs[qid] = positives_for_q if positives_for_q else list(relevant)[:5]
    total_hn += len(negatives_for_q)
    total_pos += len(positive_docs[qid])

# Also add all relevant docs from qrels as positives (even if not in top-K)
for qid in query_ids:
    relevant = set(qrels.get(qid, {}).keys())
    existing = set(positive_docs.get(qid, []))
    for did in relevant:
        if did not in existing and did in doc_id_to_idx:
            positive_docs[qid].append(did)
    total_pos = sum(len(v) for v in positive_docs.values())

print(f"Mined {total_hn} hard negatives across {len(hard_negatives)} queries", flush=True)
print(f"  Avg HN per query: {total_hn / len(hard_negatives):.1f}", flush=True)
print(f"  Total positives: {total_pos}, avg per query: {total_pos / len(positive_docs):.1f}", flush=True)

# Clean up mining FAISS index
del index, res, index_cpu, all_doc_embs, doc_matrix, q_matrix, q_embs
del scores_matrix, indices_matrix
gc.collect()
torch.cuda.empty_cache()

t_mine_end = time.time()
print(f"Mining done in {t_mine_end - t_mine_start:.1f}s", flush=True)

# ===========================================================================
# PHASE 2: Continue training with MS-MARCO + Robust04 hard negatives (300s)
# ===========================================================================

print(f"\n=== PHASE 2: Training for {PHASE2_BUDGET}s with hard negatives ===", flush=True)

# Build Robust04 training data: (query_text, positive_doc_text, hard_negative_doc_text)
robust04_triples = []
for qid in query_ids:
    q_text = queries_dict[qid]
    pos_list = positive_docs.get(qid, [])
    neg_list = hard_negatives.get(qid, [])
    if not pos_list or not neg_list:
        continue
    for pos_did in pos_list:
        for neg_did in neg_list[:3]:  # top-3 hardest negatives per positive
            pos_doc = corpus[pos_did]
            neg_doc = corpus[neg_did]
            pos_text = (pos_doc["title"] + " " + pos_doc["text"]).strip()
            neg_text = (neg_doc["title"] + " " + neg_doc["text"]).strip()
            robust04_triples.append((q_text, pos_text, neg_text))

random.shuffle(robust04_triples)
print(f"Built {len(robust04_triples)} Robust04 training triples", flush=True)

model.train()
robust04_idx = 0
phase2_training_time = 0.0

while True:
    t0 = time.time()

    # Decide if this batch is MS-MARCO or Robust04
    use_robust04 = (random.random() < ROBUST04_BATCH_RATIO) and len(robust04_triples) > 0

    queries_batch, positives_batch, negatives_batch = [], [], []

    if use_robust04:
        for _ in range(BATCH_SIZE):
            q, p, n = robust04_triples[robust04_idx % len(robust04_triples)]
            robust04_idx += 1
            queries_batch.append(QUERY_PREFIX + q)
            positives_batch.append(PASSAGE_PREFIX + p)
            negatives_batch.append(PASSAGE_PREFIX + n)
    else:
        for _ in range(BATCH_SIZE):
            q, p, n = next(train_stream)
            queries_batch.append(QUERY_PREFIX + q)
            positives_batch.append(PASSAGE_PREFIX + p)
            negatives_batch.append(PASSAGE_PREFIX + n)

    def tokenize(texts, max_len):
        return tokenizer(
            texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt"
        )

    q_enc = tokenize(queries_batch, MAX_QUERY_LEN)
    p_enc = tokenize(positives_batch, MAX_DOC_LEN)
    n_enc = tokenize(negatives_batch, MAX_DOC_LEN)

    q_ids_t = q_enc["input_ids"].to(device)
    q_mask = q_enc["attention_mask"].to(device)
    p_ids_t = p_enc["input_ids"].to(device)
    p_mask = p_enc["attention_mask"].to(device)
    n_ids_t = n_enc["input_ids"].to(device)
    n_mask = n_enc["attention_mask"].to(device)

    optimizer.zero_grad()
    q_emb, p_emb, n_emb = model(q_ids_t, q_mask, p_ids_t, p_mask, n_ids_t, n_mask)
    loss = infonce_loss(q_emb, p_emb, n_emb, TEMPERATURE)
    loss.backward()
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    phase2_training_time += dt
    total_training_time += dt

    loss_val = loss.item()
    if math.isnan(loss_val):
        print(f"\nNaN loss at step {step}, aborting.", flush=True)
        raise RuntimeError("NaN loss")
    smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
    debiased = smooth_loss / (1 - 0.9 ** (step + 1))
    remaining = max(0, PHASE2_BUDGET - phase2_training_time)
    src = "R04" if use_robust04 else "MSM"
    print(f"\rP2 step {step:05d} | loss: {debiased:.4f} | {src} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    pct_done = int(100 * total_training_time / TOTAL_TIME_BUDGET)
    while _next_curve_pct <= pct_done and _next_curve_pct <= 100:
        loss_curve.append((_next_curve_pct, round(debiased, 4)))
        _next_curve_pct += 10

    step += 1
    if phase2_training_time >= PHASE2_BUDGET:
        break

print(flush=True)
print(f"Phase 2 done: {step - phase1_steps} steps, {phase2_training_time:.1f}s, loss={debiased:.4f}", flush=True)
print(f"Total training: {step} steps, {total_training_time:.1f}s", flush=True)

del optimizer
gc.collect()
torch.cuda.empty_cache()

# ===========================================================================
# STAGE 2: Encode corpus + dense retrieval
# ===========================================================================

t_eval_start = time.time()

doc_texts_eval = [
    PASSAGE_PREFIX + (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
    for d in doc_ids
]

print(f"\nEncoding {len(doc_ids):,} documents with phase-2 model...", flush=True)
model.eval()
all_doc_embs = []
with torch.no_grad():
    for i in range(0, len(doc_texts_eval), ENCODE_BATCH):
        batch = doc_texts_eval[i : i + ENCODE_BATCH]
        enc = tokenizer(batch, max_length=MAX_DOC_LEN, padding=True, truncation=True, return_tensors="pt")
        embs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_doc_embs.append(F.normalize(embs, dim=-1).cpu().float().numpy())
        if (i // ENCODE_BATCH) % 20 == 0:
            print(f"  {i:,}/{len(doc_texts_eval):,}", end="\r", flush=True)

doc_matrix = np.vstack(all_doc_embs)
print(f"  Encoded {len(doc_matrix):,} docs, shape {doc_matrix.shape}", flush=True)

# Build FAISS index
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatIP(dim)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(doc_matrix)
print(f"FAISS index built on GPU: {index.ntotal:,} vectors", flush=True)

# Encode queries and search
print(f"Encoding {len(query_ids)} queries...", flush=True)
with torch.no_grad():
    q_enc = tokenizer(query_texts, max_length=MAX_QUERY_LEN, padding=True, truncation=True, return_tensors="pt")
    q_embs = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    q_matrix = F.normalize(q_embs, dim=-1).cpu().float().numpy()

TOP_K = 1000
scores_matrix, indices_matrix = index.search(q_matrix, TOP_K)

dense_run = {}
for qi, qid in enumerate(query_ids):
    dense_run[qid] = {doc_ids[idx]: float(scores_matrix[qi, rank]) for rank, idx in enumerate(indices_matrix[qi])}

# Save dense run
dense_run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}-dense.run"
write_trec_run(dense_run, dense_run_path, run_name=WORKTREE_NAME + "-dense")
print(f"Dense run saved: {dense_run_path}", flush=True)

# Evaluate dense-only
dense_metrics = evaluate_run(dense_run, qrels)
print(f"Dense-only: MAP@100={dense_metrics['map@100']:.4f}, nDCG@10={dense_metrics['ndcg@10']:.4f}, recall@100={dense_metrics['recall@100']:.4f}", flush=True)

# Free bi-encoder and FAISS
del model, index, res, index_cpu, all_doc_embs, doc_matrix, q_matrix, q_embs
del scores_matrix, indices_matrix
gc.collect()
torch.cuda.empty_cache()

# ===========================================================================
# STAGE 3: Load BM25+Bo1 run and fuse with RRF
# ===========================================================================

print(f"\nLoading BM25+Bo1 run: {BM25_RUN_FILE}", flush=True)
bm25_run = {}
with open(BM25_RUN_FILE) as f:
    for line in f:
        parts = line.strip().split()
        qid, _, docid, rank, score, _ = parts
        if qid not in bm25_run:
            bm25_run[qid] = {}
        bm25_run[qid][docid] = float(score)

for qid in bm25_run:
    sorted_docs = sorted(bm25_run[qid].items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    bm25_run[qid] = dict(sorted_docs)

print(f"  BM25 run: {len(bm25_run)} queries", flush=True)

# RRF fusion
print(f"Fusing with RRF (k={RRF_K})...", flush=True)
fused_run = {}
all_qids = set(dense_run.keys()) | set(bm25_run.keys())

for qid in all_qids:
    bm25_docs = bm25_run.get(qid, {})
    bm25_ranked = sorted(bm25_docs.items(), key=lambda x: x[1], reverse=True)
    bm25_rank = {did: rank + 1 for rank, (did, _) in enumerate(bm25_ranked)}

    dense_docs = dense_run.get(qid, {})
    dense_ranked = sorted(dense_docs.items(), key=lambda x: x[1], reverse=True)
    dense_rank = {did: rank + 1 for rank, (did, _) in enumerate(dense_ranked)}

    all_docs = set(bm25_rank.keys()) | set(dense_rank.keys())
    fused_scores = {}
    for did in all_docs:
        score = 0.0
        if did in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[did])
        if did in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[did])
        fused_scores[did] = score

    fused_run[qid] = fused_scores

fused_metrics = evaluate_run(fused_run, qrels)
print(f"Fused (RRF): MAP@100={fused_metrics['map@100']:.4f}, nDCG@10={fused_metrics['ndcg@10']:.4f}, recall@100={fused_metrics['recall@100']:.4f}", flush=True)

# Save fused run
fused_run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}-fused.run"
write_trec_run(fused_run, fused_run_path, run_name=WORKTREE_NAME + "-fused")

# ===========================================================================
# STAGE 4: Rerank top-100 with Qwen3-Reranker-0.6B
# ===========================================================================

print(f"\nLoading {RERANKER_MODEL}...", flush=True)
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(
    RERANKER_MODEL,
    dtype=torch.float16,
).cuda().eval()

token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
print(f"  yes={token_true_id}, no={token_false_id}", flush=True)

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
max_content_length = min(MAX_CONTENT_TOKENS, 8192 - len(prefix_tokens) - len(suffix_tokens))

print(f"  Max content length: {max_content_length}", flush=True)

print(f"Reranking top-{RERANK_TOP_K} from fused list...", flush=True)
reranked_run = {}
t_rerank_start = time.time()
pairs_done = 0

sorted_qids = sorted(fused_run.keys(), key=lambda x: int(x) if x.isdigit() else x)
total_pairs = RERANK_TOP_K * len(sorted_qids)

for qi, qid in enumerate(sorted_qids):
    query_text = queries_dict.get(qid, "")
    fused_docs = sorted(fused_run[qid].items(), key=lambda x: x[1], reverse=True)[:RERANK_TOP_K]
    doc_ids_to_rerank = [did for did, _ in fused_docs]

    all_content_texts = []
    for did in doc_ids_to_rerank:
        doc = corpus.get(did, {"title": "", "text": ""})
        doc_text = (doc["title"] + " " + doc["text"]).strip()
        content = f"<Instruct>: {TASK_INSTRUCTION}\n\n<Query>: {query_text}\n\n<Document>: {doc_text}"
        all_content_texts.append(content)

    all_scores = []
    for batch_start in range(0, len(all_content_texts), RERANK_BATCH):
        batch_texts = all_content_texts[batch_start:batch_start + RERANK_BATCH]

        batch_input_ids = []
        for text in batch_texts:
            content_tokens = reranker_tokenizer.encode(text, add_special_tokens=False)
            if len(content_tokens) > max_content_length:
                content_tokens = content_tokens[:max_content_length]
            input_ids = prefix_tokens + content_tokens + suffix_tokens
            batch_input_ids.append(input_ids)

        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = reranker_tokenizer.pad_token_id if reranker_tokenizer.pad_token_id is not None else reranker_tokenizer.eos_token_id

        padded_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long).cuda()
        attention_mask_t = torch.tensor(attention_masks, dtype=torch.long).cuda()

        with torch.no_grad():
            logits = reranker_model(input_ids=input_ids_t, attention_mask=attention_mask_t).logits[:, -1, :]

        true_logits = logits[:, token_true_id]
        false_logits = logits[:, token_false_id]
        stacked = torch.stack([false_logits, true_logits], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().tolist()

        all_scores.extend(scores)

        del input_ids_t, attention_mask_t, logits
        torch.cuda.empty_cache()

    reranked_run[qid] = {}
    for did, score in zip(doc_ids_to_rerank, all_scores):
        reranked_run[qid][did] = score

    pairs_done += len(doc_ids_to_rerank)
    if (qi + 1) % 10 == 0 or qi == 0:
        elapsed = time.time() - t_rerank_start
        rate = pairs_done / elapsed if elapsed > 0 else 0
        eta = (total_pairs - pairs_done) / rate if rate > 0 else 0
        print(f"  [{qi+1}/{len(sorted_qids)}] {pairs_done}/{total_pairs} pairs, "
              f"{rate:.1f} pairs/s, ETA {eta:.0f}s", flush=True)

t_rerank_end = time.time()
rerank_dur = t_rerank_end - t_rerank_start
print(f"  Reranking done in {rerank_dur:.1f}s ({pairs_done/rerank_dur:.1f} pairs/s)", flush=True)

# ===========================================================================
# STAGE 5: Write run file and evaluate
# ===========================================================================

run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}.run"
write_trec_run(reranked_run, run_path, run_name=WORKTREE_NAME)
print(f"TREC run written: {run_path}", flush=True)

print("Evaluating...", flush=True)
metrics = evaluate_run(reranked_run, qrels)
eval_dur = time.time() - t_eval_start

# ===========================================================================
# Summary
# ===========================================================================

t_end = time.time()
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

print("---")
print(f"ndcg@10:          {metrics['ndcg@10']:.6f}")
print(f"map@1000:         {metrics['map@1000']:.6f}")
print(f"map@100:          {metrics['map@100']:.6f}")
print(f"recall@100:       {metrics['recall@100']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"num_steps:        {step}")
print(f"encoder_model:    {ENCODER_MODEL}")
print(f"num_docs_indexed: {len(doc_ids)}")
print(f"eval_duration:    {eval_dur:.3f}")
print(f"rerank_duration:  {rerank_dur:.3f}")
print(f"rerank_top_k:     {RERANK_TOP_K}")
print(f"method:           2-phase HN mining + dense(e5-base-v2) + BM25+Bo1 RRF >> Qwen3-Reranker top-{RERANK_TOP_K}")
print(f"phase1_steps:     {phase1_steps}")
print(f"phase2_steps:     {step - phase1_steps}")
print(f"robust04_triples: {len(robust04_triples)}")
print(f"dense_map100:     {dense_metrics['map@100']:.6f}")
print(f"dense_recall100:  {dense_metrics['recall@100']:.6f}")
print(f"fused_map100:     {fused_metrics['map@100']:.6f}")
print(f"fused_recall100:  {fused_metrics['recall@100']:.6f}")
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
