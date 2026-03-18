"""
exp33-iter-hn: Three-phase iterative hard negative mining.

Phase 1 (200s): Train e5-base-v2 on MS-MARCO with InfoNCE
Mine round 1: Encode Robust04, retrieve top-100, collect hard negatives
Phase 2 (200s): Mixed training (MS-MARCO + round-1 hard negatives)
Mine round 2: Re-encode with improved model, mine NEW hard negatives
Phase 3 (200s): Mixed training (MS-MARCO + round-2 hard negatives)
Evaluate: Encode, fuse with BM25 via RRF, evaluate (no reranker)
"""

import gc
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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
BATCH_SIZE = 128
LR = 1e-5
TEMPERATURE = 0.05
ENCODE_BATCH = 512

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Three-phase time budget
PHASE1_BUDGET = 200  # seconds
PHASE2_BUDGET = 200  # seconds
PHASE3_BUDGET = 200  # seconds
TOTAL_TIME_BUDGET = PHASE1_BUDGET + PHASE2_BUDGET + PHASE3_BUDGET

# Hard negative mining config
MINE_TOP_K = 100  # retrieve top-K per query for mining
HN_PER_QUERY = 10  # hard negatives to keep per query
ROBUST04_BATCH_RATIO = 0.3  # fraction of HN-phase batches from Robust04 hard negatives

BM25_RUN_FILE = "../../runs/exp22-bm25-prf/exp22-bm25-prf.run"
RRF_K = 60

WORKTREE_NAME = "exp33-iter-hn"

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


def tokenize(texts, max_len, tokenizer):
    return tokenizer(
        texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt"
    )


# ===========================================================================
# Helper: Mine hard negatives from Robust04 using current model
# ===========================================================================

def mine_hard_negatives(model, tokenizer, corpus, queries_dict, qrels, doc_ids, doc_texts, device, round_num):
    """Encode corpus, retrieve top-K, collect hard negatives."""
    import faiss
    faiss.omp_set_num_threads(1)  # prevent threading deadlock

    print(f"\n=== MINING HARD NEGATIVES (round {round_num}) ===", flush=True)
    t_mine_start = time.time()

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # Encode corpus
    print(f"Encoding {len(doc_ids):,} documents for mining round {round_num}...", flush=True)
    model.eval()
    all_doc_embs = []
    with torch.no_grad():
        for i in range(0, len(doc_texts), ENCODE_BATCH):
            batch = doc_texts[i : i + ENCODE_BATCH]
            enc = tokenize(batch, MAX_DOC_LEN, tokenizer)
            embs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_doc_embs.append(F.normalize(embs, dim=-1).cpu().float().numpy())
            if (i // ENCODE_BATCH) % 20 == 0:
                print(f"  {i:,}/{len(doc_texts):,}", end="\r", flush=True)

    doc_matrix = np.vstack(all_doc_embs)
    print(f"  Encoded {len(doc_matrix):,} docs, shape {doc_matrix.shape}", flush=True)

    # Build FAISS index
    dim = doc_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_matrix)
    print(f"FAISS index built: {index.ntotal:,} vectors", flush=True)

    # Encode queries and search
    query_ids = list(queries_dict.keys())
    query_texts = [QUERY_PREFIX + queries_dict[qid] for qid in query_ids]

    print(f"Encoding {len(query_ids)} queries...", flush=True)
    with torch.no_grad():
        q_enc = tokenize(query_texts, MAX_QUERY_LEN, tokenizer)
        q_embs = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
        q_matrix = F.normalize(q_embs, dim=-1).cpu().float().numpy()

    scores_matrix, indices_matrix = index.search(q_matrix, MINE_TOP_K)

    # Collect hard negatives
    hard_negatives = {}
    positive_docs = {}
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

    # Add all relevant docs from qrels as positives
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

    # Clean up
    del index, doc_matrix, q_matrix
    del scores_matrix, indices_matrix
    gc.collect()
    torch.cuda.empty_cache()

    t_mine_end = time.time()
    print(f"Mining round {round_num} done in {t_mine_end - t_mine_start:.1f}s", flush=True)

    return hard_negatives, positive_docs, query_ids


def build_robust04_triples(queries_dict, corpus, positive_docs, hard_negatives, query_ids):
    """Build training triples from mined hard negatives."""
    triples = []
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
                triples.append((q_text, pos_text, neg_text))
    random.shuffle(triples)
    return triples


def train_phase(model, optimizer, tokenizer, train_stream, robust04_triples, phase_budget,
                phase_name, step, total_training_time, smooth_loss, debiased,
                loss_curve, _next_curve_pct, device):
    """Run one training phase."""
    print(f"\n=== {phase_name}: Training for {phase_budget}s ===", flush=True)
    if robust04_triples is not None:
        print(f"  Using {len(robust04_triples)} Robust04 triples (ratio={ROBUST04_BATCH_RATIO})", flush=True)

    model.train()
    robust04_idx = 0
    phase_training_time = 0.0
    phase_start_step = step

    while True:
        t0 = time.time()

        # Decide if this batch is MS-MARCO or Robust04
        use_robust04 = (robust04_triples is not None and
                        random.random() < ROBUST04_BATCH_RATIO and
                        len(robust04_triples) > 0)

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

        q_enc = tokenize(queries_batch, MAX_QUERY_LEN, tokenizer)
        p_enc = tokenize(positives_batch, MAX_DOC_LEN, tokenizer)
        n_enc = tokenize(negatives_batch, MAX_DOC_LEN, tokenizer)

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
        if step > 2:
            phase_training_time += dt
            total_training_time += dt

        loss_val = loss.item()
        if math.isnan(loss_val):
            print(f"\nNaN loss at step {step}, aborting.", flush=True)
            raise RuntimeError("NaN loss")
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        remaining = max(0, phase_budget - phase_training_time)
        src = "R04" if use_robust04 else "MSM"
        print(f"\r{phase_name} step {step:05d} | loss: {debiased:.4f} | {src} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

        # Track loss curve across full training
        pct_done = int(100 * total_training_time / TOTAL_TIME_BUDGET)
        while _next_curve_pct <= pct_done and _next_curve_pct <= 100:
            loss_curve.append((_next_curve_pct, round(debiased, 4)))
            _next_curve_pct += 10

        step += 1
        if step > 2 and phase_training_time >= phase_budget:
            break

    print(flush=True)
    phase_steps = step - phase_start_step
    print(f"{phase_name} done: {phase_steps} steps, {phase_training_time:.1f}s, loss={debiased:.4f}", flush=True)

    return step, total_training_time, smooth_loss, debiased, loss_curve, _next_curve_pct


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
model = BiEncoder(ENCODER_MODEL).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_stream = stream_msmarco_triples()

total_training_time = 0.0
step = 0
smooth_loss = 0.0
loss_curve = []
_next_curve_pct = 0
debiased = 0.0

# --- Phase 1: MS-MARCO only ---
step, total_training_time, smooth_loss, debiased, loss_curve, _next_curve_pct = train_phase(
    model, optimizer, tokenizer, train_stream,
    robust04_triples=None,
    phase_budget=PHASE1_BUDGET,
    phase_name="P1",
    step=step,
    total_training_time=total_training_time,
    smooth_loss=smooth_loss,
    debiased=debiased,
    loss_curve=loss_curve,
    _next_curve_pct=_next_curve_pct,
    device=device,
)
phase1_steps = step

# --- Load Robust04 for mining ---
print("\nLoading Robust04...", flush=True)
corpus, queries_dict, qrels = load_robust04()
doc_ids = list(corpus.keys())
doc_texts = [
    PASSAGE_PREFIX + (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
    for d in doc_ids
]

# --- Mine round 1 ---
hard_negatives_r1, positive_docs_r1, query_ids = mine_hard_negatives(
    model, tokenizer, corpus, queries_dict, qrels, doc_ids, doc_texts, device, round_num=1
)
robust04_triples_r1 = build_robust04_triples(queries_dict, corpus, positive_docs_r1, hard_negatives_r1, query_ids)
print(f"Built {len(robust04_triples_r1)} Robust04 training triples (round 1)", flush=True)

# --- Phase 2: MS-MARCO + round-1 hard negatives ---
step, total_training_time, smooth_loss, debiased, loss_curve, _next_curve_pct = train_phase(
    model, optimizer, tokenizer, train_stream,
    robust04_triples=robust04_triples_r1,
    phase_budget=PHASE2_BUDGET,
    phase_name="P2",
    step=step,
    total_training_time=total_training_time,
    smooth_loss=smooth_loss,
    debiased=debiased,
    loss_curve=loss_curve,
    _next_curve_pct=_next_curve_pct,
    device=device,
)
phase2_steps = step - phase1_steps

# Free round-1 triples
del robust04_triples_r1
gc.collect()

# --- Mine round 2 (with improved model) ---
hard_negatives_r2, positive_docs_r2, query_ids = mine_hard_negatives(
    model, tokenizer, corpus, queries_dict, qrels, doc_ids, doc_texts, device, round_num=2
)
robust04_triples_r2 = build_robust04_triples(queries_dict, corpus, positive_docs_r2, hard_negatives_r2, query_ids)
print(f"Built {len(robust04_triples_r2)} Robust04 training triples (round 2)", flush=True)

# --- Phase 3: MS-MARCO + round-2 hard negatives ---
step, total_training_time, smooth_loss, debiased, loss_curve, _next_curve_pct = train_phase(
    model, optimizer, tokenizer, train_stream,
    robust04_triples=robust04_triples_r2,
    phase_budget=PHASE3_BUDGET,
    phase_name="P3",
    step=step,
    total_training_time=total_training_time,
    smooth_loss=smooth_loss,
    debiased=debiased,
    loss_curve=loss_curve,
    _next_curve_pct=_next_curve_pct,
    device=device,
)
phase3_steps = step - phase1_steps - phase2_steps

# Free training resources
del robust04_triples_r2, optimizer
gc.collect()
torch.cuda.empty_cache()

print(f"\nTotal training: {step} steps, {total_training_time:.1f}s", flush=True)

# ===========================================================================
# ENCODE + DENSE RETRIEVAL
# ===========================================================================

import faiss
faiss.omp_set_num_threads(1)  # prevent threading deadlock

t_eval_start = time.time()

print(f"\nEncoding {len(doc_ids):,} documents with phase-3 model...", flush=True)
model.eval()
all_doc_embs = []
with torch.no_grad():
    for i in range(0, len(doc_texts), ENCODE_BATCH):
        batch = doc_texts[i : i + ENCODE_BATCH]
        enc = tokenize(batch, MAX_DOC_LEN, tokenizer)
        embs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_doc_embs.append(F.normalize(embs, dim=-1).cpu().float().numpy())
        if (i // ENCODE_BATCH) % 20 == 0:
            print(f"  {i:,}/{len(doc_texts):,}", end="\r", flush=True)

doc_matrix = np.vstack(all_doc_embs)
del all_doc_embs
dim = doc_matrix.shape[1]
print(f"  Encoded {len(doc_matrix):,} docs, shape {doc_matrix.shape}", flush=True)

# Encode queries before freeing model
query_texts_eval = [QUERY_PREFIX + queries_dict[qid] for qid in query_ids]
print(f"Encoding {len(query_ids)} queries...", flush=True)
with torch.no_grad():
    q_enc = tokenize(query_texts_eval, MAX_QUERY_LEN, tokenizer)
    q_embs = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    q_matrix = F.normalize(q_embs, dim=-1).cpu().float().numpy()

# Free model to make room for GPU FAISS
del model
gc.collect()
torch.cuda.empty_cache()

# Build FAISS index (GPU now that model is freed)
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatIP(dim)
index_cpu.add(doc_matrix)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
print(f"FAISS index built: {index.ntotal:,} vectors", flush=True)

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
del index, index_cpu, res, doc_matrix, q_matrix
del scores_matrix, indices_matrix
gc.collect()
torch.cuda.empty_cache()

# ===========================================================================
# FUSE WITH BM25 VIA RRF
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

# Save fused run (this is the main run file)
run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}.run"
write_trec_run(fused_run, run_path, run_name=WORKTREE_NAME)
print(f"TREC run written: {run_path}", flush=True)

# Use fused metrics as the final metrics
metrics = fused_metrics
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
print(f"method:           3-phase iterative HN mining + dense(e5-base-v2) + BM25+Bo1 RRF")
print(f"phase1_steps:     {phase1_steps}")
print(f"phase2_steps:     {phase2_steps}")
print(f"phase3_steps:     {phase3_steps}")
print(f"dense_map100:     {dense_metrics['map@100']:.6f}")
print(f"dense_ndcg10:     {dense_metrics['ndcg@10']:.6f}")
print(f"dense_recall100:  {dense_metrics['recall@100']:.6f}")
print(f"fused_map100:     {fused_metrics['map@100']:.6f}")
print(f"fused_ndcg10:     {fused_metrics['ndcg@10']:.6f}")
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
