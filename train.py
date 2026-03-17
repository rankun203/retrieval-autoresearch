"""
exp23-qwen3-reranker: Rerank BM25+Bo1 top-K results using Qwen3-Reranker-0.6B.

Uses the Qwen3-Reranker-0.6B model (LLM-based reranker) with yes/no token logits.
Reads existing BM25+Bo1 TREC run file, reranks top-K per query, evaluates.
"""

import os
import sys
import time
import torch

sys.path.insert(0, ".")
from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RERANK_TOP_K = 100  # rerank top-K from BM25+Bo1
BATCH_SIZE = 4      # batch size for reranking (docs are long)
MAX_CONTENT_TOKENS = 512  # max tokens for the content portion (query+doc)
BM25_RUN_FILE = "../../runs/exp22-bm25-prf/exp22-bm25-prf.run"
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
WORKTREE_NAME = "exp23-qwen3-reranker"
TASK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

t_start = time.time()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Robust04...", flush=True)
corpus, queries, qrels = load_robust04()
print(f"  {len(corpus):,} docs, {len(queries)} queries", flush=True)

# ---------------------------------------------------------------------------
# Load BM25+Bo1 run file
# ---------------------------------------------------------------------------
print(f"Loading BM25+Bo1 run file: {BM25_RUN_FILE}", flush=True)
bm25_run = {}  # {qid: {doc_id: score}}
with open(BM25_RUN_FILE) as f:
    for line in f:
        parts = line.strip().split()
        qid, _, docid, rank, score, _ = parts
        if qid not in bm25_run:
            bm25_run[qid] = {}
        bm25_run[qid][docid] = float(score)

# Keep only top-K per query
for qid in bm25_run:
    sorted_docs = sorted(bm25_run[qid].items(), key=lambda x: x[1], reverse=True)[:RERANK_TOP_K]
    bm25_run[qid] = dict(sorted_docs)

total_pairs = sum(len(docs) for docs in bm25_run.values())
print(f"  {len(bm25_run)} queries, {total_pairs} query-doc pairs to rerank", flush=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_NAME}...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).cuda().eval()

token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")
print(f"  yes token id: {token_true_id}, no token id: {token_false_id}", flush=True)

# Build prefix/suffix for the chat template
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# Max length for the content portion (between prefix and suffix)
max_content_length = min(MAX_CONTENT_TOKENS, 8192 - len(prefix_tokens) - len(suffix_tokens))

print(f"  Model loaded. Prefix tokens: {len(prefix_tokens)}, suffix tokens: {len(suffix_tokens)}", flush=True)
print(f"  Max content length: {max_content_length}", flush=True)

# Check VRAM
vram_after_load = torch.cuda.max_memory_allocated() / 1024**2
print(f"  VRAM after model load: {vram_after_load:.0f} MB", flush=True)

# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
print(f"Reranking with batch_size={BATCH_SIZE}...", flush=True)

reranked_run = {}
t_rerank_start = time.time()
pairs_done = 0

sorted_qids = sorted(bm25_run.keys(), key=lambda x: int(x) if x.isdigit() else x)

for qi, qid in enumerate(sorted_qids):
    query_text = queries.get(qid, "")
    doc_scores = bm25_run[qid]
    doc_ids = list(doc_scores.keys())

    # Build all inputs for this query
    all_content_texts = []
    for did in doc_ids:
        doc = corpus.get(did, {"title": "", "text": ""})
        doc_text = (doc["title"] + " " + doc["text"]).strip()
        content = f"<Instruct>: {TASK_INSTRUCTION}\n\n<Query>: {query_text}\n\n<Document>: {doc_text}"
        all_content_texts.append(content)

    # Process in batches
    all_scores = []
    for batch_start in range(0, len(all_content_texts), BATCH_SIZE):
        batch_texts = all_content_texts[batch_start:batch_start + BATCH_SIZE]

        # Tokenize each text, truncate content, add prefix/suffix
        batch_input_ids = []
        for text in batch_texts:
            content_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(content_tokens) > max_content_length:
                content_tokens = content_tokens[:max_content_length]
            input_ids = prefix_tokens + content_tokens + suffix_tokens
            batch_input_ids.append(input_ids)

        # Pad to same length (left padding)
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        padded_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long).cuda()
        attention_mask_t = torch.tensor(attention_masks, dtype=torch.long).cuda()

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attention_mask_t).logits[:, -1, :]

        true_logits = logits[:, token_true_id]
        false_logits = logits[:, token_false_id]
        stacked = torch.stack([false_logits, true_logits], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().tolist()  # P(yes)

        all_scores.extend(scores)

        del input_ids_t, attention_mask_t, logits
        torch.cuda.empty_cache()

    # Build reranked run for this query
    reranked_run[qid] = {}
    for did, score in zip(doc_ids, all_scores):
        reranked_run[qid][did] = score

    pairs_done += len(doc_ids)
    if (qi + 1) % 10 == 0 or qi == 0:
        elapsed = time.time() - t_rerank_start
        rate = pairs_done / elapsed if elapsed > 0 else 0
        eta = (total_pairs - pairs_done) / rate if rate > 0 else 0
        print(f"  [{qi+1}/{len(sorted_qids)}] {pairs_done}/{total_pairs} pairs, "
              f"{rate:.1f} pairs/s, ETA {eta:.0f}s", flush=True)

t_rerank_end = time.time()
rerank_dur = t_rerank_end - t_rerank_start
print(f"  Reranking done in {rerank_dur:.1f}s ({total_pairs/rerank_dur:.1f} pairs/s)", flush=True)

# ---------------------------------------------------------------------------
# Write TREC run file
# ---------------------------------------------------------------------------
run_path = f"runs/{WORKTREE_NAME}/{WORKTREE_NAME}.run"
write_trec_run(reranked_run, run_path, run_name=WORKTREE_NAME)
print(f"TREC run written: {run_path}", flush=True)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
print("Evaluating...", flush=True)
t_eval_start = time.time()
metrics = evaluate_run(reranked_run, qrels)
eval_dur = time.time() - t_eval_start

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_end = time.time()
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

print("---")
print(f"ndcg@10:          {metrics['ndcg@10']:.6f}")
print(f"map@1000:         {metrics['map@1000']:.6f}")
print(f"map@100:          {metrics['map@100']:.6f}")
print(f"recall@100:       {metrics['recall@100']:.6f}")
print(f"run_file:         {run_path}")
print(f"training_seconds: 0.0")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"num_steps:        0")
print(f"encoder_model:    Qwen3-Reranker-0.6B")
print(f"num_docs_indexed: {total_pairs}")
print(f"eval_duration:    {eval_dur:.3f}")
print(f"rerank_duration:  {rerank_dur:.3f}")
print(f"rerank_top_k:     {RERANK_TOP_K}")
print(f"method:           BM25+Bo1 >> Qwen3-Reranker-0.6B top-{RERANK_TOP_K}")
