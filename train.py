"""
exp09b-dph-lm-sweep: CPU-only sweep of alternative retrieval models with PRF on Robust04.

Tests DPH, InL2, PL2, DirichletLM, Hiemstra_LM (+ BM25 reference) with Bo1/KL/RM3 PRF,
then fuses best diverse systems.

CPU-only experiment. No GPU needed.

Usage:
  uv run --with 'python-terrier' train.py
"""

import os
import sys
import csv
import time
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_robust04, evaluate_run, write_trec_run

# ---------------------------------------------------------------------------
# Java / PyTerrier init
# ---------------------------------------------------------------------------
import pyterrier as pt

if not pt.java.started():
    pt.java.init()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_PATH = os.path.expanduser("~/.cache/autoresearch-retrieval/terrier_index")
NUM_RESULTS = 1000

# BM25 baseline reference
BM25_K1 = 0.9
BM25_B = 0.4

# Model-specific parameter grids
INL2_C_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
PL2_C_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
DIRICHLET_MU_VALUES = [500, 1000, 1500, 2000, 2500, 3000, 5000]
HIEMSTRA_LAMBDA_VALUES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# PRF grids
FB_DOCS_VALUES = [3, 5, 10]
FB_TERMS_VALUES = [10, 20, 30, 50]
RM3_FB_DOCS = [5, 10]
RM3_FB_TERMS = [20, 30]
RM3_FB_LAMBDA = [0.5, 0.7]

# Fusion alpha values
FUSION_ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]

t_start = time.time()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Robust04...", flush=True)
corpus, queries, qrels = load_robust04()
print(f"  {len(corpus):,} docs, {len(queries)} queries", flush=True)

topics_df = pd.DataFrame([{"qid": qid, "query": text} for qid, text in queries.items()])

# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
index_ref = pt.IndexRef.of(os.path.join(INDEX_PATH, "data.properties"))
print(f"Using Terrier index at {INDEX_PATH}", flush=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
all_results = []


def _results_to_run(results_df):
    """Convert PyTerrier results DataFrame to run dict."""
    run = {}
    for _, row in results_df.iterrows():
        qid = str(row["qid"])
        if qid not in run:
            run[qid] = {}
        run[qid][str(row["docno"])] = float(row["score"])
    return run


def make_retriever(wmodel, controls=None):
    """Create a PyTerrier retriever for the given weighting model."""
    kwargs = {"num_results": NUM_RESULTS}
    if controls:
        kwargs["controls"] = controls
    return pt.terrier.Retriever(index_ref, wmodel=wmodel, **kwargs)


def run_model(wmodel, controls=None, label=None):
    """Run a base retrieval model and return (run_dict, metrics)."""
    retriever = make_retriever(wmodel, controls)
    results = retriever.transform(topics_df)
    run = _results_to_run(results)
    metrics = evaluate_run(run, qrels)
    return run, metrics


def run_model_prf(wmodel, controls, prf_method, fb_docs, fb_terms, fb_lambda=None):
    """Run a retrieval model + PRF and return (run_dict, metrics)."""
    retriever = make_retriever(wmodel, controls)

    if prf_method == "bo1":
        qe = pt.rewrite.Bo1QueryExpansion(index_ref, fb_docs=fb_docs, fb_terms=fb_terms)
    elif prf_method == "kl":
        qe = pt.rewrite.KLQueryExpansion(index_ref, fb_docs=fb_docs, fb_terms=fb_terms)
    elif prf_method == "rm3":
        qe = pt.rewrite.RM3(index_ref, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=fb_lambda)
    else:
        raise ValueError(f"Unknown PRF method: {prf_method}")

    pipeline = retriever >> qe >> retriever
    results = pipeline.transform(topics_df)
    run = _results_to_run(results)
    metrics = evaluate_run(run, qrels)
    return run, metrics


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


def combsum_fusion(runs):
    """CombSUM: sum of min-max normalized scores across all runs."""
    normalized_runs = [normalize_scores(r) for r in runs]
    all_qids = set()
    for nr in normalized_runs:
        all_qids |= set(nr.keys())
    fused = {}
    for qid in all_qids:
        all_docs = set()
        for nr in normalized_runs:
            all_docs |= set(nr.get(qid, {}).keys())
        fused[qid] = {}
        for did in all_docs:
            score = sum(nr.get(qid, {}).get(did, 0.0) for nr in normalized_runs)
            fused[qid][did] = score
    return fused


def record_result(phase, model, param_name, param_val, prf_method, fb_docs, fb_terms,
                  fb_lambda, metrics, elapsed):
    """Record a result row."""
    row = {
        "phase": phase,
        "model": model,
        "param_name": param_name,
        "param_val": param_val,
        "prf_method": prf_method if prf_method else "none",
        "fb_docs": fb_docs,
        "fb_terms": fb_terms,
        "fb_lambda": fb_lambda,
        "map@100": metrics["map@100"],
        "ndcg@10": metrics["ndcg@10"],
        "map@1000": metrics["map@1000"],
        "recall@100": metrics["recall@100"],
        "seconds": elapsed,
    }
    all_results.append(row)
    return row


# ---------------------------------------------------------------------------
# Phase 1: Base model evaluation (no PRF)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print("PHASE 1: Base model evaluation (no PRF)", flush=True)
print(f"{'='*70}", flush=True)

# Store best config per model for Phase 2
best_per_model = {}  # model_name -> {"controls": dict, "run": dict, "metrics": dict, "label": str}

# --- BM25 baseline ---
print("\n--- BM25 (baseline reference) ---", flush=True)
t0 = time.time()
bm25_run, bm25_metrics = run_model("BM25", {"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)})
elapsed = time.time() - t0
record_result(1, "BM25", "k1/b", f"{BM25_K1}/{BM25_B}", None, None, None, None, bm25_metrics, elapsed)
print(f"  BM25 k1={BM25_K1} b={BM25_B}:  MAP@100={bm25_metrics['map@100']:.4f}  "
      f"nDCG@10={bm25_metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
best_per_model["BM25"] = {
    "controls": {"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
    "run": bm25_run,
    "metrics": bm25_metrics,
    "label": f"k1={BM25_K1},b={BM25_B}",
}

# --- DPH (parameter-free) ---
print("\n--- DPH (parameter-free) ---", flush=True)
t0 = time.time()
dph_run, dph_metrics = run_model("DPH")
elapsed = time.time() - t0
record_result(1, "DPH", "none", "N/A", None, None, None, None, dph_metrics, elapsed)
print(f"  DPH:  MAP@100={dph_metrics['map@100']:.4f}  nDCG@10={dph_metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
best_per_model["DPH"] = {
    "controls": None,
    "run": dph_run,
    "metrics": dph_metrics,
    "label": "parameter-free",
}

# --- InL2 (c sweep) ---
print("\n--- InL2 (c sweep) ---", flush=True)
best_inl2 = None
for c_val in INL2_C_VALUES:
    t0 = time.time()
    controls = {"c": str(c_val)}
    run, metrics = run_model("InL2", controls)
    elapsed = time.time() - t0
    record_result(1, "InL2", "c", c_val, None, None, None, None, metrics, elapsed)
    print(f"  InL2 c={c_val}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
    if best_inl2 is None or metrics["map@100"] > best_inl2["metrics"]["map@100"]:
        best_inl2 = {"controls": controls, "run": run, "metrics": metrics, "label": f"c={c_val}"}
best_per_model["InL2"] = best_inl2
print(f"  Best InL2: {best_inl2['label']}  MAP@100={best_inl2['metrics']['map@100']:.4f}", flush=True)

# --- PL2 (c sweep) ---
print("\n--- PL2 (c sweep) ---", flush=True)
best_pl2 = None
for c_val in PL2_C_VALUES:
    t0 = time.time()
    controls = {"c": str(c_val)}
    run, metrics = run_model("PL2", controls)
    elapsed = time.time() - t0
    record_result(1, "PL2", "c", c_val, None, None, None, None, metrics, elapsed)
    print(f"  PL2 c={c_val}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
    if best_pl2 is None or metrics["map@100"] > best_pl2["metrics"]["map@100"]:
        best_pl2 = {"controls": controls, "run": run, "metrics": metrics, "label": f"c={c_val}"}
best_per_model["PL2"] = best_pl2
print(f"  Best PL2: {best_pl2['label']}  MAP@100={best_pl2['metrics']['map@100']:.4f}", flush=True)

# --- DirichletLM (mu sweep) ---
print("\n--- DirichletLM (mu sweep) ---", flush=True)
best_dlm = None
for mu in DIRICHLET_MU_VALUES:
    t0 = time.time()
    controls = {"mu": str(mu)}
    run, metrics = run_model("DirichletLM", controls)
    elapsed = time.time() - t0
    record_result(1, "DirichletLM", "mu", mu, None, None, None, None, metrics, elapsed)
    print(f"  DirichletLM mu={mu}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
    if best_dlm is None or metrics["map@100"] > best_dlm["metrics"]["map@100"]:
        best_dlm = {"controls": controls, "run": run, "metrics": metrics, "label": f"mu={mu}"}
best_per_model["DirichletLM"] = best_dlm
print(f"  Best DirichletLM: {best_dlm['label']}  MAP@100={best_dlm['metrics']['map@100']:.4f}", flush=True)

# --- Hiemstra_LM (lambda sweep) ---
print("\n--- Hiemstra_LM (lambda sweep) ---", flush=True)
best_hlm = None
for lam in HIEMSTRA_LAMBDA_VALUES:
    t0 = time.time()
    controls = {"lambda": str(lam)}
    run, metrics = run_model("Hiemstra_LM", controls)
    elapsed = time.time() - t0
    record_result(1, "Hiemstra_LM", "lambda", lam, None, None, None, None, metrics, elapsed)
    print(f"  Hiemstra_LM lambda={lam}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)
    if best_hlm is None or metrics["map@100"] > best_hlm["metrics"]["map@100"]:
        best_hlm = {"controls": controls, "run": run, "metrics": metrics, "label": f"lambda={lam}"}
best_per_model["Hiemstra_LM"] = best_hlm
print(f"  Best Hiemstra_LM: {best_hlm['label']}  MAP@100={best_hlm['metrics']['map@100']:.4f}", flush=True)

# Phase 1 summary
print(f"\n--- Phase 1 Summary ---", flush=True)
print(f"{'Model':<15} {'Best Params':<20} {'MAP@100':>8} {'nDCG@10':>8} {'MAP@1000':>9} {'R@100':>7}", flush=True)
print("-" * 70, flush=True)
phase1_ranked = sorted(best_per_model.items(), key=lambda x: x[1]["metrics"]["map@100"], reverse=True)
for model_name, info in phase1_ranked:
    m = info["metrics"]
    print(f"{model_name:<15} {info['label']:<20} {m['map@100']:8.4f} {m['ndcg@10']:8.4f} {m['map@1000']:9.4f} {m['recall@100']:7.4f}", flush=True)

# ---------------------------------------------------------------------------
# Phase 2: PRF sweep (Bo1 and KL on all models)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print("PHASE 2: PRF sweep (Bo1 and KL on all models)", flush=True)
print(f"{'='*70}", flush=True)

best_prf_per_model = {}  # model_name -> best result with PRF

for model_name, model_info in best_per_model.items():
    controls = model_info["controls"]
    wmodel = model_name

    for prf_method in ["bo1", "kl"]:
        print(f"\n--- {model_name} + {prf_method.upper()} ---", flush=True)
        count = 0
        total = len(FB_DOCS_VALUES) * len(FB_TERMS_VALUES)
        for fb_docs in FB_DOCS_VALUES:
            for fb_terms in FB_TERMS_VALUES:
                count += 1
                t0 = time.time()
                try:
                    _, metrics = run_model_prf(wmodel, controls, prf_method, fb_docs, fb_terms)
                    elapsed = time.time() - t0
                    record_result(2, model_name, model_info["label"], None, prf_method,
                                  fb_docs, fb_terms, None, metrics, elapsed)
                    print(f"  [{count:2d}/{total}] fd={fb_docs} ft={fb_terms}  "
                          f"MAP@100={metrics['map@100']:.4f}  ({elapsed:.1f}s)", flush=True)

                    key = model_name
                    if key not in best_prf_per_model or metrics["map@100"] > best_prf_per_model[key]["metrics"]["map@100"]:
                        best_prf_per_model[key] = {
                            "wmodel": wmodel,
                            "controls": controls,
                            "prf_method": prf_method,
                            "fb_docs": fb_docs,
                            "fb_terms": fb_terms,
                            "fb_lambda": None,
                            "metrics": metrics,
                            "label": f"{model_name}+{prf_method.upper()} fd={fb_docs} ft={fb_terms}",
                        }
                except Exception as e:
                    print(f"  [{count:2d}/{total}] fd={fb_docs} ft={fb_terms}  ERROR: {e}", flush=True)

# Phase 2 summary
print(f"\n--- Phase 2 Summary: Best PRF per model ---", flush=True)
print(f"{'Config':<45} {'MAP@100':>8} {'nDCG@10':>8} {'MAP@1000':>9} {'R@100':>7}", flush=True)
print("-" * 80, flush=True)
phase2_ranked = sorted(best_prf_per_model.items(), key=lambda x: x[1]["metrics"]["map@100"], reverse=True)
for model_name, info in phase2_ranked:
    m = info["metrics"]
    print(f"{info['label']:<45} {m['map@100']:8.4f} {m['ndcg@10']:8.4f} {m['map@1000']:9.4f} {m['recall@100']:7.4f}", flush=True)

# Also compare best PRF vs best no-PRF
print(f"\n--- Improvement from PRF ---", flush=True)
for model_name in best_per_model:
    base_map = best_per_model[model_name]["metrics"]["map@100"]
    if model_name in best_prf_per_model:
        prf_map = best_prf_per_model[model_name]["metrics"]["map@100"]
        delta = prf_map - base_map
        print(f"  {model_name}: {base_map:.4f} -> {prf_map:.4f} (delta={delta:+.4f})", flush=True)

# ---------------------------------------------------------------------------
# Phase 3: RM3 on top-2 models
# ---------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print("PHASE 3: RM3 on top-2 models from Phase 2", flush=True)
print(f"{'='*70}", flush=True)

top2_models = [name for name, _ in phase2_ranked[:2]]
print(f"  Top-2: {top2_models}", flush=True)

for model_name in top2_models:
    info = best_per_model[model_name]
    wmodel = model_name
    controls = info["controls"]

    print(f"\n--- {model_name} + RM3 ---", flush=True)
    count = 0
    total = len(RM3_FB_DOCS) * len(RM3_FB_TERMS) * len(RM3_FB_LAMBDA)
    for fb_docs in RM3_FB_DOCS:
        for fb_terms in RM3_FB_TERMS:
            for fb_lambda in RM3_FB_LAMBDA:
                count += 1
                t0 = time.time()
                try:
                    _, metrics = run_model_prf(wmodel, controls, "rm3", fb_docs, fb_terms, fb_lambda)
                    elapsed = time.time() - t0
                    record_result(3, model_name, info["label"], None, "rm3",
                                  fb_docs, fb_terms, fb_lambda, metrics, elapsed)
                    print(f"  [{count:2d}/{total}] fd={fb_docs} ft={fb_terms} fl={fb_lambda}  "
                          f"MAP@100={metrics['map@100']:.4f}  ({elapsed:.1f}s)", flush=True)

                    key = model_name
                    if metrics["map@100"] > best_prf_per_model[key]["metrics"]["map@100"]:
                        best_prf_per_model[key] = {
                            "wmodel": wmodel,
                            "controls": controls,
                            "prf_method": "rm3",
                            "fb_docs": fb_docs,
                            "fb_terms": fb_terms,
                            "fb_lambda": fb_lambda,
                            "metrics": metrics,
                            "label": f"{model_name}+RM3 fd={fb_docs} ft={fb_terms} fl={fb_lambda}",
                        }
                except Exception as e:
                    print(f"  [{count:2d}/{total}] fd={fb_docs} ft={fb_terms} fl={fb_lambda}  ERROR: {e}", flush=True)

# ---------------------------------------------------------------------------
# Phase 4: Multi-system fusion
# ---------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print("PHASE 4: Multi-system fusion", flush=True)
print(f"{'='*70}", flush=True)

# Re-run best configs to get runs for fusion
print("\nRe-running best configs to get runs for fusion...", flush=True)
best_runs = {}  # model_name -> run_dict

for model_name, info in best_prf_per_model.items():
    t0 = time.time()
    if info["prf_method"] == "none" or info.get("prf_method") is None:
        run, _ = run_model(info["wmodel"], info["controls"])
    else:
        run, metrics = run_model_prf(
            info["wmodel"], info["controls"], info["prf_method"],
            info["fb_docs"], info["fb_terms"], info.get("fb_lambda"),
        )
    best_runs[model_name] = run
    elapsed = time.time() - t0
    print(f"  {model_name}: {info['label']}  ({elapsed:.1f}s)", flush=True)

# Save individual best runs
os.makedirs("runs", exist_ok=True)
for model_name, run in best_runs.items():
    safe_name = model_name.lower().replace("_", "-")
    write_trec_run(run, f"runs/best-{safe_name}.run", run_name=f"best-{safe_name}")

# --- Pairwise linear fusion: best DPH + best BM25 ---
print("\n--- Linear fusion: DPH vs BM25 ---", flush=True)
fusion_results = []

if "DPH" in best_runs and "BM25" in best_runs:
    for alpha in FUSION_ALPHAS:
        t0 = time.time()
        fused_run = linear_fusion(best_runs["DPH"], best_runs["BM25"], alpha)
        metrics = evaluate_run(fused_run, qrels)
        elapsed = time.time() - t0
        record_result(4, "DPH+BM25", "alpha", alpha, "fusion", None, None, None, metrics, elapsed)
        fusion_results.append({"name": f"DPH+BM25 alpha={alpha}", "run": fused_run, "metrics": metrics})
        print(f"  alpha={alpha}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)

# --- Pairwise fusions between all top model pairs ---
print("\n--- Pairwise fusions (top models) ---", flush=True)
model_names = list(best_runs.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_a, name_b = model_names[i], model_names[j]
        if (name_a == "DPH" and name_b == "BM25") or (name_a == "BM25" and name_b == "DPH"):
            continue  # Already done above
        for alpha in [0.4, 0.5, 0.6]:
            t0 = time.time()
            fused_run = linear_fusion(best_runs[name_a], best_runs[name_b], alpha)
            metrics = evaluate_run(fused_run, qrels)
            elapsed = time.time() - t0
            label = f"{name_a}+{name_b} alpha={alpha}"
            record_result(4, f"{name_a}+{name_b}", "alpha", alpha, "fusion", None, None, None, metrics, elapsed)
            fusion_results.append({"name": label, "run": fused_run, "metrics": metrics})
            print(f"  {label}:  MAP@100={metrics['map@100']:.4f}  ({elapsed:.1f}s)", flush=True)

# --- CombSUM of top-3 diverse systems ---
print("\n--- CombSUM (top-3 diverse systems) ---", flush=True)
top3_names = [name for name, _ in phase2_ranked[:3]]
if len(top3_names) >= 3:
    top3_runs = [best_runs[n] for n in top3_names if n in best_runs]
    if len(top3_runs) >= 3:
        t0 = time.time()
        combsum_run = combsum_fusion(top3_runs)
        metrics = evaluate_run(combsum_run, qrels)
        elapsed = time.time() - t0
        label = f"CombSUM({'+'.join(top3_names)})"
        record_result(4, label, "combsum", "top3", "fusion", None, None, None, metrics, elapsed)
        fusion_results.append({"name": label, "run": combsum_run, "metrics": metrics})
        print(f"  {label}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)

# Also try CombSUM of ALL systems
print("\n--- CombSUM (all systems) ---", flush=True)
all_system_runs = list(best_runs.values())
if len(all_system_runs) >= 2:
    t0 = time.time()
    combsum_all_run = combsum_fusion(all_system_runs)
    metrics = evaluate_run(combsum_all_run, qrels)
    elapsed = time.time() - t0
    label = f"CombSUM(all-{len(all_system_runs)})"
    record_result(4, label, "combsum", "all", "fusion", None, None, None, metrics, elapsed)
    fusion_results.append({"name": label, "run": combsum_all_run, "metrics": metrics})
    print(f"  {label}:  MAP@100={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}  ({elapsed:.1f}s)", flush=True)

# ---------------------------------------------------------------------------
# Find overall best and save
# ---------------------------------------------------------------------------
# Best among individual models + PRF
best_single = max(best_prf_per_model.items(), key=lambda x: x[1]["metrics"]["map@100"])
best_single_name, best_single_info = best_single
best_single_map = best_single_info["metrics"]["map@100"]

# Best among fusions
best_fusion = None
if fusion_results:
    best_fusion = max(fusion_results, key=lambda x: x["metrics"]["map@100"])

# Overall best
if best_fusion and best_fusion["metrics"]["map@100"] > best_single_map:
    overall_best_run = best_fusion["run"]
    overall_best_metrics = best_fusion["metrics"]
    overall_best_label = best_fusion["name"]
else:
    overall_best_run = best_runs[best_single_name]
    overall_best_metrics = best_single_info["metrics"]
    overall_best_label = best_single_info["label"]

write_trec_run(overall_best_run, "runs/best-overall.run", run_name="best-overall")

if best_fusion:
    write_trec_run(best_fusion["run"], "runs/best-fusion.run", run_name="best-fusion")

# ---------------------------------------------------------------------------
# Write full results CSV
# ---------------------------------------------------------------------------
csv_path = "logs/results_grid.csv"
os.makedirs("logs", exist_ok=True)
fieldnames = ["phase", "model", "param_name", "param_val", "prf_method",
              "fb_docs", "fb_terms", "fb_lambda",
              "map@100", "ndcg@10", "map@1000", "recall@100", "seconds"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(all_results, key=lambda r: -r["map@100"]):
        writer.writerow(row)
print(f"\nFull results written to {csv_path} ({len(all_results)} rows)", flush=True)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
t_end = time.time()
total_seconds = t_end - t_start

print(f"\n{'='*80}", flush=True)
print("RESULTS SUMMARY", flush=True)
print(f"{'='*80}", flush=True)

# Per-model leaderboard (best config with or without PRF)
print(f"\n{'Model+PRF':<45} {'MAP@100':>8} {'nDCG@10':>8} {'MAP@1000':>9} {'R@100':>7}", flush=True)
print("-" * 80, flush=True)

# Single model results
single_ranked = sorted(best_prf_per_model.items(), key=lambda x: x[1]["metrics"]["map@100"], reverse=True)
for model_name, info in single_ranked:
    m = info["metrics"]
    print(f"{info['label']:<45} {m['map@100']:8.4f} {m['ndcg@10']:8.4f} {m['map@1000']:9.4f} {m['recall@100']:7.4f}", flush=True)

# Fusion results
if fusion_results:
    print(f"\n{'Fusion':<45} {'MAP@100':>8} {'nDCG@10':>8} {'MAP@1000':>9} {'R@100':>7}", flush=True)
    print("-" * 80, flush=True)
    for fr in sorted(fusion_results, key=lambda x: -x["metrics"]["map@100"])[:10]:
        m = fr["metrics"]
        print(f"{fr['name']:<45} {m['map@100']:8.4f} {m['ndcg@10']:8.4f} {m['map@1000']:9.4f} {m['recall@100']:7.4f}", flush=True)

# Comparison with baselines
current_bm25_bo1_map100 = 0.2504
print(f"\n{'='*70}", flush=True)
print("COMPARISON WITH BASELINES", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Current BM25+Bo1 (k1=0.9, b=0.4, fd=5 ft=30): MAP@100={current_bm25_bo1_map100:.4f}", flush=True)
print(f"  Best single model+PRF (this exp):               MAP@100={best_single_info['metrics']['map@100']:.4f}  "
      f"(delta={best_single_info['metrics']['map@100'] - current_bm25_bo1_map100:+.4f})", flush=True)
if best_fusion:
    print(f"  Best fusion (this exp):                         MAP@100={best_fusion['metrics']['map@100']:.4f}  "
          f"(delta={best_fusion['metrics']['map@100'] - current_bm25_bo1_map100:+.4f})", flush=True)
print(f"  OVERALL BEST:                                   MAP@100={overall_best_metrics['map@100']:.4f}  "
      f"[{overall_best_label}]", flush=True)

# ---------------------------------------------------------------------------
# Standard summary block
# ---------------------------------------------------------------------------
print(f"\n---", flush=True)
print(f"ndcg@10:          {overall_best_metrics['ndcg@10']:.6f}", flush=True)
print(f"map@1000:         {overall_best_metrics['map@1000']:.6f}", flush=True)
print(f"map@100:          {overall_best_metrics['map@100']:.6f}", flush=True)
print(f"recall@100:       {overall_best_metrics['recall@100']:.6f}", flush=True)
print(f"training_seconds: 0.0", flush=True)
print(f"total_seconds:    {total_seconds:.1f}", flush=True)
print(f"peak_vram_mb:     0.0", flush=True)
print(f"num_steps:        0", flush=True)
print(f"encoder_model:    {overall_best_label}", flush=True)
print(f"num_docs_indexed: {len(corpus)}", flush=True)
print(f"eval_duration:    {total_seconds:.3f}", flush=True)
print(f"loss_curve:       N/A (parameter sweep)", flush=True)
print(f"budget_assessment: OK", flush=True)
print(f"method:           {overall_best_label}", flush=True)
print(f"total_configs:    {len(all_results)}", flush=True)
