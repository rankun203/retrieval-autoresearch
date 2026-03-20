"""
Build stable cache keys for experiment artifacts (embeddings, indexes, etc.).

Usage:
    from build_cache_key import get_cache_path, save_cache_metadata

    cache_params = dict(model="Qwen/Qwen3-Embedding-8B", max_length=512, pooling="last_token", dataset="robust04")
    cache_dir = get_cache_path("embeddings", **cache_params)
    embeddings_path = cache_dir / "doc_embeddings.npy"

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {cache_dir}", flush=True)
        doc_embeddings = np.load(embeddings_path)
    else:
        doc_embeddings = encode_all_docs(...)
        np.save(embeddings_path, doc_embeddings)
        save_cache_metadata(cache_dir, cache_type="embeddings", **cache_params)

Cache types:
    - embeddings: document/query embedding arrays
    - index: FAISS or other search indexes
    - colbert_index: ColBERT PLAID indexes
    - terrier_index: PyTerrier inverted indexes
    - bm25_run: BM25 retrieval results
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
METADATA_FILE = "metadata.json"


def _sanitize(value: str) -> str:
    """Sanitize a value for use in a file path. Replace / with _, remove unsafe chars."""
    s = str(value)
    s = s.replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]", "", s)
    return s


def build_cache_key(cache_type: str, *, model: str, **params) -> str:
    """Build a deterministic cache key from cache type, model name, and parameters.

    Args:
        cache_type: Type of cached artifact (e.g., "embeddings", "index", "colbert_index")
        model: Model name/path (e.g., "Qwen/Qwen3-Embedding-8B")
        **params: All parameters that affect the cached output (e.g., max_length=512, dataset="robust04")

    Returns:
        A clean, deterministic string like:
        "embeddings_Qwen_Qwen3-Embedding-8B_dataset-robust04_max_length-512"
    """
    parts = [_sanitize(cache_type), _sanitize(model)]
    for key in sorted(params.keys()):
        val = params[key]
        parts.append(f"{_sanitize(key)}-{_sanitize(str(val))}")
    return "_".join(parts)


def get_cache_path(cache_type: str, *, model: str, **params) -> Path:
    """Get the cache directory path for a given cache type and parameters.

    Creates the directory if it doesn't exist.

    Returns:
        Path to the cache directory, e.g.:
        .cache/embeddings_Qwen_Qwen3-Embedding-8B_max_length-512/
    """
    key = build_cache_key(cache_type, model=model, **params)
    path = CACHE_DIR / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_cache_metadata(cache_path: Path, *, cache_type: str, model: str, **params) -> None:
    """Write metadata.json to the cache directory with full parameters.

    Allows reviewers to inspect what artifact was cached and verify correctness.
    """
    metadata = {
        "cache_type": cache_type,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in sorted(params.items())},
    }
    with open(cache_path / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
