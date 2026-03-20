"""
Build stable cache keys for experiment artifacts (embeddings, indexes, etc.).

Usage:
    from build_cache_key import build_cache_key, get_cache_path, save_cache_metadata, verify_cache_metadata

    # Get cache path (creates dir + writes metadata.json)
    path = get_cache_path("embeddings", model="Qwen/Qwen3-Embedding-8B",
                          max_length=512, pooling="last_token", dataset="robust04")
    # -> Path(".cache/embeddings_Qwen_Qwen3-Embedding-8B_dataset-robust04_max_length-512_pooling-last_token/")

    # Check if cache exists and metadata matches
    if verify_cache_metadata(path, cache_type="embeddings", model="Qwen/Qwen3-Embedding-8B",
                             max_length=512, pooling="last_token", dataset="robust04"):
        embeddings = np.load(path / "doc_embeddings.npy")
    else:
        embeddings = encode_all_docs(...)
        np.save(path / "doc_embeddings.npy", embeddings)
        save_cache_metadata(path, cache_type="embeddings", model="Qwen/Qwen3-Embedding-8B",
                            max_length=512, pooling="last_token", dataset="robust04")

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

PROJECT_ROOT = Path(__file__).parent
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

    The metadata allows readers to verify they're loading the correct artifact.
    """
    metadata = {
        "cache_type": cache_type,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in sorted(params.items())},
    }
    with open(cache_path / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def verify_cache_metadata(cache_path: Path, *, cache_type: str, model: str, **params) -> bool:
    """Verify that a cache directory's metadata.json matches the expected parameters.

    Returns True if metadata exists and all parameters match. Returns False if
    metadata is missing or any parameter differs.
    """
    meta_path = cache_path / METADATA_FILE
    if not meta_path.exists():
        return False

    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if metadata.get("cache_type") != cache_type:
        return False
    if metadata.get("model") != model:
        return False

    for key, expected in params.items():
        stored = metadata.get(key)
        # Compare as strings to handle type mismatches (e.g., int vs str)
        if str(stored) != str(expected):
            return False

    return True
