"""
Build stable cache keys for experiment artifacts (embeddings, indexes, etc.).

Usage:
    from build_cache_key import build_cache_key, get_cache_path

    key = build_cache_key("embeddings", model="Qwen/Qwen3-Embedding-8B", max_length=512, pooling="last_token")
    # -> "embeddings_Qwen_Qwen3-Embedding-8B_max_length-512_pooling-last_token"

    path = get_cache_path("embeddings", model="Qwen/Qwen3-Embedding-8B", max_length=512)
    # -> Path(".cache/embeddings_Qwen_Qwen3-Embedding-8B_max_length-512/")

Cache types:
    - embeddings: document/query embedding arrays
    - index: FAISS or other search indexes
    - colbert_index: ColBERT PLAID indexes
    - terrier_index: PyTerrier inverted indexes
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / ".cache"


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
        **params: All parameters that affect the cached output (e.g., max_length=512, pooling="last_token")

    Returns:
        A clean, deterministic string like:
        "embeddings_Qwen_Qwen3-Embedding-8B_max_length-512_pooling-last_token"
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
