"""
Centralized in-memory caching layer for the Life Insurance AI Copilot.

We use **Python stdlib** (`functools.lru_cache`) combined with a lightweight
custom TTL (time-to-live) cache — no extra dependencies required.

┌────────────────────────────────────────────────────────────────────┐
│  WHY these techniques?                                            │
│                                                                   │
│  1. functools.lru_cache (Least Recently Used)                     │
│     • Built into Python — zero dependencies.                      │
│     • Perfect for PURE functions where the same input always      │
│       produces the same output (guardrails, CSV risk lookups).    │
│     • Thread-safe by default (uses internal lock).                │
│     • O(1) lookup via internal dict.                              │
│                                                                   │
│  2. Custom TTLCache (Time-To-Live)                                │
│     • For data that should expire after a while (RAG context,     │
│       session state) so stale results don't persist forever.      │
│     • Simple dict + timestamp — no external library needed.       │
│     • Configurable TTL per cache instance.                        │
│     • Thread-safe via threading.Lock.                             │
│                                                                   │
│  WHY NOT Redis / Memcached?                                       │
│     • This is a single-process app (uvicorn + Streamlit).         │
│     • No network hop = microsecond lookups vs milliseconds.       │
│     • No infrastructure to deploy/maintain.                       │
│     • If we scale to multi-process, swap TTLCache for Redis.      │
│                                                                   │
│  WHY NOT cachetools (pip)?                                        │
│     • It's a great library, but we avoid adding dependencies      │
│       when stdlib can do the job. Our TTLCache is ~30 lines.      │
└────────────────────────────────────────────────────────────────────┘
"""

import time
import threading
from functools import lru_cache
from typing import Any, Optional


# ── TTL Cache Implementation ───────────────────────────────────────────

class TTLCache:
    """
    A simple thread-safe in-memory cache with time-to-live expiration.

    Args:
        ttl_seconds: How long entries stay valid (default: 300s = 5 min).
        max_size: Maximum number of entries. Oldest are evicted when full.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 256):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if isinstance(key, str):
            key = key.strip()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            timestamp, value = entry
            if time.time() - timestamp > self._ttl:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        if isinstance(key, str):
            key = key.strip()
        with self._lock:
            # Evict oldest if at max capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), value)

    def invalidate(self, key: str) -> None:
        if isinstance(key, str):
            key = key.strip()
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "N/A",
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }


# ── Cache Instances ────────────────────────────────────────────────────

# RAG context: same query often repeated in a session; TTL prevents stale context
# if documents are re-indexed. 10-minute TTL is a good balance.
rag_cache = TTLCache(ttl_seconds=600, max_size=128)

# Guardrail results: deterministic for the same input text, but we use TTL
# in case patterns are updated at runtime. 30-minute TTL.
guardrail_cache = TTLCache(ttl_seconds=1800, max_size=512)

# Frontend state cache: avoid hammering the backend API on every Streamlit
# rerun (which can happen on every widget interaction). Short 5-second TTL.
state_cache = TTLCache(ttl_seconds=5, max_size=32)

# Session list cache: the /sessions endpoint is expensive (fetches state
# for every session). Cache for 3 seconds.
sessions_cache = TTLCache(ttl_seconds=3, max_size=1)


# ── CSV Data Cache (lru_cache — files are static) ──────────────────────

@lru_cache(maxsize=4)
def read_csv_cached(filepath: str) -> Any:
    """
    Cache CSV file reads using lru_cache.

    WHY lru_cache here:
    - CSV files on disk are STATIC during app lifetime.
    - pd.read_csv is I/O bound (~5-20ms per call).
    - lru_cache makes subsequent calls O(1) from memory.
    - maxsize=4 because we only have ~2 CSV files.
    """
    import pandas as pd
    return pd.read_csv(filepath)


# ── Aggregate Stats ────────────────────────────────────────────────────

def get_all_cache_stats() -> dict:
    """Return stats for all cache layers — useful for monitoring."""
    return {
        "rag_cache": rag_cache.stats,
        "guardrail_cache": guardrail_cache.stats,
        "state_cache": state_cache.stats,
        "sessions_cache": sessions_cache.stats,
        "csv_cache": {
            "info": str(read_csv_cached.cache_info()),
        },
    }
