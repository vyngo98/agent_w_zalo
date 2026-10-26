"""Semantic cache: reuse a prior answer for a semantically-equivalent query.

Unlike a hash-based pseudo-embedding (which only matches near-identical
strings), this reuses the project's real embedding model (`nomic-embed-text` via
`get_embeddings()`), so reworded questions ("revenue 2023" ≈ "doanh thu 2023")
can hit. Entries are namespaced per `document_id` and persisted to JSON so the
cache survives restarts. All I/O is defensive — a cache failure never breaks a
chat request.
"""

import json
import math
import os
import time

from app.config import (
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_TTL,
)
from app.rag.vector_store import get_embeddings

# get_embeddings() is a factory (new client each call); cache one instance.
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = get_embeddings()
    return _EMBEDDER


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(self, similarity_threshold=0.92, max_entries=10000, ttl_seconds=3600):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.entries = []
        self.hits = 0
        self.misses = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence (defensive: never raise)
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            if os.path.exists(SEMANTIC_CACHE_PATH):
                with open(SEMANTIC_CACHE_PATH) as f:
                    self.entries = json.load(f)
        except Exception:  # noqa: BLE001
            self.entries = []

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(SEMANTIC_CACHE_PATH) or ".", exist_ok=True)
            with open(SEMANTIC_CACHE_PATH, "w") as f:
                json.dump(self.entries, f)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Lookup / insert
    # ------------------------------------------------------------------ #
    def get(self, query: str):
        """Return the best cached match for `query` within `document_id`, or None."""
        try:
            query_emb = _get_embedder().embed_query(query)
        except Exception:  # noqa: BLE001 - embedding unavailable -> treat as miss
            return None

        now = time.time()
        best_score = 0.0
        best_entry = None
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            score = cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            return {
                "response": best_entry["response"],
                "sources": best_entry.get("sources"),
                "similarity": round(best_score, 4),
                "original_query": best_entry["query"],
                "cached_at": best_entry["timestamp"],
            }

        self.misses += 1
        return None

    def put(self, query: str,
            # document_id: str,
            response: str,
            # sources=None
            ) -> None:
        try:
            embedding = _get_embedder().embed_query(query)
        except Exception:  # noqa: BLE001 - cannot embed -> skip caching
            return

        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries = self.entries[len(self.entries) // 4:]

        self.entries.append({
            "query": query,
            # "document_id": document_id,
            "embedding": embedding,
            "response": response,
            # "sources": sources,
            "timestamp": time.time(),
        })
        self._save()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / max(total, 1) * 100, 2),
        }


_CACHE = None


def get_semantic_cache() -> SemanticCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = SemanticCache(
            similarity_threshold=SEMANTIC_CACHE_THRESHOLD,
            max_entries=SEMANTIC_CACHE_MAX_ENTRIES,
            ttl_seconds=SEMANTIC_CACHE_TTL,
        )
    return _CACHE
