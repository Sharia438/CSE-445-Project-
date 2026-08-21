"""Shared pytest fixtures.

Tests use fake, low-dimensional embeddings so the whole suite runs in
milliseconds without downloading or running the sentence-transformer
model - only ``test_vectorizer.py``'s cache-behavior tests need a
vectorizer at all, and those use a fake one too (see its ``_FakeVectorizer``).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def make_embedding(rng):
    """Deterministic unit embeddings, distinct per integer key, so two
    calls with the same key are identical (same "topic") and different
    keys are far apart (cosine near 0), without needing a real model.
    """
    cache: dict[int, np.ndarray] = {}

    def _make(key: int, dim: int = 8) -> np.ndarray:
        if key not in cache:
            local_rng = np.random.default_rng(1000 + key)
            cache[key] = unit(local_rng.normal(size=dim).astype(np.float32))
        return cache[key]

    return _make


class FakeVectorizer:
    """Deterministic, model-free stand-in for ``TextVectorizer`` so tests
    exercising ``embed_corpus_cached``/``run_replay`` don't need to
    download or run a real sentence-transformer model.

    The same text always maps to the same unit vector (via an md5-seeded
    RNG, so it's stable across processes regardless of ``PYTHONHASHSEED``);
    different text maps to a different direction.
    """

    def __init__(self, dim: int = 8, model_name: str = "fake-test-vectorizer"):
        self.dim = dim
        self.model_name = model_name

    def _embed_one(self, text: str) -> np.ndarray:
        # Seeded on (model_name, text) so two "different models" embedding
        # the same text produce different vectors, the way real models
        # would - this is what makes cache-invalidation-on-model-change
        # tests meaningful rather than coincidentally passing.
        digest = hashlib.md5(f"{self.model_name}|{text}".encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        local_rng = np.random.default_rng(seed)
        return unit(local_rng.normal(size=self.dim).astype(np.float32))

    def fit_transform(self, texts) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts])

    transform = fit_transform


@pytest.fixture
def fake_vectorizer() -> FakeVectorizer:
    return FakeVectorizer()
