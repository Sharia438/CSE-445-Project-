"""Online micro-clusters with exponential time-decay ("algorithmic forgetting").

The registry stores centroids and weights as parallel numpy arrays so that
decay and nearest-cluster lookup - both invoked on every incoming post - are
single vectorized operations instead of per-cluster Python-level calls.
``MicroCluster`` is a thin metadata view over one row of those arrays: its
``centroid``/``weight`` properties read the live registry state directly, so
they can never drift out of sync with the vectorized decay/update path.

This is the core data structure the dynamic engine (``dynamic_engine.py``)
uses to track topics as they emerge, grow, and fade.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

MAX_RECENT_TITLES = 8
_INITIAL_CAPACITY = 64
_GROWTH_FACTOR = 2
_COMPACT_DEAD_ROW_MIN = 64
_COMPACT_DEAD_ROW_RATIO = 0.1


@dataclass
class MicroCluster:
    """Metadata view of a single evolving topic cluster.

    ``centroid`` and ``weight`` are properties that read directly from the
    owning registry's backing arrays rather than being stored on the
    instance, so a vectorized decay/update in the registry is immediately
    reflected here with no separate sync step.
    """

    id: int
    created_at: datetime
    last_updated: datetime
    member_count: int = 1
    recent_titles: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_TITLES))
    registry: "MicroClusterRegistry | None" = field(default=None, repr=False, compare=False)
    row: int = field(default=-1, repr=False, compare=False)

    @property
    def centroid(self) -> np.ndarray:
        return self.registry._matrix[self.row]

    @property
    def weight(self) -> float:
        return float(self.registry._weights[self.row])

    def cosine_similarity(self, embedding: np.ndarray) -> float:
        # Both sides are already unit-normalized (TextVectorizer's output,
        # and this centroid immediately after update()), so the dot product
        # *is* cosine similarity - no norm computation needed per call.
        return float(np.dot(self.centroid, embedding))

    def update(self, embedding: np.ndarray, timestamp: datetime, title: str) -> None:
        """Fold a new member into this cluster: recompute the centroid as a
        weighted running mean, bump weight, and record the title.
        """
        self.registry._update_row(self.row, embedding)
        self.member_count += 1
        self.last_updated = timestamp
        self.recent_titles.append(title)


class MicroClusterRegistry:
    """Owns the set of currently active micro-clusters.

    Centroids and weights live in parallel numpy arrays (``_matrix``,
    ``_weights``), grown by amortized doubling. Decay is a single
    scalar-vs-vector multiply applied to every live row - correct because
    every cluster is always decayed to the same ``now``, so there is no
    per-cluster elapsed time to track separately - and nearest-cluster
    lookup is a single matrix-vector dot product instead of a Python loop.

    Pruned rows are only marked dead, not immediately removed: physically
    compacting the arrays (and remapping every survivor's row index) is
    deferred until dead rows accumulate past a threshold, which keeps
    ``add``/``prune`` cheap on the common case of a handful of dying
    clusters among many live ones.
    """

    def __init__(self) -> None:
        self._matrix = np.zeros((_INITIAL_CAPACITY, 0), dtype=np.float32)
        self._weights = np.zeros(_INITIAL_CAPACITY, dtype=np.float64)
        self._alive = np.zeros(_INITIAL_CAPACITY, dtype=bool)
        self._n_rows = 0
        self._id_counter = itertools.count()
        self._clusters: dict[int, MicroCluster] = {}
        self._row_of: dict[int, int] = {}
        self._id_of_row: dict[int, int] = {}
        self._last_decay_at: datetime | None = None

    @property
    def clusters(self) -> dict[int, MicroCluster]:
        return self._clusters

    def _ensure_capacity(self, dim: int) -> None:
        if self._matrix.shape[1] == 0 and dim:
            self._matrix = np.zeros((self._matrix.shape[0], dim), dtype=np.float32)
        if self._n_rows >= self._matrix.shape[0]:
            new_capacity = self._matrix.shape[0] * _GROWTH_FACTOR
            grown = np.zeros((new_capacity, self._matrix.shape[1]), dtype=np.float32)
            grown[: self._matrix.shape[0]] = self._matrix
            self._matrix = grown

            grown_w = np.zeros(new_capacity, dtype=np.float64)
            grown_w[: self._weights.shape[0]] = self._weights
            self._weights = grown_w

            grown_alive = np.zeros(new_capacity, dtype=bool)
            grown_alive[: self._alive.shape[0]] = self._alive
            self._alive = grown_alive

    def decay_all(self, now: datetime, half_life_seconds: float) -> None:
        """Decay every live cluster's weight to ``now`` in one vector op."""
        if self._last_decay_at is None:
            self._last_decay_at = now
            return
        elapsed = (now - self._last_decay_at).total_seconds()
        if elapsed <= 0:
            return
        factor = 2.0 ** (-elapsed / half_life_seconds)
        self._weights[: self._n_rows] *= factor
        self._last_decay_at = now

    def find_best_match(self, embedding: np.ndarray) -> tuple[MicroCluster | None, float]:
        """Return the most similar active cluster and its similarity score,
        or ``(None, 0.0)`` if there are no live clusters.
        """
        if self._n_rows == 0:
            return None, 0.0
        active = self._alive[: self._n_rows]
        if not active.any():
            return None, 0.0

        scores = self._matrix[: self._n_rows] @ np.asarray(embedding, dtype=np.float32)
        scores = np.where(active, scores, -np.inf)
        best_row = int(np.argmax(scores))
        best_similarity = float(scores[best_row])
        if not np.isfinite(best_similarity):
            return None, 0.0

        cluster_id = self._id_of_row[best_row]
        return self._clusters[cluster_id], best_similarity

    def add(self, embedding: np.ndarray, timestamp: datetime, title: str) -> MicroCluster:
        embedding = np.asarray(embedding, dtype=np.float32)
        self._ensure_capacity(embedding.shape[0])

        row = self._n_rows
        self._n_rows += 1
        norm = np.linalg.norm(embedding) or 1.0
        self._matrix[row] = embedding / norm
        self._weights[row] = 1.0
        self._alive[row] = True

        cluster_id = next(self._id_counter)
        cluster = MicroCluster(
            id=cluster_id,
            created_at=timestamp,
            last_updated=timestamp,
            registry=self,
            row=row,
        )
        cluster.recent_titles.append(title)

        self._clusters[cluster_id] = cluster
        self._row_of[cluster_id] = row
        self._id_of_row[row] = cluster_id
        return cluster

    def _update_row(self, row: int, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        weight = self._weights[row]
        total_weight = weight + 1.0
        centroid = (self._matrix[row] * weight + embedding) / total_weight
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self._matrix[row] = centroid
        self._weights[row] = total_weight

    def prune(self, min_weight: float = 0.05) -> list[int]:
        """Drop clusters whose decayed weight has fallen below
        ``min_weight``. Returns the list of removed cluster ids.
        """
        if self._n_rows == 0:
            return []
        active = self._alive[: self._n_rows]
        dying = active & (self._weights[: self._n_rows] < min_weight)
        dying_rows = np.nonzero(dying)[0]
        if dying_rows.size == 0:
            return []

        forgotten_ids = [self._id_of_row[int(r)] for r in dying_rows]
        for row in dying_rows:
            row = int(row)
            cluster_id = self._id_of_row.pop(row)
            del self._clusters[cluster_id]
            del self._row_of[cluster_id]
        self._alive[dying_rows] = False

        dead_count = self._n_rows - int(self._alive[: self._n_rows].sum())
        if dead_count >= max(_COMPACT_DEAD_ROW_MIN, int(_COMPACT_DEAD_ROW_RATIO * self._n_rows)):
            self._compact()

        return forgotten_ids

    def _compact(self) -> None:
        """Physically remove dead rows and remap every survivor's row
        index. Deferred until dead rows pile up (see ``prune``) so this
        O(n) shuffle doesn't run on every call.
        """
        alive_rows = np.nonzero(self._alive[: self._n_rows])[0]
        new_n = alive_rows.size

        self._matrix[:new_n] = self._matrix[alive_rows]
        self._weights[:new_n] = self._weights[alive_rows]
        self._alive[:new_n] = True
        self._alive[new_n:] = False

        new_id_of_row: dict[int, int] = {}
        for new_row, old_row in enumerate(alive_rows):
            cluster_id = self._id_of_row[int(old_row)]
            new_id_of_row[new_row] = cluster_id
            self._row_of[cluster_id] = new_row
            self._clusters[cluster_id].row = new_row
        self._id_of_row = new_id_of_row
        self._n_rows = new_n
