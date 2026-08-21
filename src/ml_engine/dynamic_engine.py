"""Dynamic (online) trend-detection engine.

Ties together ``micro_cluster.py`` (time-decayed cluster state) and
``velocity_scorer.py`` (burst/trend classification) into a single engine
that consumes posts one at a time, in chronological order, the way a real
streaming pipeline would. ``run_replay`` drives this over a historical
DataFrame to simulate that streaming behavior for demos, evaluation, and
the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.ml_engine.micro_cluster import MicroClusterRegistry
from src.ml_engine.vectorizer import TextVectorizer, embed_corpus_cached
from src.ml_engine.velocity_scorer import (
    DEFAULT_BURST_Z_THRESHOLD,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_TRENDING_GROWTH,
    VelocityTracker,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_HALF_LIFE_SECONDS = 6 * 3600.0
DEFAULT_MIN_CLUSTER_WEIGHT = 0.05


class DynamicClusteringEngine:
    """Online micro-clustering engine with time-decay and burst scoring."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS,
        min_cluster_weight: float = DEFAULT_MIN_CLUSTER_WEIGHT,
        velocity_window_size: int = 20,
        snapshot_interval_seconds: float | None = None,
        trending_growth: float = DEFAULT_TRENDING_GROWTH,
        burst_z_threshold: float = DEFAULT_BURST_Z_THRESHOLD,
        track_state_log: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.half_life_seconds = half_life_seconds
        self.min_cluster_weight = min_cluster_weight
        # Sample the whole registry ~8 times per half-life by default: often
        # enough that a burst's rise-and-fall is visible in the history
        # window, without snapshotting on every single post.
        self.snapshot_interval_seconds = (
            snapshot_interval_seconds
            if snapshot_interval_seconds is not None
            else half_life_seconds / 8.0
        )
        self.trending_growth = trending_growth
        self.burst_z_threshold = burst_z_threshold
        self.track_state_log = track_state_log

        self.registry = MicroClusterRegistry()
        self.velocity = VelocityTracker(window_size=velocity_window_size)
        self._last_snapshot_at: datetime | None = None
        self._last_states: dict[int, str] = {}
        self.state_log: list[tuple[datetime, int, str]] = []

    def process_post(
        self, post_id: str, title: str, timestamp: datetime, embedding: np.ndarray
    ) -> int:
        """Feed a single post through the engine. Returns the id of the
        cluster it was assigned to (existing or newly created).
        """
        self.registry.decay_all(timestamp, self.half_life_seconds)
        forgotten = self.registry.prune(self.min_cluster_weight)
        if forgotten:
            self.velocity.forget(forgotten)

        best_cluster, similarity = self.registry.find_best_match(embedding)

        if best_cluster is not None and similarity >= self.similarity_threshold:
            best_cluster.update(embedding, timestamp, title)
            cluster = best_cluster
        else:
            cluster = self.registry.add(embedding, timestamp, title)

        self._maybe_snapshot(timestamp)
        return cluster.id

    def _maybe_snapshot(self, timestamp: datetime) -> None:
        """Record every live cluster's current (decayed) weight on a
        regular time grid.

        Sampling only the cluster that was just touched (as an earlier
        version of this engine did) means decay between reinforcements is
        never observed, so a cluster can never be seen shrinking - this
        grid is what makes "declining" and "trending" actually reachable.
        """
        due = (
            self._last_snapshot_at is None
            or (timestamp - self._last_snapshot_at).total_seconds()
            >= self.snapshot_interval_seconds
        )
        if not due:
            return
        self._last_snapshot_at = timestamp

        for cluster_id, cluster in self.registry.clusters.items():
            self.velocity.record(cluster_id, timestamp, cluster.weight)

        if self.track_state_log:
            snap = self.velocity.snapshot(
                half_life_seconds=self.half_life_seconds,
                trending_growth=self.trending_growth,
                burst_z_threshold=self.burst_z_threshold,
            )
            for row in snap.itertuples():
                if self._last_states.get(row.cluster_id) != row.state:
                    self.state_log.append((timestamp, row.cluster_id, row.state))
                    self._last_states[row.cluster_id] = row.state

    def active_cluster_summary(
        self,
        trending_growth: float | None = None,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        burst_z_threshold: float | None = None,
        samples_per_cluster: int = 5,
        auto_calibrate: bool = False,
    ) -> pd.DataFrame:
        """Snapshot of every currently-active cluster: weight, growth rate,
        burst z-score, trending state, and a few recent sample titles.
        """
        velocity_df = self.velocity.snapshot(
            half_life_seconds=self.half_life_seconds,
            trending_growth=trending_growth if trending_growth is not None else self.trending_growth,
            min_weight=min_weight,
            burst_z_threshold=(
                burst_z_threshold if burst_z_threshold is not None else self.burst_z_threshold
            ),
            auto_calibrate=auto_calibrate,
        )
        if velocity_df.empty:
            return velocity_df.assign(sample_titles=[], member_count=[])

        # velocity.forget() already drops pruned clusters, but guard here
        # too in case a caller holds a stale snapshot/reference.
        live_ids = set(self.registry.clusters)
        velocity_df = velocity_df[velocity_df["cluster_id"].isin(live_ids)].reset_index(drop=True)
        if velocity_df.empty:
            return velocity_df.assign(sample_titles=[], member_count=[])

        sample_titles = []
        member_counts = []
        for cluster_id in velocity_df["cluster_id"]:
            cluster = self.registry.clusters.get(cluster_id)
            sample_titles.append(
                " | ".join(list(cluster.recent_titles)[-samples_per_cluster:]) if cluster else ""
            )
            member_counts.append(cluster.member_count if cluster else 0)
        velocity_df["sample_titles"] = sample_titles
        velocity_df["member_count"] = member_counts

        return velocity_df.sort_values(["state", "growth_rate"], ascending=[True, False]).reset_index(
            drop=True
        )


@dataclass
class ReplayResult:
    """Everything a caller might want from replaying a corpus through the
    dynamic engine: the engine's final state, a snapshot of active
    clusters, per-post cluster assignments (for scoring against pseudo-
    labels), the raw weight-over-time history, and (if the engine was
    constructed with ``track_state_log=True``) the timeline of state
    transitions used by the burst-detection benchmark.
    """

    engine: DynamicClusteringEngine
    summary: pd.DataFrame
    assignments: pd.DataFrame
    history: pd.DataFrame
    state_log: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["timestamp", "cluster_id", "state"]))

    def __iter__(self):
        # Preserves the old `engine, summary = run_replay(...)` call shape
        # for any external code that hasn't been updated to the richer
        # ReplayResult yet.
        return iter((self.engine, self.summary))


def run_replay(
    df: pd.DataFrame,
    vectorizer: TextVectorizer | None = None,
    engine: DynamicClusteringEngine | None = None,
    embeddings_path: str | None = None,
    ids_path: str | None = None,
    meta_path: str | None = None,
    text_mode: str = "text",
) -> ReplayResult:
    """Replay a historical corpus through the dynamic engine in
    chronological order, simulating a live stream.

    ``embeddings_path`` / ``ids_path`` / ``meta_path`` are forwarded to
    ``embed_corpus_cached``. Pass a writable location (e.g. Kaggle's
    ``/kaggle/working``) when the default ``data/`` folder is read-only.

    Returns a ``ReplayResult`` with the engine's final state, an active
    cluster summary, per-post assignments (for scoring against pseudo
    labels), the full weight history, and the state-transition log (empty
    unless ``engine`` was constructed with ``track_state_log=True``).
    """
    vectorizer = vectorizer or TextVectorizer()
    engine = engine or DynamicClusteringEngine()

    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, format="ISO8601")
    ordered = ordered.sort_values("timestamp").reset_index(drop=True)

    LOGGER.info("Embedding %d posts for replay", len(ordered))
    cache_kwargs = {}
    if embeddings_path is not None:
        cache_kwargs["embeddings_path"] = embeddings_path
    if ids_path is not None:
        cache_kwargs["ids_path"] = ids_path
    if meta_path is not None:
        cache_kwargs["meta_path"] = meta_path
    embeddings = embed_corpus_cached(
        ordered, vectorizer=vectorizer, text_mode=text_mode, **cache_kwargs
    )

    assignments = []
    for row, embedding in zip(ordered.itertuples(index=False), embeddings):
        cluster_id = engine.process_post(
            row.post_id, row.title, row.timestamp.to_pydatetime(), embedding
        )
        assignments.append((row.post_id, cluster_id, row.timestamp))

    assignments_df = pd.DataFrame(assignments, columns=["post_id", "cluster_id", "timestamp"])

    LOGGER.info(
        "Replay complete: %d active clusters after processing %d posts",
        len(engine.registry.clusters),
        len(ordered),
    )

    history_frames = []
    for cluster_id in engine.velocity.tracked_cluster_ids():
        hist = engine.velocity.history_df(cluster_id)
        hist["cluster_id"] = cluster_id
        history_frames.append(hist)
    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(columns=["timestamp", "weight", "cluster_id"])
    )

    state_log_df = pd.DataFrame(engine.state_log, columns=["timestamp", "cluster_id", "state"])
    if not state_log_df.empty:
        state_log_df["timestamp"] = pd.to_datetime(state_log_df["timestamp"], utc=True)

    return ReplayResult(
        engine=engine,
        summary=engine.active_cluster_summary(),
        assignments=assignments_df,
        history=history_df,
        state_log=state_log_df,
    )


if __name__ == "__main__":
    import argparse
    import sys

    from src.ml_engine.vectorizer import DEFAULT_CSV_PATH, load_corpus

    # Real multi-platform text (Twitter/YouTube/Reddit) can contain
    # non-Latin-1 characters that Windows' default console encoding
    # (cp1252) can't display; degrade to '?' instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run the dynamic clustering engine over a CSV.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--text-mode", choices=["text", "title_text"], default="text")
    args = parser.parse_args()

    corpus = load_corpus(args.csv_path)
    result = run_replay(corpus, text_mode=args.text_mode)
    with pd.option_context("display.max_colwidth", 80):
        print(result.summary.to_string(index=False))
