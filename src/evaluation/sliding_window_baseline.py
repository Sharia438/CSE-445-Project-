"""Sliding-window HDBSCAN baseline: periodically re-cluster a trailing
window of recent posts from scratch - the natural "brute force" alternative
to ``DynamicClusteringEngine``'s incremental online micro-clustering.

Scoring this on the same burst-detection benchmark as the dynamic engine
(``burst_benchmark.py``) is what actually demonstrates the value of the
online approach: not just "can a periodic-recluster baseline detect bursts
too" (it can), but at what compute cost (posts reclustered) and latency
relative to never throwing away incremental state.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src.ml_engine.static_clustering import cluster_hdbscan, reduce_umap
from src.ml_engine.vectorizer import TextVectorizer, embed_corpus_cached

LOGGER = logging.getLogger(__name__)

NOISE_LABEL = -1
DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_REFRESH_INTERVAL_HOURS = 6.0
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_GROWTH_RATIO_THRESHOLD = 2.0  # size now / size at this tracked_id's last appearance
DEFAULT_MIN_NEW_SIZE = 5  # a brand-new cluster needs at least this many posts to count as "surging"
DEFAULT_GRACE_HOURS = 6.0


def _cluster_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    for label in set(labels.tolist()):
        if label == NOISE_LABEL:
            continue
        member_embeddings = embeddings[labels == label]
        centroid = member_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[int(label)] = centroid / norm if norm > 0 else centroid
    return centroids


def run_sliding_window(
    df: pd.DataFrame,
    vectorizer: TextVectorizer | None = None,
    text_mode: str = "text",
    window_hours: float = DEFAULT_WINDOW_HOURS,
    refresh_interval_hours: float = DEFAULT_REFRESH_INTERVAL_HOURS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    embeddings_path=None,
    ids_path=None,
    meta_path=None,
) -> dict:
    """Replay ``df`` by periodically re-clustering (from scratch, UMAP +
    HDBSCAN) the trailing ``window_hours`` of posts every
    ``refresh_interval_hours``.

    HDBSCAN cluster labels aren't stable across independent reclustering
    runs, so continuity is tracked by nearest-centroid cosine similarity
    against the *previous* refresh: a cluster whose best match exceeds
    ``similarity_threshold`` inherits that match's ``tracked_id`` (and its
    size history for computing growth); anything else gets a fresh
    ``tracked_id`` and counts as newly appeared.

    Returns ``{"refresh_log": DataFrame, "n_refreshes": int,
    "total_posts_reclustered": int}``. ``refresh_log`` has one row per
    (refresh_timestamp, tracked_id): ``size``, ``is_new``, ``growth_ratio``
    (``inf`` for a brand-new tracked_id), and ``post_ids`` (a frozenset,
    used internally for burst-majority matching - not meant for CSV export).
    """
    vectorizer = vectorizer or TextVectorizer()
    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, format="ISO8601")
    ordered = ordered.sort_values("timestamp").reset_index(drop=True)

    cache_kwargs = {}
    if embeddings_path is not None:
        cache_kwargs["embeddings_path"] = embeddings_path
    if ids_path is not None:
        cache_kwargs["ids_path"] = ids_path
    if meta_path is not None:
        cache_kwargs["meta_path"] = meta_path
    embeddings = embed_corpus_cached(ordered, vectorizer=vectorizer, text_mode=text_mode, **cache_kwargs)

    start = ordered["timestamp"].min()
    end = ordered["timestamp"].max()
    window = timedelta(hours=window_hours)
    refresh_step = timedelta(hours=refresh_interval_hours)

    prev_centroids: dict[int, np.ndarray] = {}
    prev_sizes: dict[int, int] = {}
    next_tracked_id = 0
    rows = []
    total_posts_reclustered = 0
    n_refreshes = 0

    t = start + window  # first refresh once a full window has accumulated
    final_refresh = end + refresh_step  # one extra pass to catch the tail
    while t <= final_refresh:
        window_mask = (ordered["timestamp"] > t - window) & (ordered["timestamp"] <= t)
        window_df = ordered.loc[window_mask]

        if len(window_df) >= min_cluster_size:
            window_embeddings = embeddings[window_df.index.to_numpy()]
            total_posts_reclustered += len(window_df)
            n_refreshes += 1

            n_components = min(5, max(len(window_df) - 2, 2))
            reduced = reduce_umap(window_embeddings, n_components=n_components)
            labels = cluster_hdbscan(reduced, min_cluster_size=min_cluster_size, min_samples=min_samples)
            centroids = _cluster_centroids(window_embeddings, labels)

            new_centroids: dict[int, np.ndarray] = {}
            new_sizes: dict[int, int] = {}
            for label, centroid in centroids.items():
                member_mask = labels == label
                size = int(member_mask.sum())
                post_ids = frozenset(window_df.loc[member_mask, "post_id"].astype(str))

                best_tracked_id = None
                best_similarity = 0.0
                for candidate_id, candidate_centroid in prev_centroids.items():
                    similarity = float(np.dot(centroid, candidate_centroid))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_tracked_id = candidate_id

                if best_tracked_id is not None and best_similarity >= similarity_threshold:
                    tracked_id = best_tracked_id
                    is_new = False
                    previous_size = prev_sizes.get(tracked_id, size)
                    growth_ratio = size / max(previous_size, 1)
                else:
                    tracked_id = next_tracked_id
                    next_tracked_id += 1
                    is_new = True
                    growth_ratio = float("inf")

                new_centroids[tracked_id] = centroid
                new_sizes[tracked_id] = size
                rows.append(
                    {
                        "refresh_timestamp": t,
                        "tracked_id": tracked_id,
                        "size": size,
                        "is_new": is_new,
                        "growth_ratio": growth_ratio,
                        "post_ids": post_ids,
                    }
                )

            prev_centroids, prev_sizes = new_centroids, new_sizes

        t = t + refresh_step

    refresh_log = pd.DataFrame(
        rows, columns=["refresh_timestamp", "tracked_id", "size", "is_new", "growth_ratio", "post_ids"]
    )
    LOGGER.info(
        "Sliding-window baseline: %d refreshes, %d total posts reclustered",
        n_refreshes,
        total_posts_reclustered,
    )
    return {
        "refresh_log": refresh_log,
        "n_refreshes": n_refreshes,
        "total_posts_reclustered": total_posts_reclustered,
    }


def flag_surging(
    refresh_log: pd.DataFrame,
    growth_ratio_threshold: float = DEFAULT_GROWTH_RATIO_THRESHOLD,
    min_new_size: int = DEFAULT_MIN_NEW_SIZE,
) -> pd.DataFrame:
    """Mark rows of ``refresh_log`` as "surging" - this baseline's analog
    of the dynamic engine's ``trending`` state: either a brand-new cluster
    large enough to matter, or an existing one growing fast.
    """
    log = refresh_log.copy()
    if log.empty:
        return log.assign(flagged=pd.Series(dtype=bool))
    # growth_ratio is a placeholder (inf) for brand-new clusters, not a
    # meaningful growth signal - the size-based new-cluster check and the
    # growth-ratio check must apply to disjoint cases, or every new
    # cluster would trivially satisfy the growth_ratio OR-branch.
    is_new = log["is_new"]
    log["flagged"] = (is_new & (log["size"] >= min_new_size)) | (
        ~is_new & (log["growth_ratio"] >= growth_ratio_threshold)
    )
    return log


def evaluate_burst_detection_sliding_window(
    df: pd.DataFrame,
    truth_df: pd.DataFrame,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    refresh_interval_hours: float = DEFAULT_REFRESH_INTERVAL_HOURS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    growth_ratio_threshold: float = DEFAULT_GROWTH_RATIO_THRESHOLD,
    min_new_size: int = DEFAULT_MIN_NEW_SIZE,
    grace_hours: float = DEFAULT_GRACE_HOURS,
    vectorizer: TextVectorizer | None = None,
    text_mode: str = "text",
    embeddings_path=None,
    ids_path=None,
    meta_path=None,
) -> dict:
    """Score the sliding-window baseline on the same injected-burst corpus
    ``burst_benchmark.evaluate_burst_detection`` uses, with a matching
    return shape (``per_burst``, ``detection_rate``, ``precision``,
    ``recall``, ``f1``, latencies) plus two compute-cost fields
    (``n_refreshes``, ``total_posts_reclustered``) that the dynamic engine
    doesn't need - direct evidence of the incremental engine's efficiency
    advantage, not just parity on detection.
    """
    result = run_sliding_window(
        df,
        vectorizer=vectorizer,
        text_mode=text_mode,
        window_hours=window_hours,
        refresh_interval_hours=refresh_interval_hours,
        similarity_threshold=similarity_threshold,
        embeddings_path=embeddings_path,
        ids_path=ids_path,
        meta_path=meta_path,
    )
    refresh_log = flag_surging(result["refresh_log"], growth_ratio_threshold, min_new_size)
    flagged_log = refresh_log[refresh_log["flagged"]] if not refresh_log.empty else refresh_log

    per_burst_rows = []
    matched_tracked_ids: set[int] = set()

    for row in truth_df.itertuples():
        burst_start = pd.Timestamp(row.burst_start)
        burst_end = pd.Timestamp(row.burst_end)
        deadline = burst_end + timedelta(hours=grace_hours)
        prefix = f"injected_{row.burst_id}_"

        candidate_rows = (
            refresh_log[
                (refresh_log["refresh_timestamp"] >= burst_start)
                & (refresh_log["refresh_timestamp"] <= deadline)
            ]
            if not refresh_log.empty
            else refresh_log
        )

        best_tracked_id = None
        best_overlap = 0
        for candidate in candidate_rows.itertuples():
            overlap = sum(1 for pid in candidate.post_ids if pid.startswith(prefix))
            if overlap > best_overlap:
                best_overlap = overlap
                best_tracked_id = candidate.tracked_id

        detected = False
        latency_seconds = None
        if best_tracked_id is not None and not flagged_log.empty:
            flags = flagged_log[
                (flagged_log["tracked_id"] == best_tracked_id)
                & (flagged_log["refresh_timestamp"] >= burst_start)
                & (flagged_log["refresh_timestamp"] <= deadline)
            ]
            if not flags.empty:
                first_flag = flags["refresh_timestamp"].min()
                detected = True
                latency_seconds = (first_flag - burst_start).total_seconds()
                matched_tracked_ids.add(best_tracked_id)

        per_burst_rows.append(
            {
                "burst_id": row.burst_id,
                "topic_label": row.topic_label,
                "burst_start": burst_start,
                "majority_tracked_id": best_tracked_id,
                "overlap": best_overlap,
                "detected": detected,
                "latency_seconds": latency_seconds,
            }
        )

    per_burst = pd.DataFrame(per_burst_rows)
    all_flagged_ids = set(flagged_log["tracked_id"].unique().tolist()) if not flagged_log.empty else set()
    false_positive_ids = all_flagged_ids - matched_tracked_ids

    n_bursts = len(truth_df)
    n_detected = int(per_burst["detected"].sum()) if not per_burst.empty else 0
    recall = n_detected / n_bursts if n_bursts else 0.0

    tp = len(matched_tracked_ids)
    fp = len(false_positive_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    latencies = (
        per_burst.loc[per_burst["detected"], "latency_seconds"]
        if not per_burst.empty
        else pd.Series(dtype=float)
    )

    LOGGER.info(
        "Sliding-window burst benchmark: %d/%d detected, precision=%.2f, recall=%.2f, "
        "%d refreshes / %d posts reclustered",
        n_detected,
        n_bursts,
        precision,
        recall,
        result["n_refreshes"],
        result["total_posts_reclustered"],
    )

    return {
        "per_burst": per_burst,
        "detection_rate": recall,
        "median_latency_seconds": float(latencies.median()) if len(latencies) else None,
        "p90_latency_seconds": float(latencies.quantile(0.9)) if len(latencies) else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_bursts": n_bursts,
        "n_false_positive_clusters": fp,
        "n_refreshes": result["n_refreshes"],
        "total_posts_reclustered": result["total_posts_reclustered"],
    }
