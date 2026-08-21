"""Burst-detection benchmark: how fast, and how accurately, does the
dynamic engine flag an injected burst as "trending"?

Pairs with ``injection.inject_bursts``, which produces a corpus with known
burst windows. This is the primary evidence that the "dynamic clustering
model identifies emerging trends" claim in the problem statement actually
holds, with a number attached (detection latency, precision, recall)
instead of an unverified assertion.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.ml_engine.dynamic_engine import DynamicClusteringEngine, run_replay
from src.ml_engine.vectorizer import TextVectorizer

LOGGER = logging.getLogger(__name__)

DEFAULT_GRACE_HOURS = 6.0


def _majority_cluster_for_burst(
    assignments: pd.DataFrame, burst_id: int
) -> tuple[int | None, float]:
    """Cluster id that captured the largest share of a burst's injected
    posts, and that share (purity). ``None`` if none of the burst's posts
    made it into the assignments (shouldn't happen, but is possible if the
    corpus was truncated).
    """
    prefix = f"injected_{burst_id}_"
    burst_assignments = assignments[assignments["post_id"].str.startswith(prefix)]
    if burst_assignments.empty:
        return None, 0.0
    counts = burst_assignments["cluster_id"].value_counts()
    majority_cluster = int(counts.index[0])
    purity = float(counts.iloc[0] / len(burst_assignments))
    return majority_cluster, purity


def evaluate_burst_detection(
    df: pd.DataFrame,
    truth_df: pd.DataFrame,
    engine_params: dict | None = None,
    vectorizer: TextVectorizer | None = None,
    grace_hours: float = DEFAULT_GRACE_HOURS,
    text_mode: str = "text",
    embeddings_path: str | Path | None = None,
    ids_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> dict:
    """Replay ``df`` (background + injected bursts, from
    ``injection.inject_bursts``) through the dynamic engine and score how
    well it detects each burst in ``truth_df``.

    For each ground-truth burst, the cluster capturing the majority of its
    injected posts is identified (``purity``); detection succeeds if that
    cluster is flagged ``"trending"`` at some point between the burst's
    start and ``burst_end + grace_hours``, with latency measured from
    ``burst_start``. Any cluster that goes ``"trending"`` without being the
    majority cluster for some burst counts as a false positive.

    Returns a dict with ``per_burst`` (DataFrame), ``detection_rate``,
    ``median_latency_seconds``, ``p90_latency_seconds``, ``precision``,
    ``recall``, ``f1``, ``n_bursts``, ``n_false_positive_clusters``.
    """
    engine_params = engine_params or {}
    engine = DynamicClusteringEngine(track_state_log=True, **engine_params)
    cache_kwargs = {}
    if embeddings_path is not None:
        cache_kwargs["embeddings_path"] = embeddings_path
    if ids_path is not None:
        cache_kwargs["ids_path"] = ids_path
    if meta_path is not None:
        cache_kwargs["meta_path"] = meta_path

    result = run_replay(df, vectorizer=vectorizer, engine=engine, text_mode=text_mode, **cache_kwargs)

    state_log = result.state_log
    trending_log = state_log[state_log["state"] == "trending"] if not state_log.empty else state_log

    per_burst_rows = []
    matched_clusters: set[int] = set()

    for row in truth_df.itertuples():
        burst_start = pd.Timestamp(row.burst_start)
        burst_end = pd.Timestamp(row.burst_end)
        deadline = burst_end + timedelta(hours=grace_hours)

        majority_cluster, purity = _majority_cluster_for_burst(result.assignments, row.burst_id)

        detected = False
        latency_seconds = None
        if majority_cluster is not None and not trending_log.empty:
            candidate_flags = trending_log[
                (trending_log["cluster_id"] == majority_cluster)
                & (trending_log["timestamp"] >= burst_start)
                & (trending_log["timestamp"] <= deadline)
            ]
            if not candidate_flags.empty:
                first_flag = candidate_flags["timestamp"].min()
                detected = True
                latency_seconds = (first_flag - burst_start).total_seconds()
                matched_clusters.add(majority_cluster)

        per_burst_rows.append(
            {
                "burst_id": row.burst_id,
                "topic_label": row.topic_label,
                "burst_start": burst_start,
                "majority_cluster": majority_cluster,
                "purity": purity,
                "detected": detected,
                "latency_seconds": latency_seconds,
            }
        )

    per_burst = pd.DataFrame(per_burst_rows)

    all_trending_clusters = (
        set(trending_log["cluster_id"].unique().tolist()) if not trending_log.empty else set()
    )
    false_positive_clusters = all_trending_clusters - matched_clusters

    n_bursts = len(truth_df)
    n_detected = int(per_burst["detected"].sum()) if not per_burst.empty else 0
    recall = n_detected / n_bursts if n_bursts else 0.0

    tp = len(matched_clusters)
    fp = len(false_positive_clusters)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    latencies = (
        per_burst.loc[per_burst["detected"], "latency_seconds"]
        if not per_burst.empty
        else pd.Series(dtype=float)
    )

    LOGGER.info(
        "Burst benchmark: %d/%d detected, precision=%.2f, recall=%.2f, median latency=%s",
        n_detected,
        n_bursts,
        precision,
        recall,
        f"{latencies.median() / 3600:.1f}h" if len(latencies) else "n/a",
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
    }


if __name__ == "__main__":
    import argparse
    import sys

    from src.data_ingestion.historical_loader import ALL_SOURCES, load_all_sources
    from src.evaluation.injection import inject_bursts

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run the semi-synthetic burst-detection benchmark.")
    parser.add_argument("--source", choices=ALL_SOURCES, default="reddit")
    parser.add_argument("--n-bursts", type=int, default=6)
    parser.add_argument("--posts-per-burst", type=int, default=80)
    args = parser.parse_args()

    background = load_all_sources([args.source])
    if background.empty:
        raise SystemExit(
            f"No background data available for source={args.source!r}; "
            "download the corresponding Kaggle CSV first (see README)."
        )

    corpus, truth = inject_bursts(background, n_bursts=args.n_bursts, posts_per_burst=args.posts_per_burst)
    metrics = evaluate_burst_detection(corpus, truth)

    with pd.option_context("display.max_colwidth", 80):
        print(metrics["per_burst"].to_string(index=False))
    print()
    print(
        f"detection_rate={metrics['detection_rate']:.2f} "
        f"precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} f1={metrics['f1']:.2f}"
    )
    if metrics["median_latency_seconds"] is not None:
        print(f"median latency: {metrics['median_latency_seconds'] / 3600:.1f}h")
