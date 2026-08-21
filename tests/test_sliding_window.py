"""Tests for the sliding-window HDBSCAN baseline - the periodic
"re-cluster from scratch" comparison against the dynamic engine's
incremental online micro-clustering.
"""

from __future__ import annotations

from src.data_ingestion.historical_loader import generate_synthetic_dataset
from src.evaluation.injection import inject_bursts
from src.evaluation.sliding_window_baseline import (
    evaluate_burst_detection_sliding_window,
    flag_surging,
    run_sliding_window,
)


def _background_df():
    return generate_synthetic_dataset(n_per_topic=25, seed=1)


def test_run_sliding_window_reclusters_and_tracks_continuity(fake_vectorizer, tmp_path):
    df = _background_df()
    result = run_sliding_window(
        df,
        vectorizer=fake_vectorizer,
        window_hours=12.0,
        refresh_interval_hours=4.0,
        min_cluster_size=3,
        min_samples=3,
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )

    assert result["n_refreshes"] > 0
    assert result["total_posts_reclustered"] > 0
    log = result["refresh_log"]
    assert not log.empty
    assert set(log.columns) >= {"refresh_timestamp", "tracked_id", "size", "is_new", "growth_ratio", "post_ids"}


def test_flag_surging_marks_new_and_fast_growing_clusters():
    import pandas as pd

    log = pd.DataFrame(
        {
            "refresh_timestamp": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"], utc=True),
            "tracked_id": [1, 2, 3],
            "size": [10, 3, 20],
            "is_new": [True, True, False],
            "growth_ratio": [float("inf"), float("inf"), 3.0],
            "post_ids": [frozenset(), frozenset(), frozenset()],
        }
    )
    flagged = flag_surging(log, growth_ratio_threshold=2.0, min_new_size=5)
    assert flagged.loc[flagged["tracked_id"] == 1, "flagged"].iloc[0]  # new + large enough
    assert not flagged.loc[flagged["tracked_id"] == 2, "flagged"].iloc[0]  # new but too small
    assert flagged.loc[flagged["tracked_id"] == 3, "flagged"].iloc[0]  # not new but fast-growing


def test_sliding_window_burst_benchmark_matches_burst_benchmark_shape(fake_vectorizer, tmp_path):
    background = _background_df()
    df, truth = inject_bursts(background, n_bursts=3, posts_per_burst=20, burst_span_hours=4.0, seed=11)

    metrics = evaluate_burst_detection_sliding_window(
        df,
        truth,
        window_hours=12.0,
        refresh_interval_hours=4.0,
        vectorizer=fake_vectorizer,
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )

    expected_keys = {
        "per_burst",
        "detection_rate",
        "median_latency_seconds",
        "p90_latency_seconds",
        "precision",
        "recall",
        "f1",
        "n_bursts",
        "n_false_positive_clusters",
        "n_refreshes",
        "total_posts_reclustered",
    }
    assert expected_keys <= set(metrics.keys())
    assert metrics["n_bursts"] == 3
    assert metrics["n_refreshes"] > 0
    assert metrics["total_posts_reclustered"] > 0
