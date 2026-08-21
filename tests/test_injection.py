"""Tests for semi-synthetic burst injection and the burst-detection
benchmark it feeds.
"""

from __future__ import annotations

import pandas as pd

from src.data_ingestion.historical_loader import generate_synthetic_dataset
from src.evaluation.burst_benchmark import evaluate_burst_detection
from src.evaluation.injection import inject_bursts


def _background_df() -> pd.DataFrame:
    # A small, fast-to-embed background corpus with its own (pre-existing)
    # topics, standing in for real platform data.
    return generate_synthetic_dataset(n_per_topic=25, seed=1)


def test_inject_bursts_windows_match_generated_timestamps():
    background = _background_df()
    df, truth = inject_bursts(background, n_bursts=3, posts_per_burst=15, burst_span_hours=4.0, seed=3)

    assert len(truth) == 3
    injected = df[df["platform"] == "injected"].copy()
    injected["timestamp"] = pd.to_datetime(injected["timestamp"], utc=True, format="ISO8601")

    for row in truth.itertuples():
        posts = injected[injected["source"] == row.topic_label]
        assert len(posts) == row.n_posts
        burst_start = pd.Timestamp(row.burst_start)
        burst_end = pd.Timestamp(row.burst_end)
        assert (posts["timestamp"] >= burst_start).all()
        assert (posts["timestamp"] <= burst_end).all()


def test_inject_bursts_preserves_background_row_count():
    background = _background_df()
    df, _truth = inject_bursts(background, n_bursts=2, posts_per_burst=10, seed=5)
    assert (df["platform"] == "injected").sum() == 20
    assert (df["platform"] != "injected").sum() == len(background)


def test_burst_detection_finds_most_injected_bursts(fake_vectorizer, tmp_path):
    """End-to-end: inject bursts into a small background corpus and confirm
    the benchmark (using a fast, model-free fake vectorizer) detects most
    of them. This is the same pipeline the Kaggle notebook runs on the
    real corpus, just small enough to run in a unit test.
    """
    background = _background_df()
    df, truth = inject_bursts(background, n_bursts=4, posts_per_burst=20, burst_span_hours=4.0, seed=11)

    metrics = evaluate_burst_detection(
        df,
        truth,
        engine_params={"half_life_seconds": 6 * 3600.0, "similarity_threshold": 0.6},
        vectorizer=fake_vectorizer,
        embeddings_path=tmp_path / "embeddings.npy",
        ids_path=tmp_path / "embeddings_ids.npy",
        meta_path=tmp_path / "embeddings_meta.json",
    )

    assert metrics["n_bursts"] == 4
    assert metrics["detection_rate"] >= 0.5
    assert not metrics["per_burst"].empty
