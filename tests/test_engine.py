"""Tests for DynamicClusteringEngine and run_replay.

Covers the two properties the original bug report was measured against:
every state (including "trending") must be reachable, and
``active_cluster_summary`` must only ever report clusters the registry
still considers live (previously it leaked pruned clusters' history).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.ml_engine.dynamic_engine import DynamicClusteringEngine, run_replay

HALF_LIFE = 3600.0


def test_process_post_assigns_similar_posts_to_same_cluster(base_time, make_embedding):
    engine = DynamicClusteringEngine(similarity_threshold=0.8, half_life_seconds=HALF_LIFE)
    c1 = engine.process_post("p1", "post one", base_time, make_embedding(0))
    c2 = engine.process_post("p2", "post two", base_time + timedelta(seconds=1), make_embedding(0))
    c3 = engine.process_post("p3", "different topic", base_time, make_embedding(1))

    assert c1 == c2
    assert c3 != c1


def test_snapshot_grid_makes_declining_reachable(base_time, make_embedding):
    """A single cluster that receives no further posts must eventually
    show up as "declining" once enough time passes for its grid-sampled
    weight history to reflect the decay - this was unreachable before the
    fix, when velocity was only sampled on reinforcement.
    """
    engine = DynamicClusteringEngine(
        half_life_seconds=HALF_LIFE,
        snapshot_interval_seconds=HALF_LIFE / 4,
        min_cluster_weight=1e-6,
    )
    engine.process_post("p1", "one-off topic", base_time, make_embedding(0))

    # Feed unrelated posts far away in embedding space, spaced out in time,
    # so the grid advances and re-samples the first cluster's decaying
    # weight without ever reinforcing it.
    t = base_time
    for i in range(1, 12):
        t = t + timedelta(seconds=HALF_LIFE / 2)
        engine.process_post(f"p{i}", f"unrelated {i}", t, make_embedding(100 + i))

    summary = engine.active_cluster_summary(min_weight=0.0)
    assert "declining" in set(summary["state"])


def test_reinforced_cluster_can_reach_trending(base_time, make_embedding):
    engine = DynamicClusteringEngine(
        similarity_threshold=0.8,
        half_life_seconds=HALF_LIFE,
        snapshot_interval_seconds=60.0,
        trending_growth=0.25,
    )
    t = base_time
    for i in range(30):
        engine.process_post(f"p{i}", "bursting topic", t, make_embedding(0))
        t = t + timedelta(seconds=30)

    summary = engine.active_cluster_summary(min_weight=0.0)
    assert "trending" in set(summary["state"])


def test_active_cluster_summary_excludes_pruned_clusters(base_time, make_embedding):
    engine = DynamicClusteringEngine(
        half_life_seconds=HALF_LIFE,
        snapshot_interval_seconds=1.0,
        min_cluster_weight=0.05,
    )
    engine.process_post("p1", "fading topic", base_time, make_embedding(0))
    # Jump far enough forward (and touch the engine again) that the first
    # cluster decays below min_cluster_weight and gets pruned.
    later = base_time + timedelta(seconds=HALF_LIFE * 20)
    engine.process_post("p2", "unrelated", later, make_embedding(1))

    summary = engine.active_cluster_summary(min_weight=0.0)
    live_ids = set(engine.registry.clusters)
    assert set(summary["cluster_id"]).issubset(live_ids)
    assert len(summary) == len(live_ids)


def test_run_replay_returns_one_assignment_per_post(base_time, fake_vectorizer, tmp_path):
    df = pd.DataFrame(
        {
            "post_id": ["a", "b", "c"],
            "title": ["topic one", "topic one again", "topic two"],
            "text": ["topic one", "topic one again", "topic two"],
            "source": ["s1", "s1", "s2"],
            "platform": ["synthetic", "synthetic", "synthetic"],
            "timestamp": [
                base_time.isoformat(),
                (base_time + timedelta(minutes=1)).isoformat(),
                (base_time + timedelta(minutes=2)).isoformat(),
            ],
        }
    )
    result = run_replay(
        df,
        vectorizer=fake_vectorizer,
        embeddings_path=tmp_path / "embeddings.npy",
        ids_path=tmp_path / "embeddings_ids.npy",
        meta_path=tmp_path / "embeddings_meta.json",
    )
    assert len(result.assignments) == 3
    assert set(result.assignments["post_id"]) == {"a", "b", "c"}


def test_engine_track_state_log_records_transitions(base_time, make_embedding):
    engine = DynamicClusteringEngine(
        similarity_threshold=0.8,
        half_life_seconds=HALF_LIFE,
        snapshot_interval_seconds=30.0,
        track_state_log=True,
    )
    t = base_time
    for i in range(20):
        engine.process_post(f"p{i}", "bursting topic", t, make_embedding(0))
        t = t + timedelta(seconds=30)

    assert len(engine.state_log) > 0
    states_seen = {entry[2] for entry in engine.state_log}
    assert states_seen  # non-empty
