"""Tests for the dimensionless growth-rate velocity scoring.

The key property under test: a cluster receiving no new posts must show
``growth_rate == -0.5`` after exactly one half-life (by construction of
the decay formula), and that must classify as ``declining`` - this is the
state the pre-fix engine could never reach because it only sampled weight
right after a reinforcement.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.ml_engine.velocity_scorer import VelocityTracker

HALF_LIFE = 3600.0


def test_growth_rate_is_zero_with_insufficient_history(base_time):
    tracker = VelocityTracker()
    assert tracker.compute_growth_rate(cluster_id=1, half_life_seconds=HALF_LIFE) == 0.0
    tracker.record(1, base_time, weight=1.0)
    assert tracker.compute_growth_rate(cluster_id=1, half_life_seconds=HALF_LIFE) == 0.0


def test_pure_decay_yields_minus_half_growth_rate_and_declining(base_time):
    tracker = VelocityTracker()
    tracker.record(1, base_time, weight=1.0)
    # Weight after exactly one half-life with no reinforcement, per the
    # decay formula in micro_cluster.MicroClusterRegistry.decay_all.
    tracker.record(1, base_time + timedelta(seconds=HALF_LIFE), weight=0.5)

    growth = tracker.compute_growth_rate(1, half_life_seconds=HALF_LIFE)
    assert growth == pytest.approx(-0.5, rel=1e-6)
    assert tracker.classify_state(growth, weight=0.5) == "declining"


def test_reinforced_cluster_has_positive_growth_rate(base_time):
    tracker = VelocityTracker()
    tracker.record(1, base_time, weight=1.0)
    # Growing faster than decay would shrink it.
    tracker.record(1, base_time + timedelta(seconds=HALF_LIFE), weight=3.0)

    growth = tracker.compute_growth_rate(1, half_life_seconds=HALF_LIFE)
    assert growth > 0
    assert tracker.classify_state(growth, weight=3.0) == "trending"


def test_slow_growth_is_rising_not_trending(base_time):
    tracker = VelocityTracker()
    tracker.record(1, base_time, weight=1.0)
    tracker.record(1, base_time + timedelta(seconds=HALF_LIFE), weight=1.05)

    growth = tracker.compute_growth_rate(1, half_life_seconds=HALF_LIFE)
    assert 0 < growth < 0.25
    assert tracker.classify_state(growth, weight=1.05, trending_growth=0.25) == "rising"


def test_forget_drops_history(base_time):
    tracker = VelocityTracker()
    tracker.record(1, base_time, weight=1.0)
    tracker.record(2, base_time, weight=1.0)

    tracker.forget([1])

    assert tracker.tracked_cluster_ids() == [2]
    assert tracker.compute_growth_rate(1, half_life_seconds=HALF_LIFE) == 0.0
    assert tracker.latest_weight(1) == 0.0


def test_snapshot_reports_every_tracked_cluster(base_time):
    tracker = VelocityTracker()
    tracker.record(1, base_time, weight=1.0)
    tracker.record(1, base_time + timedelta(seconds=HALF_LIFE), weight=0.5)
    tracker.record(2, base_time, weight=1.0)
    tracker.record(2, base_time + timedelta(seconds=HALF_LIFE), weight=4.0)

    snapshot = tracker.snapshot(half_life_seconds=HALF_LIFE)
    assert set(snapshot["cluster_id"]) == {1, 2}
    states = dict(zip(snapshot["cluster_id"], snapshot["state"]))
    assert states[1] == "declining"
    assert states[2] == "trending"


def test_snapshot_empty_tracker_returns_empty_frame_with_state_column():
    tracker = VelocityTracker()
    snapshot = tracker.snapshot(half_life_seconds=HALF_LIFE)
    assert snapshot.empty
    assert "state" in snapshot.columns
