"""Tests for the vectorized micro-cluster registry.

These exercise exactly the bug this rewrite fixed: decay must be
mathematically exact (half-life halves weight), centroids must stay unit
vectors after updates, and pruned clusters must actually disappear from
both cluster lookup and nearest-match search.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from src.ml_engine.micro_cluster import MicroClusterRegistry

HALF_LIFE = 3600.0  # seconds


def test_decay_halves_weight_after_one_half_life(base_time, make_embedding):
    registry = MicroClusterRegistry()
    # decay_all's first-ever call just establishes the baseline timestamp
    # (there's nothing to decay from yet) - this mirrors process_post,
    # which always decays before adding, so a cluster's "clock" starts
    # from the decay call immediately preceding its own creation.
    registry.decay_all(base_time, HALF_LIFE)
    cluster = registry.add(make_embedding(0), base_time, "first post")
    assert cluster.weight == pytest.approx(1.0)

    registry.decay_all(base_time + timedelta(seconds=HALF_LIFE), HALF_LIFE)
    assert cluster.weight == pytest.approx(0.5, rel=1e-6)


def test_decay_is_multiplicative_across_calls(base_time, make_embedding):
    """Two decay_all calls covering half + half of a half-life should land
    on the same weight as one call covering the full half-life - decay must
    not double-apply or under-apply across incremental steps.
    """
    stepped = MicroClusterRegistry()
    stepped.decay_all(base_time, HALF_LIFE)
    cluster_stepped = stepped.add(make_embedding(0), base_time, "post")
    stepped.decay_all(base_time + timedelta(seconds=HALF_LIFE / 2), HALF_LIFE)
    stepped.decay_all(base_time + timedelta(seconds=HALF_LIFE), HALF_LIFE)

    direct = MicroClusterRegistry()
    direct.decay_all(base_time, HALF_LIFE)
    cluster_direct = direct.add(make_embedding(0), base_time, "post")
    direct.decay_all(base_time + timedelta(seconds=HALF_LIFE), HALF_LIFE)

    assert cluster_stepped.weight == pytest.approx(cluster_direct.weight, rel=1e-9)


def test_decay_before_first_call_is_a_noop(base_time, make_embedding):
    """The very first decay_all call just establishes the baseline
    timestamp - there's no prior state to decay from.
    """
    registry = MicroClusterRegistry()
    cluster = registry.add(make_embedding(0), base_time, "post")
    registry.decay_all(base_time, HALF_LIFE)
    assert cluster.weight == pytest.approx(1.0)


def test_update_keeps_centroid_unit_norm(base_time, make_embedding):
    registry = MicroClusterRegistry()
    cluster = registry.add(make_embedding(0), base_time, "post 1")
    cluster.update(make_embedding(1), base_time + timedelta(minutes=1), "post 2")
    assert np.linalg.norm(cluster.centroid) == pytest.approx(1.0, rel=1e-5)


def test_update_increases_weight_by_one(base_time, make_embedding):
    registry = MicroClusterRegistry()
    cluster = registry.add(make_embedding(0), base_time, "post 1")
    cluster.update(make_embedding(0), base_time + timedelta(minutes=1), "post 2")
    assert cluster.weight == pytest.approx(2.0)
    assert cluster.member_count == 2


def test_find_best_match_prefers_similar_embedding(base_time, make_embedding):
    registry = MicroClusterRegistry()
    registry.add(make_embedding(0), base_time, "topic A")
    registry.add(make_embedding(1), base_time, "topic B")

    best, similarity = registry.find_best_match(make_embedding(0))
    assert best is not None
    assert best.recent_titles[-1] == "topic A"
    assert similarity == pytest.approx(1.0, rel=1e-5)


def test_find_best_match_empty_registry_returns_none(make_embedding):
    registry = MicroClusterRegistry()
    best, similarity = registry.find_best_match(make_embedding(0))
    assert best is None
    assert similarity == 0.0


def test_prune_removes_cluster_from_lookup_and_matching(base_time, make_embedding):
    registry = MicroClusterRegistry()
    registry.decay_all(base_time, HALF_LIFE)
    cluster = registry.add(make_embedding(0), base_time, "dying topic")
    registry.decay_all(base_time + timedelta(seconds=HALF_LIFE * 10), HALF_LIFE)

    forgotten = registry.prune(min_weight=0.05)
    assert forgotten == [cluster.id]
    assert cluster.id not in registry.clusters

    best, similarity = registry.find_best_match(make_embedding(0))
    assert best is None
    assert similarity == 0.0


def test_prune_survives_many_cycles_without_row_corruption(base_time, make_embedding):
    """Repeatedly add short-lived clusters that churn out via pruning,
    past the compaction threshold, while a periodically-reinforced
    survivor should keep correct centroid/weight throughout - this is the
    scenario that would surface a row-remapping bug in ``_compact``.
    """
    registry = MicroClusterRegistry()
    t = base_time
    survivor = registry.add(make_embedding(999), t, "long-lived topic")

    for i in range(200):
        t = t + timedelta(seconds=HALF_LIFE)
        registry.decay_all(t, HALF_LIFE)
        # Reinforcing every cycle after exactly one half-life of decay
        # converges to a steady-state weight of 2.0 (0.5*w + 1.0 = w) -
        # comfortably above the prune threshold.
        survivor.update(make_embedding(999), t, f"reinforcement {i}")
        # Never reinforced again, so this decays below min_weight within a
        # few cycles and gets pruned - exercising churn + compaction.
        registry.add(make_embedding(1000 + i), t, f"short-lived {i}")
        registry.prune(min_weight=0.05)

    assert survivor.id in registry.clusters
    assert survivor.weight > 0.05
    assert np.linalg.norm(survivor.centroid) == pytest.approx(1.0, rel=1e-4)
