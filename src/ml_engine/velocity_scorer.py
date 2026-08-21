"""Velocity / burst scoring for micro-clusters.

Tracks a rolling history of each cluster's decayed weight, sampled on a
regular time grid by the engine (not only when a cluster is reinforced -
otherwise decay between reinforcements is never observed, and "declining"
can never fire). Converts that history into a dimensionless growth rate and
a burst z-score, which is what lets the dashboard show "rising" vs
"trending" vs "declining" vs "dormant" topics in a way that means the same
thing whether the underlying corpus spans 48 simulated hours or several
real months.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

State = Literal["dormant", "rising", "trending", "declining"]

DEFAULT_WINDOW_SIZE = 20
DEFAULT_TRENDING_GROWTH = 0.25  # fractional growth per half-life
DEFAULT_MIN_WEIGHT = 0.5
DEFAULT_BURST_Z_THRESHOLD = 2.5
_GROWTH_EPS = 1e-9

# Kept for anyone importing the old name; prefer DEFAULT_TRENDING_GROWTH.
DEFAULT_TRENDING_THRESHOLD = DEFAULT_TRENDING_GROWTH


@dataclass
class _Snapshot:
    timestamp: datetime
    weight: float


class VelocityTracker:
    """Maintains a bounded weight history per cluster and derives a
    dimensionless growth rate, burst z-score, and trending state from it.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._history: dict[int, deque[_Snapshot]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._growth_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, cluster_id: int, timestamp: datetime, weight: float) -> None:
        history = self._history[cluster_id]
        if history:
            prev = history[-1]
            elapsed = (timestamp - prev.timestamp).total_seconds()
            if elapsed > 0:
                step_growth = (weight - prev.weight) / max(prev.weight, _GROWTH_EPS)
                self._growth_history[cluster_id].append(step_growth)
        history.append(_Snapshot(timestamp, weight))

    def forget(self, cluster_ids) -> None:
        """Drop tracked history for clusters that no longer exist (pruned
        by the registry) - otherwise history accumulates forever and
        ``snapshot()`` keeps reporting dead clusters.
        """
        for cluster_id in cluster_ids:
            self._history.pop(cluster_id, None)
            self._growth_history.pop(cluster_id, None)

    def tracked_cluster_ids(self) -> list[int]:
        return list(self._history)

    def compute_velocity(self, cluster_id: int) -> float:
        """Slope of weight over the tracked window, in weight-units/second.
        Returns 0.0 if there isn't enough history yet. Kept alongside the
        dimensionless growth rate for continuity with existing figures.
        """
        history = self._history.get(cluster_id)
        if not history or len(history) < 2:
            return 0.0

        earliest, latest = history[0], history[-1]
        elapsed = (latest.timestamp - earliest.timestamp).total_seconds()
        if elapsed <= 0:
            return 0.0

        return (latest.weight - earliest.weight) / elapsed

    def compute_growth_rate(self, cluster_id: int, half_life_seconds: float) -> float:
        """Fractional change in weight over the tracked window, normalized
        to "per half-life" units.

        A cluster receiving no new posts decays to
        ``weight * 2 ** (-elapsed / half_life)``, so over exactly one
        half-life this returns -0.5 by construction - that's the
        "abandoned" baseline. ``0.0`` means the cluster is exactly
        replacing its own decay (steady state); positive means genuine
        reinforcement. Unlike a raw weight/second slope, this reads the
        same way on a 48-hour synthetic corpus and a month-long real one.
        """
        history = self._history.get(cluster_id)
        if not history or len(history) < 2:
            return 0.0

        earliest, latest = history[0], history[-1]
        elapsed = (latest.timestamp - earliest.timestamp).total_seconds()
        if elapsed <= 0:
            return 0.0

        fractional = (latest.weight - earliest.weight) / max(earliest.weight, _GROWTH_EPS)
        return fractional * (half_life_seconds / elapsed)

    def compute_burst_z(self, cluster_id: int) -> float:
        """Z-score of the cluster's most recent step-to-step growth against
        its own trailing distribution of growth steps. ``0.0`` if there
        isn't enough history to form a distribution.
        """
        growth = self._growth_history.get(cluster_id)
        if not growth or len(growth) < 3:
            return 0.0

        series = pd.Series(growth)
        std = series.std()
        if not std or std < _GROWTH_EPS:
            return 0.0

        return float((series.iloc[-1] - series.mean()) / std)

    def latest_weight(self, cluster_id: int) -> float:
        history = self._history.get(cluster_id)
        return history[-1].weight if history else 0.0

    @staticmethod
    def classify_state(
        growth_rate: float,
        weight: float,
        burst_z: float = 0.0,
        trending_growth: float = DEFAULT_TRENDING_GROWTH,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        burst_z_threshold: float = DEFAULT_BURST_Z_THRESHOLD,
    ) -> State:
        """Threshold-based state machine over the dimensionless growth rate.

        - trending: growing fast (by an absolute growth-rate cutoff, OR by
          a burst z-score relative to the cluster's own recent history)
          and already has meaningful volume
        - rising: growing, but not yet fast/large enough to call "trending"
        - declining: shrinking
        - dormant: flat/negligible, or no usable history
        """
        is_bursting = growth_rate >= trending_growth or burst_z >= burst_z_threshold
        if is_bursting and weight >= min_weight:
            return "trending"
        if growth_rate > _GROWTH_EPS:
            return "rising"
        if growth_rate < -_GROWTH_EPS:
            return "declining"
        return "dormant"

    def snapshot(
        self,
        half_life_seconds: float,
        trending_growth: float = DEFAULT_TRENDING_GROWTH,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        burst_z_threshold: float = DEFAULT_BURST_Z_THRESHOLD,
        auto_calibrate: bool = False,
    ) -> pd.DataFrame:
        """Current velocity/growth-rate/state per tracked cluster.

        With ``auto_calibrate=True``, the trending growth-rate cutoff is
        replaced by the 95th percentile of growth rates among clusters at
        or above ``min_weight`` - useful for a live demo where a single
        fixed threshold may not suit whatever corpus happens to be loaded.
        """
        rows = []
        for cluster_id in self._history:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "weight": self.latest_weight(cluster_id),
                    "velocity": self.compute_velocity(cluster_id),
                    "growth_rate": self.compute_growth_rate(cluster_id, half_life_seconds),
                    "burst_z": self.compute_burst_z(cluster_id),
                }
            )

        df = pd.DataFrame(rows, columns=["cluster_id", "weight", "velocity", "growth_rate", "burst_z"])
        if df.empty:
            return df.assign(state=pd.Series(dtype="object"))

        effective_cutoff = trending_growth
        if auto_calibrate:
            eligible = df.loc[df["weight"] >= min_weight, "growth_rate"]
            if len(eligible) >= 5:
                effective_cutoff = max(trending_growth, float(eligible.quantile(0.95)))

        df["state"] = [
            self.classify_state(
                row.growth_rate,
                row.weight,
                row.burst_z,
                effective_cutoff,
                min_weight,
                burst_z_threshold,
            )
            for row in df.itertuples()
        ]
        return df

    def history_df(self, cluster_id: int) -> pd.DataFrame:
        """Full (timestamp, weight) history for a single cluster, useful for
        plotting.
        """
        history = self._history.get(cluster_id, [])
        return pd.DataFrame(
            {
                "timestamp": [snap.timestamp for snap in history],
                "weight": [snap.weight for snap in history],
            }
        )
