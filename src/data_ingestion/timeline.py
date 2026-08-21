"""Cross-platform timeline handling.

The real historical datasets each cover a different, largely disjoint time
window (Reddit: Aug 2024; Twitter: Dec 2020-Nov 2021; YouTube: 2014-2026),
so a genuine cross-platform trend can never actually co-occur in the data as
collected. Evaluation must run on real timestamps per platform - anything
else would be measuring a fiction. This module provides two explicitly
labelled ways to still show a cross-platform view for demo purposes,
without pretending the timing is real.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def platform_windows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per platform: its real min/max timestamp and post count."""
    working = df.copy()
    working["_ts"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce", format="ISO8601")
    return (
        working.groupby("platform")["_ts"]
        .agg(min_timestamp="min", max_timestamp="max", n_posts="count")
        .reset_index()
    )


def restrict_to_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Intersect every platform's ``[min, max]`` window and return only the
    posts that fall inside the common overlap.

    If the platforms don't actually overlap (true of this project's real
    corpus today), this returns an empty frame - which is itself the
    finding: report it rather than silently treating an empty result as a
    bug.
    """
    windows = platform_windows(df)
    if windows.empty:
        return df.iloc[0:0]

    overlap_start = windows["min_timestamp"].max()
    overlap_end = windows["max_timestamp"].min()
    if overlap_start > overlap_end:
        LOGGER.warning(
            "Platforms do not overlap in time (latest start=%s > earliest end=%s); "
            "restrict_to_overlap() returns 0 rows.",
            overlap_start,
            overlap_end,
        )
        return df.iloc[0:0]

    working = df.copy()
    working["_ts"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce", format="ISO8601")
    mask = (working["_ts"] >= overlap_start) & (working["_ts"] <= overlap_end)
    return df.loc[mask].reset_index(drop=True)


def align_platform_timelines(
    df: pd.DataFrame, target_window_days: float = 30.0
) -> pd.DataFrame:
    """Shift (and optionally rescale) each platform's timestamps onto a
    shared window purely for demo purposes.

    Each platform's posts are linearly rescaled so its own span maps onto
    ``[0, target_window_days]``, all anchored at the same start. This makes
    a "trend spanning multiple platforms" visible in the dashboard, but the
    resulting timestamps are synthetic - the original real timestamp is
    preserved in ``timestamp_original`` and every row is stamped
    ``timeline_mode="aligned"`` so this is never confused with real data.
    Callers doing evaluation should use the real ``timestamp`` column (or
    call this only for the aligned-demo path), never this aligned one.
    """
    working = df.copy()
    working["timestamp_original"] = working["timestamp"]
    working["_ts"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce", format="ISO8601")

    anchor = working["_ts"].min()
    aligned_parts = []
    for platform, group in working.groupby("platform"):
        span = (group["_ts"].max() - group["_ts"].min()).total_seconds()
        target_seconds = target_window_days * 86400.0
        scale = (target_seconds / span) if span > 0 else 1.0
        elapsed = (group["_ts"] - group["_ts"].min()).dt.total_seconds()
        group = group.copy()
        group["_ts"] = anchor + pd.to_timedelta(elapsed * scale, unit="s")
        aligned_parts.append(group)

    aligned = pd.concat(aligned_parts).sort_index()
    aligned["timestamp"] = aligned["_ts"].apply(lambda ts: ts.isoformat() if pd.notna(ts) else None)
    aligned["timeline_mode"] = "aligned"
    return aligned.drop(columns="_ts")
