"""Semi-synthetic burst injection for burst-detection evaluation.

Real posts serve as background "noise" (the world keeps posting about
everything else while a burst happens), and paraphrased burst topics are
injected at known timestamps. This is the standard protocol for getting
real, defensible detection-latency and precision/recall numbers without
needing to hand-label a real corpus for burst starts, which is both
expensive and something reasonable annotators would disagree on anyway.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src.data_ingestion.historical_loader import paraphrase_template

LOGGER = logging.getLogger(__name__)

# Six distinct, mutually dissimilar burst topics so `n_bursts` can go up to
# 6 without reusing a topic label.
BURST_TOPIC_TEMPLATES: dict[str, list[str]] = {
    "burst_chip_shortage": [
        "New chip shortage disrupts global electronics supply chain",
        "Massive shortage of semiconductor chips hits manufacturers",
        "Tech industry reveals surprise chip shortage crisis",
        "Chip shortage sparks record price hikes across devices",
    ],
    "burst_data_leak": [
        "Massive data leak exposes millions of user records",
        "New data breach sparks fresh privacy concerns",
        "Company reveals shocking data leak affecting users worldwide",
        "Security researchers unveil record-breaking data breach",
    ],
    "burst_merger": [
        "Two tech giants announce surprise merger deal",
        "Massive merger reshapes industry landscape overnight",
        "Company reveals record-breaking acquisition plan",
        "New merger sparks antitrust review fears",
    ],
    "burst_recall": [
        "Automaker announces massive vehicle recall",
        "New safety recall affects millions of units",
        "Company reveals surprise product recall after complaints",
        "Recall sparks investigation into manufacturing defects",
    ],
    "burst_launch": [
        "Company unveils surprise new flagship product",
        "New product launch breaks preorder records",
        "Massive hype builds ahead of surprise product reveal",
        "Launch event sparks record online engagement",
    ],
    "burst_outage": [
        "Massive outage takes down popular platform for hours",
        "New service outage sparks user frustration nationwide",
        "Company reveals cause of shocking system outage",
        "Outage breaks records for longest platform downtime",
    ],
}


def inject_bursts(
    background_df: pd.DataFrame,
    n_bursts: int = 6,
    posts_per_burst: int = 80,
    burst_span_hours: float = 6.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject ``n_bursts`` paraphrased burst topics into ``background_df``
    at known, staggered timestamps.

    Returns ``(df, truth_df)``:

    - ``df``: ``background_df`` plus the injected posts, sorted by
      timestamp, same schema (``post_id``, ``title``, ``text``, ``source``,
      ``platform``, ``timestamp``); injected rows have
      ``platform="injected"`` and ``source=<topic_label>``.
    - ``truth_df``: one row per injected burst - ``burst_id``,
      ``topic_label``, ``burst_start``, ``peak``, ``burst_end``,
      ``n_posts`` - the ground truth ``burst_benchmark.py`` scores against.
    """
    rng = np.random.default_rng(seed)
    background_df = background_df.copy()
    background_df["timestamp"] = pd.to_datetime(
        background_df["timestamp"], utc=True, format="ISO8601"
    )

    topics = list(BURST_TOPIC_TEMPLATES.items())
    if n_bursts > len(topics):
        raise ValueError(f"n_bursts={n_bursts} exceeds the {len(topics)} available burst topics")
    chosen = [topics[i] for i in rng.permutation(len(topics))[:n_bursts]]

    background_start = background_df["timestamp"].min()
    background_end = background_df["timestamp"].max()
    span_seconds = (background_end - background_start).total_seconds()
    if not np.isfinite(span_seconds) or span_seconds <= 0:
        raise ValueError(
            "background_df must span a positive amount of time (got "
            f"start={background_start!r}, end={background_end!r} - empty or "
            "all-NaT timestamps?)"
        )

    # Spread burst start times evenly across the middle 80% of the
    # background window, so there's real background both before and after
    # each burst.
    anchors = np.linspace(0.1, 0.9, n_bursts) * span_seconds

    injected_rows = []
    truth_rows = []
    beta_a, beta_b = 1.5, 4.0
    beta_mean_fraction = beta_a / (beta_a + beta_b)

    for burst_id, (anchor_seconds, (topic_label, templates)) in enumerate(zip(anchors, chosen)):
        burst_start = background_start + timedelta(seconds=float(anchor_seconds))
        offsets = rng.beta(a=beta_a, b=beta_b, size=posts_per_burst) * burst_span_hours * 3600

        for i, offset_seconds in enumerate(sorted(offsets)):
            template = templates[rng.integers(0, len(templates))]
            template = paraphrase_template(template, rng)
            timestamp = burst_start + timedelta(seconds=float(offset_seconds))
            injected_rows.append(
                {
                    "post_id": f"injected_{burst_id}_{i}",
                    "title": template,
                    "text": template,
                    "source": topic_label,
                    "platform": "injected",
                    "timestamp": timestamp,
                }
            )

        truth_rows.append(
            {
                "burst_id": burst_id,
                "topic_label": topic_label,
                "burst_start": burst_start,
                "peak": burst_start + timedelta(hours=burst_span_hours * beta_mean_fraction),
                "burst_end": burst_start + timedelta(hours=burst_span_hours),
                "n_posts": posts_per_burst,
            }
        )

    injected_df = pd.DataFrame(injected_rows)
    combined = (
        pd.concat([background_df, injected_df], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined["timestamp"] = combined["timestamp"].apply(
        lambda ts: ts.isoformat() if pd.notna(ts) else None
    )

    truth_df = pd.DataFrame(truth_rows)
    LOGGER.info(
        "Injected %d bursts (%d posts each) into %d background posts",
        n_bursts,
        posts_per_burst,
        len(background_df),
    )
    return combined, truth_df
