"""Clustering-quality metrics.

Pseudo-labels come from the existing ``source`` column (subreddit /
YouTube topic / synthetic topic label) - a free, imperfect but genuinely
informative ground truth for scoring both the static and dynamic
clustering paths against, without needing any manual annotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

NOISE_LABEL = -1
_SILHOUETTE_MAX_SAMPLES = 5000


def cluster_quality(labels_true, labels_pred) -> dict:
    """ARI / NMI / homogeneity / completeness / V-measure between
    predicted cluster ids and pseudo-labels, plus cluster count and noise
    ratio for context.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    n_clusters = len(set(labels_pred.tolist())) - (1 if NOISE_LABEL in labels_pred else 0)
    noise_ratio = float(np.mean(labels_pred == NOISE_LABEL)) if len(labels_pred) else 0.0

    return {
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "v_measure": float(v_measure_score(labels_true, labels_pred)),
        "n_clusters": int(n_clusters),
        "n_posts": int(len(labels_pred)),
        "noise_ratio": noise_ratio,
    }


def embedding_quality(embeddings: np.ndarray, labels, seed: int = 42) -> dict:
    """Cosine silhouette score over non-noise points, subsampled for
    tractability on large corpora. Returns ``silhouette: None`` if there
    are too few points/clusters to score.
    """
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return {"silhouette": None, "n_scored": int(mask.sum())}

    emb = np.asarray(embeddings)[mask]
    lbl = labels[mask]
    if emb.shape[0] > _SILHOUETTE_MAX_SAMPLES:
        rng = np.random.default_rng(seed)
        idx = rng.choice(emb.shape[0], size=_SILHOUETTE_MAX_SAMPLES, replace=False)
        emb = emb[idx]
        lbl = lbl[idx]

    score = silhouette_score(emb, lbl, metric="cosine")
    return {"silhouette": float(score), "n_scored": int(emb.shape[0])}


def score_static(
    df: pd.DataFrame,
    labels,
    embeddings: np.ndarray | None = None,
    label_column: str = "source",
) -> dict:
    """Score a static-clustering run's labels against ``df[label_column]``
    pseudo-labels, optionally adding silhouette when ``embeddings`` is
    given.
    """
    result = cluster_quality(df[label_column].tolist(), labels)
    if embeddings is not None:
        result.update(embedding_quality(embeddings, labels))
    return result


def score_dynamic(
    assignments: pd.DataFrame, df: pd.DataFrame, label_column: str = "source"
) -> dict:
    """Score the dynamic engine's per-post cluster assignments against
    pseudo-labels.

    ``assignments`` is ``run_replay(...).assignments`` (columns:
    ``post_id``, ``cluster_id``, ``timestamp``); ``df`` must contain
    ``post_id`` and ``label_column``.
    """
    merged = assignments.merge(df[["post_id", label_column]], on="post_id", how="inner")
    return cluster_quality(merged[label_column].tolist(), merged["cluster_id"].tolist())
