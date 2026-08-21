"""Static (offline) clustering baseline: UMAP dimensionality reduction +
HDBSCAN / DBSCAN over post embeddings.

This is the Week 3-4 baseline referenced in the README roadmap. It runs once
over a snapshot of ``data/master_dataset.csv`` and groups semantically
similar posts together. Phase 3's dynamic engine builds on top of this with
online, time-decayed micro-clustering.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.ml_engine.vectorizer import (
    DEFAULT_CSV_PATH,
    TextVectorizer,
    embed_corpus_cached,
    load_corpus,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_UMAP_COMPONENTS = 5
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 5
DEFAULT_DBSCAN_EPS = 0.5
NOISE_LABEL = -1


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = DEFAULT_UMAP_COMPONENTS,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional sentence embeddings into a lower-dimensional
    space that is friendlier to density-based clustering.
    """
    import umap

    n_components = min(n_components, max(embeddings.shape[0] - 2, 2))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(embeddings.shape[0] - 1, 2)),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN. Returns integer labels, -1 = noise."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


def cluster_dbscan(
    embeddings: np.ndarray,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """Cluster embeddings with DBSCAN as a simpler baseline for comparison."""
    from sklearn.cluster import DBSCAN

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return clusterer.fit_predict(embeddings)


def cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    text_column: str = "title",
    samples_per_cluster: int = 3,
) -> pd.DataFrame:
    """Summarize cluster sizes and a few sample titles per cluster.

    Clusters are sorted by size (descending), with noise (-1) listed last.
    """
    summary_df = df.copy()
    summary_df["cluster"] = labels

    rows = []
    for cluster_id, group in summary_df.groupby("cluster"):
        samples = group[text_column].head(samples_per_cluster).tolist()
        rows.append(
            {
                "cluster": cluster_id,
                "size": len(group),
                "sample_titles": " | ".join(samples),
            }
        )

    result = pd.DataFrame(rows)
    is_noise = result["cluster"] == NOISE_LABEL
    result = pd.concat(
        [
            result[~is_noise].sort_values("size", ascending=False),
            result[is_noise],
        ],
        ignore_index=True,
    )
    return result


def run_static_pipeline(
    csv_path: str = str(DEFAULT_CSV_PATH),
    df: pd.DataFrame | None = None,
    vectorizer: TextVectorizer | None = None,
    method: str = "hdbscan",
    use_umap: bool = True,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps: float = DEFAULT_DBSCAN_EPS,
    text_mode: str = "text",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """End-to-end baseline: load corpus -> embed -> (optional UMAP) -> cluster.

    Pass ``df`` directly (e.g. a demo subset already loaded by a caller like
    the dashboard) to skip re-reading ``csv_path`` from disk.

    Returns:
        df: the loaded corpus (post-dedup/cleaning)
        embeddings: the raw sentence embeddings (before UMAP)
        labels: cluster label per row in ``df``
    """
    if method not in ("hdbscan", "dbscan"):
        raise ValueError(f"Unknown clustering method: {method!r}")

    if df is None:
        df = load_corpus(csv_path)
        LOGGER.info("Loaded %d rows from %s", len(df), csv_path)

    embeddings = embed_corpus_cached(df, vectorizer=vectorizer, text_mode=text_mode)
    LOGGER.info("Embedded corpus into shape %s", embeddings.shape)

    cluster_input = embeddings
    if use_umap:
        cluster_input = reduce_umap(embeddings, n_components=umap_components)
        LOGGER.info("Reduced embeddings to shape %s via UMAP", cluster_input.shape)

    if method == "hdbscan":
        labels = cluster_hdbscan(
            cluster_input, min_cluster_size=min_cluster_size, min_samples=min_samples
        )
    else:
        labels = cluster_dbscan(cluster_input, eps=eps, min_samples=min_samples)

    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    noise_ratio = float(np.mean(labels == NOISE_LABEL))
    LOGGER.info(
        "Found %d clusters (%.1f%% noise) using %s", n_clusters, noise_ratio * 100, method
    )

    return df, embeddings, labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the static clustering baseline.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--method", choices=["hdbscan", "dbscan"], default="hdbscan")
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP reduction")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--text-mode", choices=["text", "title_text"], default="text")
    return parser.parse_args()


def main() -> None:
    # Real multi-platform text (Twitter/YouTube/Reddit) can contain
    # non-Latin-1 characters that Windows' default console encoding
    # (cp1252) can't display; degrade to '?' instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    df, _embeddings, labels = run_static_pipeline(
        csv_path=args.csv_path,
        method=args.method,
        use_umap=not args.no_umap,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        eps=args.eps,
        text_mode=args.text_mode,
    )

    summary = cluster_summary(df, labels)
    with pd.option_context("display.max_colwidth", 80):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
