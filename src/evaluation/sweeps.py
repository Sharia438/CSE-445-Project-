"""Parameter sweeps for the dynamic engine and static baseline.

Scores each grid cell on both static/dynamic cluster quality (ARI/NMI
against ``source`` pseudo-labels) and the burst-detection benchmark
(detection rate, precision, latency), so the chosen defaults are picked
from measured numbers instead of guessed. Meant to run on the full corpus
on Kaggle (see ``notebooks/04_kaggle_gpu_pipeline.ipynb``); also runs on a
small local subset for a quick sanity check.
"""

from __future__ import annotations

import itertools
import logging

import pandas as pd

from src.evaluation.burst_benchmark import evaluate_burst_detection
from src.evaluation.injection import inject_bursts
from src.evaluation.metrics import score_dynamic
from src.ml_engine.dynamic_engine import DynamicClusteringEngine, run_replay
from src.ml_engine.vectorizer import TextVectorizer

LOGGER = logging.getLogger(__name__)

DEFAULT_SIMILARITY_GRID = (0.45, 0.55, 0.65, 0.75)
DEFAULT_HALF_LIFE_HOURS_GRID = (6.0, 24.0, 72.0)
DEFAULT_TEXT_MODE_GRID = ("text", "title_text")


def sweep_dynamic_engine(
    df: pd.DataFrame,
    similarity_grid=DEFAULT_SIMILARITY_GRID,
    half_life_hours_grid=DEFAULT_HALF_LIFE_HOURS_GRID,
    text_mode_grid=DEFAULT_TEXT_MODE_GRID,
    n_bursts: int = 6,
    posts_per_burst: int = 60,
    label_column: str = "source",
) -> pd.DataFrame:
    """Grid over (similarity_threshold, half_life_hours, text_mode),
    scoring each cell on cluster quality (ARI/NMI vs ``label_column``) and
    burst-detection performance (recall/precision/latency, via a fresh
    burst injection per cell).

    Returns a tidy DataFrame, one row per grid cell, sorted by a combined
    score (mean of ARI and burst F1) descending - useful for picking new
    engine defaults, and for the sweep heatmap figures in the Kaggle
    notebook.
    """
    vectorizer = TextVectorizer()
    rows = []

    for text_mode in text_mode_grid:
        corpus, truth = inject_bursts(df, n_bursts=n_bursts, posts_per_burst=posts_per_burst)

        for similarity_threshold, half_life_hours in itertools.product(
            similarity_grid, half_life_hours_grid
        ):
            half_life_seconds = half_life_hours * 3600.0
            LOGGER.info(
                "Sweeping similarity=%.2f half_life=%.0fh text_mode=%s",
                similarity_threshold,
                half_life_hours,
                text_mode,
            )

            quality_engine = DynamicClusteringEngine(
                similarity_threshold=similarity_threshold, half_life_seconds=half_life_seconds
            )
            quality_result = run_replay(
                df, vectorizer=vectorizer, engine=quality_engine, text_mode=text_mode
            )
            quality = score_dynamic(quality_result.assignments, df, label_column=label_column)

            burst_metrics = evaluate_burst_detection(
                corpus,
                truth,
                engine_params={
                    "similarity_threshold": similarity_threshold,
                    "half_life_seconds": half_life_seconds,
                },
                vectorizer=vectorizer,
                text_mode=text_mode,
            )

            rows.append(
                {
                    "similarity_threshold": similarity_threshold,
                    "half_life_hours": half_life_hours,
                    "text_mode": text_mode,
                    "ari": quality["ari"],
                    "nmi": quality["nmi"],
                    "n_clusters": quality["n_clusters"],
                    "burst_detection_rate": burst_metrics["detection_rate"],
                    "burst_precision": burst_metrics["precision"],
                    "burst_f1": burst_metrics["f1"],
                    "burst_median_latency_hours": (
                        burst_metrics["median_latency_seconds"] / 3600.0
                        if burst_metrics["median_latency_seconds"] is not None
                        else None
                    ),
                }
            )

    results = pd.DataFrame(rows)
    results["combined_score"] = (results["ari"].clip(lower=0) + results["burst_f1"]) / 2.0
    return results.sort_values("combined_score", ascending=False).reset_index(drop=True)


def best_params(sweep_results: pd.DataFrame) -> dict:
    """Pick the winning row of a sweep and return it as engine kwargs."""
    if sweep_results.empty:
        raise ValueError("sweep_results is empty")
    top = sweep_results.iloc[0]
    return {
        "similarity_threshold": float(top["similarity_threshold"]),
        "half_life_seconds": float(top["half_life_hours"]) * 3600.0,
        "text_mode": str(top["text_mode"]),
    }


if __name__ == "__main__":
    import argparse
    import sys

    from src.ml_engine.vectorizer import DEFAULT_CSV_PATH, load_corpus

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Sweep dynamic-engine parameters.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--max-posts", type=int, default=3000)
    args = parser.parse_args()

    corpus = load_corpus(args.csv_path)
    if len(corpus) > args.max_posts:
        corpus = (
            corpus.assign(_ts=pd.to_datetime(corpus["timestamp"], utc=True, format="ISO8601"))
            .sort_values("_ts")
            .tail(args.max_posts)
            .drop(columns="_ts")
            .reset_index(drop=True)
        )

    results = sweep_dynamic_engine(corpus)
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(results.to_string(index=False))
    print()
    print("Chosen params:", best_params(results))
