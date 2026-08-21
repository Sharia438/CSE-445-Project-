"""Evaluation: cluster-quality metrics and burst-detection benchmarking."""

from src.evaluation.burst_benchmark import evaluate_burst_detection
from src.evaluation.injection import inject_bursts
from src.evaluation.metrics import cluster_quality, embedding_quality, score_dynamic, score_static
from src.evaluation.sliding_window_baseline import (
    evaluate_burst_detection_sliding_window,
    run_sliding_window,
)
from src.evaluation.sweeps import best_params, sweep_dynamic_engine

__all__ = [
    "cluster_quality",
    "embedding_quality",
    "score_static",
    "score_dynamic",
    "inject_bursts",
    "evaluate_burst_detection",
    "run_sliding_window",
    "evaluate_burst_detection_sliding_window",
    "sweep_dynamic_engine",
    "best_params",
]
