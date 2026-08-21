"""Assemble evaluation results (static/dynamic cluster quality, burst
detection, sweep results) into the JSON-serializable shape the artifact
bundle exports as ``evaluation.json``, plus the per-table CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_evaluation_report(
    static_metrics: dict | None = None,
    dynamic_metrics: dict | None = None,
    burst_metrics: dict | None = None,
    sliding_window_metrics: dict | None = None,
    sweep_results: pd.DataFrame | None = None,
    per_platform: pd.DataFrame | None = None,
) -> dict:
    """Bundle every evaluation result into one JSON-safe dict.

    DataFrame-valued fields (``burst_metrics["per_burst"]``,
    ``sliding_window_metrics["per_burst"]``, ``sweep_results``,
    ``per_platform``) are excluded from the JSON dict and are expected to
    be written as separate CSVs by ``save_evaluation_csvs`` - JSON is for
    scalars/summaries, CSVs are for tables.

    ``sliding_window_metrics`` (from
    ``sliding_window_baseline.evaluate_burst_detection_sliding_window``)
    is the periodic-recluster baseline scored on the same burst benchmark
    as the dynamic engine - the direct "is online actually better than
    just re-running HDBSCAN on a rolling window" comparison.
    """
    report: dict = {}
    if static_metrics is not None:
        report["static"] = static_metrics
    if dynamic_metrics is not None:
        report["dynamic"] = dynamic_metrics
    if burst_metrics is not None:
        report["burst"] = {k: v for k, v in burst_metrics.items() if k != "per_burst"}
    if sliding_window_metrics is not None:
        report["sliding_window_burst"] = {
            k: v for k, v in sliding_window_metrics.items() if k != "per_burst"
        }
    if sweep_results is not None and not sweep_results.empty:
        report["sweep_best"] = sweep_results.iloc[0].to_dict()
    return report


def save_evaluation_csvs(
    output_dir: str | Path,
    burst_metrics: dict | None = None,
    sliding_window_metrics: dict | None = None,
    sweep_results: pd.DataFrame | None = None,
    per_platform: pd.DataFrame | None = None,
) -> list[Path]:
    """Write the tabular parts of an evaluation run as CSVs under
    ``output_dir``. Returns the list of paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if burst_metrics is not None and "per_burst" in burst_metrics:
        path = output_dir / "eval_bursts.csv"
        burst_metrics["per_burst"].to_csv(path, index=False)
        written.append(path)

    if sliding_window_metrics is not None and "per_burst" in sliding_window_metrics:
        path = output_dir / "eval_bursts_sliding_window.csv"
        sliding_window_metrics["per_burst"].to_csv(path, index=False)
        written.append(path)

    if sweep_results is not None and not sweep_results.empty:
        path = output_dir / "sweep_results.csv"
        sweep_results.to_csv(path, index=False)
        written.append(path)

    if per_platform is not None and not per_platform.empty:
        path = output_dir / "eval_per_platform.csv"
        per_platform.to_csv(path, index=False)
        written.append(path)

    return written
