"""Artifact bundle: everything Kaggle computes, that Streamlit reads.

Kaggle has the GPU and the full corpus; the local Streamlit dashboard
should not have to re-embed 44k+ posts (or even re-run clustering) just to
render a demo. ``save_bundle`` writes every precomputed piece into one
directory with fixed filenames; ``load_bundle`` reads it back. A
fingerprint over the corpus's post_ids lets the dashboard warn (not fail)
if the bundle looks like it was built from a different dataset than the
one currently on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

BUNDLE_FILENAMES = {
    "run_meta": "run_meta.json",
    "embeddings": "embeddings.npy",
    "embeddings_ids": "embeddings_ids.npy",
    "embeddings_meta": "embeddings_meta.json",
    "static_labels": "static_labels.csv",
    "static_umap2d": "static_umap2d.npy",
    "static_summary": "static_summary.csv",
    "dynamic_assignments": "dynamic_assignments.csv",
    "dynamic_history": "dynamic_history.csv",
    "dynamic_summary": "dynamic_summary.csv",
    "evaluation": "evaluation.json",
}


def corpus_fingerprint(df: pd.DataFrame, id_column: str = "post_id") -> str:
    """Order-independent fingerprint over a corpus's ids - a cheap way to
    detect "this bundle was built from a different dataset" without
    hashing the full text.
    """
    ids = sorted(df[id_column].astype(str).tolist())
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class ArtifactBundle:
    """Everything Kaggle precomputed for the local Streamlit dashboard."""

    run_meta: dict
    embeddings: np.ndarray | None = None
    embeddings_ids: list[str] | None = None
    static_labels: pd.DataFrame | None = None
    static_umap2d: np.ndarray | None = None
    static_summary: pd.DataFrame | None = None
    dynamic_assignments: pd.DataFrame | None = None
    dynamic_history: pd.DataFrame | None = None
    dynamic_summary: pd.DataFrame | None = None
    evaluation: dict = field(default_factory=dict)
    figures_dir: Path | None = None


def save_bundle(
    output_dir: str | Path,
    run_meta: dict,
    embeddings: np.ndarray | None = None,
    embeddings_ids: list[str] | None = None,
    embeddings_model_name: str | None = None,
    embeddings_text_mode: str | None = None,
    static_labels: pd.DataFrame | None = None,
    static_umap2d: np.ndarray | None = None,
    static_summary: pd.DataFrame | None = None,
    dynamic_assignments: pd.DataFrame | None = None,
    dynamic_history: pd.DataFrame | None = None,
    dynamic_summary: pd.DataFrame | None = None,
    evaluation: dict | None = None,
) -> Path:
    """Write every provided artifact into ``output_dir`` under fixed
    filenames. Any argument left ``None`` is skipped, so callers (e.g. the
    Kaggle notebook, which runs the static and dynamic stages as separate
    cells) can build up the bundle incrementally across calls without
    clobbering pieces written by an earlier call.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        **run_meta,
        "generated_at": run_meta.get("generated_at") or datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / BUNDLE_FILENAMES["run_meta"]).write_text(
        json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
    )

    if embeddings is not None and embeddings_ids is not None:
        np.save(output_dir / BUNDLE_FILENAMES["embeddings"], embeddings)
        np.save(output_dir / BUNDLE_FILENAMES["embeddings_ids"], np.array(embeddings_ids))
        meta = {
            "model_name": embeddings_model_name,
            "text_mode": embeddings_text_mode or "text",
            "count": len(embeddings_ids),
        }
        (output_dir / BUNDLE_FILENAMES["embeddings_meta"]).write_text(
            json.dumps(meta), encoding="utf-8"
        )

    if static_labels is not None:
        static_labels.to_csv(output_dir / BUNDLE_FILENAMES["static_labels"], index=False)
    if static_umap2d is not None:
        np.save(output_dir / BUNDLE_FILENAMES["static_umap2d"], static_umap2d)
    if static_summary is not None:
        static_summary.to_csv(output_dir / BUNDLE_FILENAMES["static_summary"], index=False)

    if dynamic_assignments is not None:
        dynamic_assignments.to_csv(output_dir / BUNDLE_FILENAMES["dynamic_assignments"], index=False)
    if dynamic_history is not None:
        dynamic_history.to_csv(output_dir / BUNDLE_FILENAMES["dynamic_history"], index=False)
    if dynamic_summary is not None:
        dynamic_summary.to_csv(output_dir / BUNDLE_FILENAMES["dynamic_summary"], index=False)

    if evaluation is not None:
        (output_dir / BUNDLE_FILENAMES["evaluation"]).write_text(
            json.dumps(evaluation, indent=2, default=str), encoding="utf-8"
        )

    LOGGER.info("Saved artifact bundle to %s", output_dir)
    return output_dir


def load_bundle(input_dir: str | Path) -> ArtifactBundle | None:
    """Read back everything ``save_bundle`` wrote.

    Returns ``None`` if no ``run_meta.json`` is present (i.e. no bundle has
    been generated yet) - callers should treat that as "fall back to local
    compute", never as an error.
    """
    input_dir = Path(input_dir)
    meta_path = input_dir / BUNDLE_FILENAMES["run_meta"]
    if not meta_path.exists():
        return None

    try:
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Could not read %s (%s); treating bundle as absent", meta_path, error)
        return None

    def _read_csv(name: str) -> pd.DataFrame | None:
        path = input_dir / BUNDLE_FILENAMES[name]
        return pd.read_csv(path) if path.exists() else None

    def _read_npy(name: str) -> np.ndarray | None:
        path = input_dir / BUNDLE_FILENAMES[name]
        return np.load(path) if path.exists() else None

    embeddings = _read_npy("embeddings")
    embeddings_ids_path = input_dir / BUNDLE_FILENAMES["embeddings_ids"]
    embeddings_ids = (
        [str(x) for x in np.load(embeddings_ids_path, allow_pickle=True).tolist()]
        if embeddings_ids_path.exists()
        else None
    )

    evaluation_path = input_dir / BUNDLE_FILENAMES["evaluation"]
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
    )

    figures_dir = input_dir / "figures"

    return ArtifactBundle(
        run_meta=run_meta,
        embeddings=embeddings,
        embeddings_ids=embeddings_ids,
        static_labels=_read_csv("static_labels"),
        static_umap2d=_read_npy("static_umap2d"),
        static_summary=_read_csv("static_summary"),
        dynamic_assignments=_read_csv("dynamic_assignments"),
        dynamic_history=_read_csv("dynamic_history"),
        dynamic_summary=_read_csv("dynamic_summary"),
        evaluation=evaluation,
        figures_dir=figures_dir if figures_dir.exists() else None,
    )
