"""Tests for the Kaggle-to-Streamlit artifact bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.artifacts import corpus_fingerprint, load_bundle, save_bundle


def test_corpus_fingerprint_is_order_independent():
    df = pd.DataFrame({"post_id": ["a", "b", "c"]})
    assert corpus_fingerprint(df) == corpus_fingerprint(df.iloc[::-1].reset_index(drop=True))


def test_corpus_fingerprint_changes_with_different_ids():
    df1 = pd.DataFrame({"post_id": ["a", "b", "c"]})
    df2 = pd.DataFrame({"post_id": ["a", "b", "d"]})
    assert corpus_fingerprint(df1) != corpus_fingerprint(df2)


def test_load_bundle_returns_none_when_absent(tmp_path):
    assert load_bundle(tmp_path / "does_not_exist") is None


def test_save_and_load_bundle_round_trips_every_piece(tmp_path):
    output_dir = tmp_path / "artifacts"
    embeddings = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    static_labels = pd.DataFrame({"post_id": ["a", "b", "c"], "cluster": [0, 0, 1]})
    dynamic_summary = pd.DataFrame({"cluster_id": [1], "weight": [2.0], "state": ["trending"]})
    evaluation = {"static": {"ari": 0.5}, "burst": {"recall": 1.0}}

    save_bundle(
        output_dir,
        run_meta={"n_posts": 3, "corpus_fingerprint": "abc123"},
        embeddings=embeddings,
        embeddings_ids=["a", "b", "c"],
        embeddings_model_name="fake-model",
        embeddings_text_mode="title_text",
        static_labels=static_labels,
        dynamic_summary=dynamic_summary,
        evaluation=evaluation,
    )

    bundle = load_bundle(output_dir)
    assert bundle is not None
    assert bundle.run_meta["n_posts"] == 3
    assert bundle.run_meta["corpus_fingerprint"] == "abc123"
    np.testing.assert_allclose(bundle.embeddings, embeddings)
    assert bundle.embeddings_ids == ["a", "b", "c"]
    pd.testing.assert_frame_equal(bundle.static_labels, static_labels)
    pd.testing.assert_frame_equal(bundle.dynamic_summary, dynamic_summary)
    assert bundle.evaluation == evaluation


def test_save_bundle_is_incremental(tmp_path):
    """Calling save_bundle twice with different pieces each time (as the
    Kaggle notebook does, one cell per stage) must not erase earlier
    pieces.
    """
    output_dir = tmp_path / "artifacts"
    save_bundle(output_dir, run_meta={"n_posts": 1}, static_labels=pd.DataFrame({"post_id": ["a"], "cluster": [0]}))
    save_bundle(output_dir, run_meta={"n_posts": 1}, dynamic_summary=pd.DataFrame({"cluster_id": [1]}))

    bundle = load_bundle(output_dir)
    assert bundle is not None
    assert bundle.static_labels is not None
    assert bundle.dynamic_summary is not None
