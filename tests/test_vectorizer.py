"""Tests for embedding-text construction and the embedding cache's merge
(rather than overwrite) and invalidation behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml_engine.vectorizer import build_embedding_text, embed_corpus_cached


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_build_embedding_text_mode_text_uses_raw_text():
    df = _df([{"title": "Video title", "text": "a comment"}])
    result = build_embedding_text(df, text_mode="text")
    assert result.iloc[0] == "a comment"


def test_build_embedding_text_mode_title_text_combines_when_different():
    df = _df([{"title": "Video title", "text": "a comment"}])
    result = build_embedding_text(df, text_mode="title_text")
    assert result.iloc[0] == "Video title. a comment"


def test_build_embedding_text_mode_title_text_avoids_duplication_when_same():
    df = _df([{"title": "same text", "text": "same text"}])
    result = build_embedding_text(df, text_mode="title_text")
    assert result.iloc[0] == "same text"


def test_embed_corpus_cached_reuses_cache_for_known_ids(fake_vectorizer, tmp_path):
    df = _df(
        [
            {"post_id": "a", "title": "t", "text": "hello world"},
            {"post_id": "b", "title": "t", "text": "goodbye world"},
        ]
    )
    paths = dict(
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )

    first = embed_corpus_cached(df, vectorizer=fake_vectorizer, **paths)
    second = embed_corpus_cached(df, vectorizer=fake_vectorizer, **paths)

    np.testing.assert_array_equal(first, second)


def test_embed_corpus_cached_merges_new_ids_without_recomputing_old(fake_vectorizer, tmp_path):
    paths = dict(
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )
    initial = _df([{"post_id": "a", "title": "t", "text": "hello world"}])
    embed_corpus_cached(initial, vectorizer=fake_vectorizer, **paths)

    grown = _df(
        [
            {"post_id": "a", "title": "t", "text": "hello world"},
            {"post_id": "b", "title": "t", "text": "a brand new post"},
        ]
    )
    result = embed_corpus_cached(grown, vectorizer=fake_vectorizer, **paths)

    assert result.shape[0] == 2
    cached_ids = np.load(paths["ids_path"], allow_pickle=True).tolist()
    assert set(cached_ids) == {"a", "b"}

    # The row for "a" should be identical to embedding it alone (i.e. it
    # wasn't recomputed with different context/order effects).
    solo = fake_vectorizer.fit_transform(["hello world"])[0]
    np.testing.assert_allclose(result[0], solo, atol=1e-6)


def test_embed_corpus_cached_invalidates_on_text_mode_change(fake_vectorizer, tmp_path):
    paths = dict(
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )
    df = _df([{"post_id": "a", "title": "A Title", "text": "some comment"}])

    embed_corpus_cached(df, vectorizer=fake_vectorizer, text_mode="text", **paths)
    result_title_text = embed_corpus_cached(df, vectorizer=fake_vectorizer, text_mode="title_text", **paths)

    expected = fake_vectorizer.fit_transform(["A Title. some comment"])
    np.testing.assert_allclose(result_title_text, expected, atol=1e-6)


def test_embed_corpus_cached_invalidates_on_model_change(tmp_path):
    from tests.conftest import FakeVectorizer

    paths = dict(
        embeddings_path=tmp_path / "emb.npy",
        ids_path=tmp_path / "emb_ids.npy",
        meta_path=tmp_path / "emb_meta.json",
    )
    df = _df([{"post_id": "a", "title": "t", "text": "some text"}])

    v1 = FakeVectorizer(model_name="model-one")
    v2 = FakeVectorizer(model_name="model-two")

    embed_corpus_cached(df, vectorizer=v1, **paths)
    result_v2 = embed_corpus_cached(df, vectorizer=v2, **paths)

    expected = v2.fit_transform(["some text"])
    np.testing.assert_allclose(result_v2, expected, atol=1e-6)
