"""Text loading and sentence-embedding utilities for the trend detector.

Turns raw rows from ``data/master_dataset.csv`` into dense embedding vectors
that downstream clustering (``static_clustering.py``) can consume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "master_dataset.csv"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ["post_id", "title", "text", "source", "platform", "timestamp"]

DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
DEFAULT_EMBEDDINGS_IDS_PATH = PROJECT_ROOT / "data" / "embeddings_ids.npy"
DEFAULT_EMBEDDINGS_META_PATH = PROJECT_ROOT / "data" / "embeddings_meta.json"


def _auto_device() -> str:
    """Pick ``cuda`` when available (e.g. a Kaggle T4 notebook), else ``cpu``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_embedding_text(df: pd.DataFrame, text_mode: str = "text") -> pd.Series:
    """Build the text series that gets embedded.

    ``"text"`` embeds the raw ``text`` column as-is. ``"title_text"``
    concatenates ``title`` and ``text`` when they differ - this matters for
    platforms like YouTube where ``text`` is just the comment body and
    ``title`` is the video it was posted under; embedding the comment alone
    throws away the topic context and clusters by comment wording instead
    of subject matter.
    """
    if text_mode not in ("text", "title_text"):
        raise ValueError(f"Unknown text_mode: {text_mode!r}")

    text = df["text"].fillna("").astype(str)
    if text_mode == "text":
        return text

    title = df["title"].fillna("").astype(str)
    combined = np.where(
        (title != "") & (title != text),
        title + ". " + text,
        text,
    )
    return pd.Series(combined, index=df.index)


def load_corpus(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load and validate the master dataset, dropping rows with empty text.

    Raises:
        FileNotFoundError: if ``csv_path`` does not exist (e.g. the Reddit
            streamer or historical loader has not been run yet).
        ValueError: if required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run "
            "`python src/data_ingestion/reddit_streamer.py` (or the "
            "historical loader) to populate it first."
        )

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {csv_path} is missing columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    dropped = before - len(df)
    if dropped:
        LOGGER.info("Dropped %d empty/duplicate rows from %s", dropped, csv_path)

    return df


class TextVectorizer:
    """Thin wrapper around a `sentence-transformers` model.

    Lazily loads the underlying model on first use so importing this module
    stays cheap.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # None means "auto-detect": cuda on a GPU box (e.g. Kaggle T4), cpu
        # otherwise. Same code path works unmodified on both.
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device or _auto_device()
            LOGGER.info(
                "Loading sentence-transformer model: %s (device=%s)", self.model_name, device
            )
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors."""
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors."""
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def save(embeddings: np.ndarray, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        return np.load(Path(path))


def embed_corpus(
    df: pd.DataFrame,
    vectorizer: TextVectorizer | None = None,
    text_column: str = "text",
    text_mode: str = "text",
) -> np.ndarray:
    """Convenience helper: embed a corpus DataFrame's text.

    ``text_mode="title_text"`` overrides ``text_column`` and embeds
    ``title + text`` combined (see ``build_embedding_text``).
    """
    vectorizer = vectorizer or TextVectorizer()
    if text_mode == "title_text":
        texts = build_embedding_text(df, text_mode=text_mode)
    else:
        texts = df[text_column].fillna("").astype(str)
    return vectorizer.fit_transform(texts.tolist())


def _load_cache(
    ids_path: Path, meta_path: Path
) -> tuple[list[str], dict] | None:
    try:
        cached_ids = [str(x) for x in np.load(ids_path, allow_pickle=True).tolist()]
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        LOGGER.warning("Ignoring unreadable embedding cache (%s)", error)
        return None
    return cached_ids, meta


def _resolve_texts(df: pd.DataFrame, text_column: str, text_mode: str) -> pd.Series:
    if text_mode == "title_text":
        return build_embedding_text(df, text_mode=text_mode)
    return df[text_column].fillna("").astype(str)


def _save_cache(
    embeddings_path: Path,
    ids_path: Path,
    meta_path: Path,
    embeddings: np.ndarray,
    ids: list[str],
    model_name: str,
    text_mode: str,
) -> None:
    try:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        np.save(ids_path, np.array(ids))
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump({"model_name": model_name, "text_mode": text_mode, "count": len(ids)}, fh)
        LOGGER.info("Cached %d embeddings to %s", len(ids), embeddings_path)
    except OSError as error:
        # Kaggle input datasets are read-only; still return the vectors.
        LOGGER.warning("Could not write embedding cache to %s (%s)", embeddings_path, error)


def embed_corpus_cached(
    df: pd.DataFrame,
    vectorizer: TextVectorizer | None = None,
    text_column: str = "text",
    id_column: str = "post_id",
    text_mode: str = "text",
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    ids_path: str | Path = DEFAULT_EMBEDDINGS_IDS_PATH,
    meta_path: str | Path = DEFAULT_EMBEDDINGS_META_PATH,
) -> np.ndarray:
    """Embed ``df``'s text, reusing a cached ``.npy`` file whenever possible
    instead of re-running the sentence-transformer model.

    This is what lets a Streamlit click reuse the same 45k-post embeddings
    instead of re-encoding them every time, and what lets an
    ``embeddings.npy`` computed on a Kaggle GPU notebook be dropped into
    ``data/`` and reused locally with no GPU at all.

    The cache is valid whenever the cached model name *and* ``text_mode``
    both match. If some of ``df``'s ids are missing from the cache (e.g. new
    posts were ingested since the cache was built), only those rows are
    embedded and merged into the cache on disk - the whole cache is never
    thrown away just because the corpus grew, which previously forced a
    full re-embed after every "max posts" slider change in the dashboard.
    A model or ``text_mode`` change invalidates the cache entirely, since
    old vectors aren't comparable to new ones.
    """
    vectorizer = vectorizer or TextVectorizer()
    embeddings_path = Path(embeddings_path)
    ids_path = Path(ids_path)
    meta_path = Path(meta_path)

    ids = df[id_column].astype(str).tolist()

    if embeddings_path.exists() and ids_path.exists() and meta_path.exists():
        cached = _load_cache(ids_path, meta_path)
        if cached is not None:
            cached_ids, meta = cached
            same_model = meta.get("model_name") == vectorizer.model_name
            same_text_mode = meta.get("text_mode", "text") == text_mode
            if same_model and same_text_mode:
                cached_id_set = set(cached_ids)
                missing_ids = [pid for pid in ids if pid not in cached_id_set]
                if not missing_ids:
                    LOGGER.info(
                        "Reusing %d cached embeddings from %s (corpus has %d rows)",
                        len(cached_ids),
                        embeddings_path,
                        len(ids),
                    )
                    cached_embeddings = np.load(embeddings_path)
                    index_of = {post_id: i for i, post_id in enumerate(cached_ids)}
                    order = [index_of[post_id] for post_id in ids]
                    return cached_embeddings[order]

                missing_set = set(missing_ids)
                missing_df = df[df[id_column].astype(str).isin(missing_set)]
                LOGGER.info(
                    "Embedding cache missing %d new id(s); embedding just those and merging",
                    len(missing_df),
                )
                new_embeddings = vectorizer.fit_transform(
                    _resolve_texts(missing_df, text_column, text_mode).tolist()
                )
                cached_embeddings = np.load(embeddings_path)
                merged_ids = cached_ids + missing_df[id_column].astype(str).tolist()
                merged_embeddings = np.concatenate([cached_embeddings, new_embeddings], axis=0)
                _save_cache(
                    embeddings_path,
                    ids_path,
                    meta_path,
                    merged_embeddings,
                    merged_ids,
                    vectorizer.model_name,
                    text_mode,
                )
                index_of = {post_id: i for i, post_id in enumerate(merged_ids)}
                order = [index_of[post_id] for post_id in ids]
                return merged_embeddings[order]

            LOGGER.info(
                "Embedding cache at %s is stale (model or text_mode changed); re-embedding",
                embeddings_path,
            )

    embeddings = vectorizer.fit_transform(_resolve_texts(df, text_column, text_mode).tolist())
    _save_cache(embeddings_path, ids_path, meta_path, embeddings, ids, vectorizer.model_name, text_mode)
    return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    corpus = load_corpus()
    LOGGER.info("Loaded %d rows", len(corpus))
    vectors = embed_corpus(corpus)
    LOGGER.info("Embedded corpus into shape %s", vectors.shape)
