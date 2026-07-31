"""Text loading and sentence-embedding utilities for the trend detector.

Turns raw rows from ``data/master_dataset.csv`` into dense embedding vectors
that downstream clustering (``static_clustering.py``) can consume.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "master_dataset.csv"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ["post_id", "title", "text", "source", "timestamp"]


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

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading sentence-transformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
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
) -> np.ndarray:
    """Convenience helper: embed a corpus DataFrame's text column."""
    vectorizer = vectorizer or TextVectorizer()
    return vectorizer.fit_transform(df[text_column].tolist())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    corpus = load_corpus()
    LOGGER.info("Loaded %d rows", len(corpus))
    vectors = embed_corpus(corpus)
    LOGGER.info("Embedded corpus into shape %s", vectors.shape)
