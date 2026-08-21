"""Turn a cluster's raw post titles into a human-readable trend label.

Two strategies:

- ``gemini_label``: calls the Gemini API (``google-genai``) for a punchy
  label + one-sentence description. Requires ``GEMINI_API_KEY``.
- ``heuristic_label``: free, fully offline TF-IDF keyword extraction. Used
  whenever no API key is configured, or if the Gemini call fails for any
  reason (network, quota, parsing).

``label_cluster`` is the single entry point callers (the dashboard) should
use; it picks whichever strategy is available and never raises.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def heuristic_label(titles: list[str], top_n_terms: int = 3) -> dict[str, str]:
    """Free, offline label: top TF-IDF n-grams across the cluster's titles
    plus a templated description.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    titles = [t for t in titles if t and t.strip()]
    if not titles:
        return {"label": "Unlabeled topic", "description": "No sample titles available."}

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=50
        )
        tfidf = vectorizer.fit_transform(titles)
        scores = tfidf.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:top_n_terms]
        top_terms = [terms[i] for i in top_indices if scores[i] > 0]
    except ValueError:
        top_terms = []

    label = " / ".join(term.title() for term in top_terms) if top_terms else titles[0][:60]
    description = f'{len(titles)} related posts, e.g. "{titles[0]}"'

    return {"label": label, "description": description}


def _build_prompt(titles: list[str]) -> str:
    bulleted = "\n".join(f"- {t}" for t in titles[:15])
    return (
        "The following are post titles that were automatically grouped into "
        "one trending topic cluster on social media:\n\n"
        f"{bulleted}\n\n"
        "Respond with exactly two lines:\n"
        "Label: <a punchy trend label, at most 6 words>\n"
        "Description: <one sentence summarizing the trend>"
    )


def _parse_gemini_response(text: str, fallback_title: str) -> dict[str, str]:
    label = None
    description = None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            label = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()

    if not label:
        label = fallback_title[:60]
    if not description:
        description = text.strip()[:200]

    return {"label": label, "description": description}


def gemini_label(
    titles: list[str], api_key: str, model: str = DEFAULT_GEMINI_MODEL
) -> dict[str, str] | None:
    """Ask Gemini for a label + description. Returns ``None`` on any
    failure so callers can fall back to the heuristic method.
    """
    titles = [t for t in titles if t and t.strip()]
    if not titles:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=_build_prompt(titles),
        )
        text = getattr(response, "text", None)
        if not text:
            return None
        return _parse_gemini_response(text, fallback_title=titles[0])
    except Exception:  # noqa: BLE001 - any SDK/network error should just fall back
        LOGGER.exception("Gemini labeling failed; falling back to heuristic label")
        return None


def label_cluster(titles: list[str], api_key: str | None = None) -> dict[str, str]:
    """Label a cluster's titles, preferring Gemini when a key is available.

    ``api_key`` takes precedence over the ``GEMINI_API_KEY`` environment
    variable (loaded from ``.env`` if present).
    """
    if api_key is None:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None

    if api_key:
        result = gemini_label(titles, api_key)
        if result is not None:
            return result

    return heuristic_label(titles)
