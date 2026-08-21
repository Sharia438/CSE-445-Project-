"""Historical data ingestion: Kaggle/HuggingFace dumps from multiple social
platforms, plus synthetic demo data.

All loaders converge on the same master schema used by ``reddit_streamer.py``
(``post_id``, ``title``, ``text``, ``source``, ``platform``, ``timestamp``)
so downstream embedding/clustering code never needs to know which platform a
row originally came from. ``source`` is the fine-grained community/topic
label (e.g. a subreddit name); ``platform`` is the top-level origin
("reddit", "twitter", "youtube", "bluesky", "synthetic").

Real-data sources (each requires a one-time manual Kaggle download, except
Bluesky which is fetched automatically over the network):

- Reddit: "Reddit Popular" (https://www.kaggle.com/datasets/angelopimienta/reddit-popular)
  -> save `main.csv` as ``data/raw_kaggle_data.csv``
- Twitter: "Pfizer Vaccine Tweets" (https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets)
  -> save `vaccination_tweets.csv` as ``data/raw_twitter_data.csv``
- YouTube: "YouTube Comments Sentiment Dataset (23K Comments)"
  (https://www.kaggle.com/datasets/ansumansatapathy30/youtube-comments-sentiment-dataset-23k-comments)
  -> save `youtube_comments_dataset.csv` as ``data/raw_youtube_data.csv``
- Bluesky: streamed directly from the Hugging Face dataset
  `alpindale/two-million-bluesky-posts`, no download needed.

If none of the real sources are available, ``generate_synthetic_dataset``
fabricates a small multi-topic corpus with deliberate time-bursts so the
rest of the pipeline (in particular velocity/burst scoring) has something
realistic to chew on.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion.reddit_streamer import CSV_PATH as DEFAULT_CSV_PATH
from src.data_ingestion.reddit_streamer import FIELDNAMES

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_KAGGLE_PATH = PROJECT_ROOT / "data" / "raw_kaggle_data.csv"
DEFAULT_TWITTER_RAW_PATH = PROJECT_ROOT / "data" / "raw_twitter_data.csv"
DEFAULT_YOUTUBE_RAW_PATH = PROJECT_ROOT / "data" / "raw_youtube_data.csv"
BLUESKY_HF_DATASET = "alpindale/two-million-bluesky-posts"
DEFAULT_BLUESKY_SAMPLE_SIZE = 5000

SUBREDDIT_URL_RE = re.compile(r"/r/([^/]+)/")

ALL_SOURCES = ["reddit", "twitter", "youtube", "bluesky"]


def parse_subreddit_from_url(url: str) -> str:
    """Extract the subreddit name from a Reddit permalink, e.g.

    ``https://www.reddit.com/r/technology/comments/...`` -> ``technology``.
    Falls back to ``"unknown"`` if the URL doesn't match the expected shape.
    """
    if not isinstance(url, str):
        return "unknown"
    match = SUBREDDIT_URL_RE.search(url)
    return match.group(1) if match else "unknown"


def _find_column(columns: list[str], *candidates: str) -> str:
    """Case-insensitive lookup of the first matching column name.

    Community CSVs vary slightly in casing/underscores, so callers pass a
    few likely spellings and we match against a lowercased/underscore-free
    view of the actual columns.
    """
    normalized = {col.lower().replace(" ", "_"): col for col in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of {candidates} found in columns: {columns}")


def load_kaggle_reddit_popular(path: str | Path = DEFAULT_RAW_KAGGLE_PATH) -> pd.DataFrame:
    """Load the Kaggle "Reddit Popular" ``main.csv`` and map it to the
    master schema.

    Expected source columns: ``post_id``, ``create_utc``, ``post_url``,
    ``title`` (plus ``comment1..3`` / ``comment1..3_score``, unused here).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found at {path}. Download `main.csv` from "
            "https://www.kaggle.com/datasets/angelopimienta/reddit-popular "
            f"and save it to {path}, or use --sources synthetic instead."
        )

    # The real Kaggle download is actually tab-separated despite the `.csv`
    # extension, and raw comment/title text contains unescaped/unbalanced
    # quote characters that break normal CSV quote parsing -- so we read it
    # as a raw tab-delimited file with quoting disabled entirely.
    raw = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, engine="python")

    required = {"post_id", "create_utc", "post_url", "title"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Kaggle CSV at {path} is missing expected columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "post_id": raw["post_id"].apply(lambda x: f"reddit_kaggle_{x}"),
            "title": raw["title"].fillna("").astype(str).str.strip(),
            "text": raw["title"].fillna("").astype(str).str.strip(),
            "source": raw["post_url"].apply(parse_subreddit_from_url),
            "platform": "reddit",
            "timestamp": pd.to_datetime(raw["create_utc"], unit="s", utc=True).apply(
                lambda ts: ts.isoformat()
            ),
        }
    )
    df = df[df["text"] != ""].reset_index(drop=True)
    LOGGER.info("Loaded %d rows from Kaggle Reddit Popular dataset", len(df))
    return df


def load_kaggle_twitter_vaccine_tweets(path: str | Path = DEFAULT_TWITTER_RAW_PATH) -> pd.DataFrame:
    """Load the Kaggle "Pfizer Vaccine Tweets" dataset and map it to the
    master schema.

    Expected source columns include ``id``, ``date``, ``text`` (plus user
    metadata columns, unused here). This dataset is single-topic (COVID
    vaccine discourse), so ``source`` is a fixed label.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found at {path}. Download `vaccination_tweets.csv` from "
            "https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets "
            f"and save it to {path}, or omit 'twitter' from --sources."
        )

    raw = pd.read_csv(path)

    required = {"id", "date", "text"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Kaggle CSV at {path} is missing expected columns: {sorted(missing)}")

    text = raw["text"].fillna("").astype(str).str.strip()
    df = pd.DataFrame(
        {
            "post_id": raw["id"].apply(lambda x: f"twitter_{x}"),
            "title": text.str.slice(0, 80),
            "text": text,
            "source": "covid_vaccine",
            "platform": "twitter",
            "timestamp": pd.to_datetime(raw["date"], utc=True, errors="coerce").apply(
                lambda ts: ts.isoformat() if pd.notna(ts) else None
            ),
        }
    )
    df = df[(df["text"] != "") & df["timestamp"].notna()].reset_index(drop=True)
    LOGGER.info("Loaded %d rows from Kaggle Pfizer Vaccine Tweets dataset", len(df))
    return df


def load_kaggle_youtube_comments(path: str | Path = DEFAULT_YOUTUBE_RAW_PATH) -> pd.DataFrame:
    """Load the Kaggle "YouTube Comments Sentiment Dataset" and map it to
    the master schema.

    Expected source columns (case-insensitive): ``Topic``, ``Video_Title``,
    ``Comment``, ``Published_Date``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found at {path}. Download `youtube_comments_dataset.csv` from "
            "https://www.kaggle.com/datasets/ansumansatapathy30/"
            "youtube-comments-sentiment-dataset-23k-comments "
            f"and save it to {path}, or omit 'youtube' from --sources."
        )

    raw = pd.read_csv(path)

    try:
        topic_col = _find_column(list(raw.columns), "Topic")
        title_col = _find_column(list(raw.columns), "Video_Title", "VideoTitle")
        comment_col = _find_column(list(raw.columns), "Comment")
        date_col = _find_column(list(raw.columns), "Published_Date", "PublishedDate")
    except KeyError as error:
        raise ValueError(f"Kaggle CSV at {path} is missing an expected column: {error}") from error

    comment_text = raw[comment_col].fillna("").astype(str).str.strip()
    df = pd.DataFrame(
        {
            "post_id": [f"youtube_{i}" for i in range(len(raw))],
            "title": raw[title_col].fillna("").astype(str).str.strip(),
            "text": comment_text,
            "source": raw[topic_col].fillna("general").astype(str).str.strip(),
            "platform": "youtube",
            "timestamp": pd.to_datetime(raw[date_col], utc=True, errors="coerce").apply(
                lambda ts: ts.isoformat() if pd.notna(ts) else None
            ),
        }
    )
    df = df[(df["text"] != "") & df["timestamp"].notna()].reset_index(drop=True)
    LOGGER.info("Loaded %d rows from Kaggle YouTube Comments dataset", len(df))
    return df


def load_bluesky_sample(n_rows: int | None = None) -> pd.DataFrame:
    """Stream a small sample of public Bluesky posts from the Hugging Face
    dataset ``alpindale/two-million-bluesky-posts``. Requires network access
    and the ``datasets`` package; no manual download needed.
    """
    n_rows = n_rows if n_rows is not None else DEFAULT_BLUESKY_SAMPLE_SIZE
    try:
        from datasets import load_dataset
    except Exception as error:  # noqa: BLE001 - broken installs raise various error types
        raise RuntimeError(
            "Could not import the `datasets` package needed for Bluesky ingestion "
            f"({error}). Install/repair it with `pip install datasets`, or omit "
            "'bluesky' from --sources."
        ) from error

    try:
        stream = load_dataset(BLUESKY_HF_DATASET, split="train", streaming=True)
        rows = list(stream.take(n_rows))
    except Exception as error:  # noqa: BLE001 - network/HF errors should surface clearly
        raise RuntimeError(
            f"Failed to stream the Bluesky dataset ({BLUESKY_HF_DATASET}) from "
            "Hugging Face. Check your network connection, or omit 'bluesky' "
            "from --sources."
        ) from error

    if not rows:
        raise RuntimeError(f"Bluesky dataset stream ({BLUESKY_HF_DATASET}) returned no rows.")

    texts = [str(row.get("text", "")).strip() for row in rows]
    df = pd.DataFrame(
        {
            "post_id": [f"bluesky_{i}" for i in range(len(rows))],
            "title": [t[:80] for t in texts],
            "text": texts,
            "source": "bluesky_firehose",
            "platform": "bluesky",
            "timestamp": pd.to_datetime(
                [row.get("created_at") for row in rows], utc=True, errors="coerce"
            ).map(lambda ts: ts.isoformat() if pd.notna(ts) else None),
        }
    )
    df = df[(df["text"] != "") & df["timestamp"].notna()].reset_index(drop=True)
    LOGGER.info("Streamed %d rows from Bluesky (%s)", len(df), BLUESKY_HF_DATASET)
    return df


# Topics used for synthetic demo data. Each topic gets its own burst window
# so velocity scoring has clear rising/declining signal to detect.
_SYNTHETIC_TOPICS: dict[str, list[str]] = {
    "technology": [
        "New AI model beats every benchmark overnight",
        "Startup unveils custom AI chip for edge devices",
        "Tech giant launches next-gen AI coding assistant",
        "Researchers publish breakthrough in AI training efficiency",
        "Chipmaker reveals roadmap for AI accelerators",
        "Open-source AI model surpasses closed competitors",
        "Cloud provider slashes AI inference pricing",
        "AI startup raises massive funding round",
    ],
    "sports": [
        "Underdog team clinches championship in stunning final",
        "Star player signs record-breaking contract extension",
        "Championship match ends in dramatic penalty shootout",
        "Veteran player breaks decades-old scoring record",
        "Coach announces surprise retirement after title win",
        "League finals draw record TV viewership",
        "Injury forces star athlete out of championship game",
        "Underdog upset shakes up league standings",
    ],
    "politics": [
        "Election results trigger nationwide reaction",
        "New policy proposal sparks heated debate in parliament",
        "Government unveils sweeping tax reform plan",
        "Election turnout hits record high nationwide",
        "Lawmakers clash over controversial new bill",
        "Policy reform faces fierce opposition from unions",
        "Coalition talks collapse after election deadlock",
        "Public protests erupt over new legislation",
    ],
    "entertainment": [
        "Blockbuster sequel shatters opening weekend record",
        "Award show winners spark social media debate",
        "Beloved franchise announces surprise reboot",
        "Streaming platform renews hit series for new season",
        "Celebrity announcement floods social feeds",
        "Music festival lineup announcement breaks the internet",
        "Long-awaited movie trailer drops to huge reaction",
        "Fan theory about show finale goes viral",
    ],
}


# Word-substitution pool used to paraphrase synthetic templates so posts
# about the same topic aren't byte-identical. Exact duplicates give cosine
# similarity of 1.0, which makes clustering trivially easy and never
# exercises the similarity threshold the way real near-duplicate posts do.
_PARAPHRASE_POOL: dict[str, list[str]] = {
    "new": ["new", "fresh", "latest", "brand-new"],
    "unveils": ["unveils", "reveals", "announces", "debuts"],
    "breaks": ["breaks", "shatters", "smashes"],
    "record": ["record", "milestone", "all-time-high"],
    "stunning": ["stunning", "shocking", "dramatic", "surprising"],
    "massive": ["massive", "huge", "enormous", "record-breaking"],
    "sparks": ["sparks", "ignites", "triggers", "fuels"],
    "surprise": ["surprise", "unexpected", "shock"],
    "reveals": ["reveals", "unveils", "shares", "announces"],
    "beats": ["beats", "tops", "outperforms", "surpasses"],
}


def paraphrase_template(template: str, rng: np.random.Generator, p_replace: float = 0.5) -> str:
    """Light word-substitution paraphrasing of a template sentence."""
    words = template.split(" ")
    out = []
    for word in words:
        bare = word.strip(".,!?").lower()
        pool = _PARAPHRASE_POOL.get(bare)
        if pool and rng.random() < p_replace:
            replacement = str(rng.choice(pool))
            out.append(replacement.capitalize() if word[:1].isupper() else replacement)
        else:
            out.append(word)
    return " ".join(out)


def generate_synthetic_dataset(
    n_per_topic: int = 60,
    seed: int = 42,
    noise_fraction: float = 0.0,
    overlap_fraction: float = 0.0,
    paraphrase: bool = True,
    return_windows: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fabricate a small multi-topic corpus with staggered time-bursts.

    Each topic is concentrated into its own few-hour burst window within a
    simulated 48-hour period, so a downstream dynamic engine sees clear
    rising-then-declining activity per topic rather than uniform noise.

    ``paraphrase`` lightly varies each post's wording (see ``_paraphrase``)
    so posts in the same burst aren't exact duplicates. ``noise_fraction``
    adds that fraction (relative to the real posts) of off-topic filler
    posts scattered uniformly across the whole window, to test whether
    detection can tell a real burst apart from background chatter.
    ``overlap_fraction`` (0=sequential, back-to-back windows; towards 1=
    heavily overlapping) shrinks the stride between topics' burst windows
    without shrinking the windows themselves, so multiple topics can be
    simultaneously active. With ``return_windows=True``, also returns a
    DataFrame of ground-truth burst windows (one row per topic:
    ``source``, ``burst_start``, ``peak``, ``burst_end``, ``n_posts``) for
    use as detection-latency ground truth.
    """
    rng = np.random.default_rng(seed)
    topics = list(_SYNTHETIC_TOPICS.items())
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: list[dict[str, str]] = []
    window_rows: list[dict] = []
    post_counter = 0
    burst_span_hours = 48 / len(topics)
    stride_hours = burst_span_hours * (1.0 - overlap_fraction)
    beta_a, beta_b = 1.5, 4.0
    beta_mean_fraction = beta_a / (beta_a + beta_b)

    for topic_index, (source, templates) in enumerate(topics):
        burst_start = base_time + timedelta(hours=topic_index * stride_hours)
        # Weight timestamps toward the start of the window (a burst that
        # rises quickly then tapers off) using a Beta distribution.
        offsets = rng.beta(a=beta_a, b=beta_b, size=n_per_topic) * burst_span_hours * 3600

        for offset_seconds in sorted(offsets):
            template = templates[rng.integers(0, len(templates))]
            if paraphrase:
                template = paraphrase_template(template, rng)
            timestamp = burst_start + timedelta(seconds=float(offset_seconds))
            rows.append(
                {
                    "post_id": f"synthetic_{post_counter}",
                    "title": template,
                    "text": template,
                    "source": source,
                    "platform": "synthetic",
                    "timestamp": timestamp.isoformat(),
                }
            )
            post_counter += 1

        window_rows.append(
            {
                "source": source,
                "burst_start": burst_start.isoformat(),
                "peak": (burst_start + timedelta(hours=burst_span_hours * beta_mean_fraction)).isoformat(),
                "burst_end": (burst_start + timedelta(hours=burst_span_hours)).isoformat(),
                "n_posts": n_per_topic,
            }
        )

    if noise_fraction > 0:
        n_noise = int(round(len(topics) * n_per_topic * noise_fraction))
        total_span_hours = (topic_index * stride_hours) + burst_span_hours if topics else 48.0
        all_templates = [t for _, templates in topics for t in templates]
        for _ in range(n_noise):
            template = all_templates[rng.integers(0, len(all_templates))]
            if paraphrase:
                template = paraphrase_template(template, rng)
            offset_hours = rng.uniform(0, total_span_hours)
            timestamp = base_time + timedelta(hours=float(offset_hours))
            rows.append(
                {
                    "post_id": f"synthetic_{post_counter}",
                    "title": template,
                    "text": template,
                    "source": "background_noise",
                    "platform": "synthetic",
                    "timestamp": timestamp.isoformat(),
                }
            )
            post_counter += 1

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    LOGGER.info(
        "Generated %d synthetic rows across %d topics (%d noise)",
        len(df),
        len(topics),
        n_noise if noise_fraction > 0 else 0,
    )

    if return_windows:
        windows_df = pd.DataFrame(window_rows)
        return df, windows_df
    return df


def write_master_dataset(df: pd.DataFrame, csv_path: str | Path = DEFAULT_CSV_PATH) -> None:
    """Merge ``df`` into the master CSV, deduping by ``post_id`` and sorting
    by ``timestamp``.
    """
    csv_path = Path(csv_path)
    df = df[FIELDNAMES]

    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = combined.drop_duplicates(subset=["post_id"]).sort_values("timestamp")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %d total rows to %s", len(combined), csv_path)


def _default_loaders(bluesky_sample_size: int | None = None) -> dict:
    return {
        "reddit": lambda: load_kaggle_reddit_popular(DEFAULT_RAW_KAGGLE_PATH),
        "twitter": lambda: load_kaggle_twitter_vaccine_tweets(DEFAULT_TWITTER_RAW_PATH),
        "youtube": lambda: load_kaggle_youtube_comments(DEFAULT_YOUTUBE_RAW_PATH),
        "bluesky": lambda: load_bluesky_sample(bluesky_sample_size),
    }


def load_all_sources(sources: list[str], bluesky_sample_size: int | None = None) -> pd.DataFrame:
    """Attempt to load each requested source independently. A source that
    fails (missing file, network error, etc.) is logged as a warning and
    skipped rather than aborting the whole run.
    """
    loaders = _default_loaders(bluesky_sample_size)
    frames = []
    for source in sources:
        loader = loaders.get(source)
        if loader is None:
            LOGGER.warning("Unknown source %r, skipping", source)
            continue
        try:
            frames.append(loader())
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            LOGGER.warning("Skipping source %r: %s", source, error)

    if not frames:
        return pd.DataFrame(columns=FIELDNAMES)

    return pd.concat(frames, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load historical data into master_dataset.csv")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[*ALL_SOURCES, "synthetic"],
        default=ALL_SOURCES,
        help="Which sources to load and merge. Missing/unreachable sources are skipped "
        "with a warning rather than failing the run.",
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--n-per-topic", type=int, default=60)
    parser.add_argument(
        "--bluesky-sample-size",
        type=int,
        default=DEFAULT_BLUESKY_SAMPLE_SIZE,
        help="How many Bluesky posts to stream when 'bluesky' is in --sources.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    real_sources = [s for s in args.sources if s != "synthetic"]
    df = (
        load_all_sources(real_sources, bluesky_sample_size=args.bluesky_sample_size)
        if real_sources
        else pd.DataFrame(columns=FIELDNAMES)
    )

    if df.empty:
        LOGGER.warning(
            "No real data could be loaded from %s; falling back to synthetic demo data.",
            real_sources,
        )
        df = generate_synthetic_dataset(n_per_topic=args.n_per_topic)
    elif "synthetic" in args.sources:
        df = pd.concat([df, generate_synthetic_dataset(n_per_topic=args.n_per_topic)], ignore_index=True)

    write_master_dataset(df, args.csv_path)


if __name__ == "__main__":
    main()
