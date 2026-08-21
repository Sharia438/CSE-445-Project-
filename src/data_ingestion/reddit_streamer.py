"""Continuously ingest live Reddit submissions into master_dataset.csv."""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import praw
from dotenv import load_dotenv
from prawcore.exceptions import PrawcoreException

SUBREDDITS = "technology+news+worldnews"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "master_dataset.csv"
DOTENV_PATH = PROJECT_ROOT / ".env"
FIELDNAMES = ["post_id", "title", "text", "source", "platform", "timestamp"]
RECONNECT_DELAY = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_credentials() -> tuple[str, str, str]:
    """Load and validate Reddit credentials from .env."""
    load_dotenv(dotenv_path=DOTENV_PATH)

    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv("REDDIT_USER_AGENT", "").strip()

    missing = [
        key
        for key, value in (
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        )
        if not value
    ]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_keys}")

    return client_id, client_secret, user_agent


def build_reddit() -> praw.Reddit:
    """Create an authenticated, read-only Reddit client."""
    client_id, client_secret, user_agent = load_credentials()
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split()).strip()


def parse_submission(submission: Any) -> dict[str, str] | None:
    """Convert a PRAW submission into the target schema."""
    title = _normalize_text(getattr(submission, "title", "") or "")
    selftext = _normalize_text(getattr(submission, "selftext", "") or "")

    combined_text = " ".join(part for part in (title, selftext) if part).strip()
    if not combined_text:
        return None

    timestamp = datetime.fromtimestamp(
        float(submission.created_utc), tz=timezone.utc
    ).isoformat()

    return {
        "post_id": f"reddit_{submission.id}",
        "title": title,
        "text": combined_text,
        "source": str(submission.subreddit),
        "platform": "reddit",
        "timestamp": timestamp,
    }


def append_row(row: dict[str, str]) -> None:
    """Append a single row to the CSV, writing headers only once."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    with CSV_PATH.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def stream_loop() -> None:
    """Consume live submissions and persist them one by one."""
    reddit = build_reddit()
    subreddit = reddit.subreddit(SUBREDDITS)
    LOGGER.info("Listening to subreddit stream: %s", SUBREDDITS)

    for submission in subreddit.stream.submissions(skip_existing=True):
        row = parse_submission(submission)
        if row is None:
            continue
        append_row(row)
        LOGGER.info(
            "Saved %s | source=%s | title=%s",
            row["post_id"],
            row["source"],
            row["title"][:80],
        )


def main() -> None:
    """Run the stream forever with automatic reconnects."""
    while True:
        try:
            stream_loop()
        except KeyboardInterrupt:
            LOGGER.info("Streamer stopped by user.")
            break
        except (PrawcoreException, OSError, TimeoutError) as error:
            LOGGER.error(
                "Stream interrupted (%s). Reconnecting in %s seconds...",
                error,
                RECONNECT_DELAY,
            )
            time.sleep(RECONNECT_DELAY)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception(
                "Unexpected error (%s). Reconnecting in %s seconds...",
                error,
                RECONNECT_DELAY,
            )
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()
