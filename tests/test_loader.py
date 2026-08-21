"""Tests for the historical data loaders: each platform-specific loader
maps its raw CSV shape onto the shared master schema, and
``write_master_dataset`` dedupes/sorts correctly when merging.
"""

from __future__ import annotations

import pandas as pd

from src.data_ingestion.historical_loader import (
    FIELDNAMES,
    load_kaggle_reddit_popular,
    load_kaggle_twitter_vaccine_tweets,
    load_kaggle_youtube_comments,
    write_master_dataset,
)


def test_load_kaggle_reddit_popular_maps_to_master_schema(tmp_path):
    raw_path = tmp_path / "raw_kaggle_data.csv"
    raw_path.write_text(
        "post_id\tcreate_utc\tpost_url\ttitle\n"
        "abc123\t1700000000\thttps://www.reddit.com/r/technology/comments/abc123/some_post/\tHello world\n",
        encoding="utf-8",
    )

    df = load_kaggle_reddit_popular(raw_path)

    assert list(df.columns) == FIELDNAMES
    assert df.loc[0, "post_id"] == "reddit_kaggle_abc123"
    assert df.loc[0, "source"] == "technology"
    assert df.loc[0, "platform"] == "reddit"
    assert df.loc[0, "title"] == "Hello world"


def test_load_kaggle_twitter_vaccine_tweets_maps_to_master_schema(tmp_path):
    raw_path = tmp_path / "raw_twitter_data.csv"
    raw_path.write_text(
        "id,date,text\n"
        "42,2021-01-05 10:00:00,Got my vaccine today!\n",
        encoding="utf-8",
    )

    df = load_kaggle_twitter_vaccine_tweets(raw_path)

    assert list(df.columns) == FIELDNAMES
    assert df.loc[0, "post_id"] == "twitter_42"
    assert df.loc[0, "source"] == "covid_vaccine"
    assert df.loc[0, "platform"] == "twitter"


def test_load_kaggle_youtube_comments_maps_to_master_schema(tmp_path):
    raw_path = tmp_path / "raw_youtube_data.csv"
    raw_path.write_text(
        "Topic,Video_Title,Comment,Published_Date\n"
        "gaming,Big Game Review,This game is amazing,2023-03-01\n",
        encoding="utf-8",
    )

    df = load_kaggle_youtube_comments(raw_path)

    assert list(df.columns) == FIELDNAMES
    assert df.loc[0, "post_id"] == "youtube_0"
    assert df.loc[0, "source"] == "gaming"
    assert df.loc[0, "title"] == "Big Game Review"
    assert df.loc[0, "text"] == "This game is amazing"


def test_write_master_dataset_dedupes_by_post_id_and_sorts(tmp_path):
    csv_path = tmp_path / "master_dataset.csv"

    first = pd.DataFrame(
        {
            "post_id": ["a", "b"],
            "title": ["A", "B"],
            "text": ["A", "B"],
            "source": ["s", "s"],
            "platform": ["synthetic", "synthetic"],
            "timestamp": ["2026-01-01T02:00:00+00:00", "2026-01-01T01:00:00+00:00"],
        }
    )
    write_master_dataset(first, csv_path)

    # "a" reappears with different text - should NOT duplicate; "c" is new.
    second = pd.DataFrame(
        {
            "post_id": ["a", "c"],
            "title": ["A-changed", "C"],
            "text": ["A-changed", "C"],
            "source": ["s", "s"],
            "platform": ["synthetic", "synthetic"],
            "timestamp": ["2026-01-01T02:00:00+00:00", "2026-01-01T00:30:00+00:00"],
        }
    )
    write_master_dataset(second, csv_path)

    result = pd.read_csv(csv_path)
    assert len(result) == 3
    assert sorted(result["post_id"]) == ["a", "b", "c"]
    # First-seen "a" row wins on dedup (pandas drop_duplicates keeps first).
    assert result.loc[result["post_id"] == "a", "title"].iloc[0] == "A"
    # Sorted by timestamp ascending.
    assert result["timestamp"].tolist() == sorted(result["timestamp"].tolist())
