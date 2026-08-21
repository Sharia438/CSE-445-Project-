"""Streamlit dashboard for the Dynamic Social Media Trend Detection project.

Run with:

    streamlit run src/dashboard/app.py

Two modes, chosen in the sidebar:

- **Precomputed (Kaggle)**: reads the artifact bundle a Kaggle GPU run
  exported to ``data/artifacts/`` (see ``src/artifacts.py`` and
  ``notebooks/04_kaggle_gpu_pipeline.ipynb``) - instant, and includes the
  evaluation numbers (cluster quality, burst-detection latency) the Kaggle
  run computed. This is the default whenever a bundle is present.
- **Compute locally**: the original click-to-run path, embedding/replaying
  on this machine. Useful without a Kaggle run, or to try different
  parameters live; slow on the full corpus without a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.artifacts import ArtifactBundle, corpus_fingerprint, load_bundle
from src.dashboard.components import (
    render_static_scatter,
    render_trending_cards,
    render_velocity_chart,
)
from src.data_ingestion.historical_loader import (
    DEFAULT_RAW_KAGGLE_PATH,
    DEFAULT_TWITTER_RAW_PATH,
    DEFAULT_YOUTUBE_RAW_PATH,
    generate_synthetic_dataset,
    load_bluesky_sample,
    load_kaggle_reddit_popular,
    load_kaggle_twitter_vaccine_tweets,
    load_kaggle_youtube_comments,
    write_master_dataset,
)
from src.data_ingestion.timeline import align_platform_timelines, platform_windows
from src.ml_engine.dynamic_engine import (
    DEFAULT_HALF_LIFE_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DynamicClusteringEngine,
    run_replay,
)
from src.ml_engine.static_clustering import cluster_summary, reduce_umap, run_static_pipeline
from src.ml_engine.vectorizer import DEFAULT_CSV_PATH, TextVectorizer, load_corpus
from src.summarization.trend_labeler import label_cluster

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
TOP_N_LABEL = 10

st.set_page_config(page_title="Dynamic Social Media Trend Detection", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_vectorizer() -> TextVectorizer:
    return TextVectorizer()


@st.cache_resource(show_spinner=False)
def _get_bundle(artifacts_dir: str) -> ArtifactBundle | None:
    return load_bundle(artifacts_dir)


def _load_or_none(csv_path: Path) -> pd.DataFrame | None:
    try:
        return load_corpus(csv_path)
    except (FileNotFoundError, ValueError) as error:
        st.warning(str(error))
        return None


def _load_and_merge(label: str, loader) -> None:
    """Run a single-source loader, merge it into the master CSV, and report
    the outcome in the sidebar."""
    try:
        df = loader()
        write_master_dataset(df, DEFAULT_CSV_PATH)
        st.sidebar.success(f"Merged {len(df)} {label} rows into {DEFAULT_CSV_PATH.name}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.sidebar.error(str(error))


def _render_sidebar_data_controls() -> None:
    st.sidebar.header("1. Data")
    st.sidebar.caption(
        "Load one or more platforms below; each merges into the same "
        "`master_dataset.csv` (deduped by `post_id`), so you can mix and "
        "match real and synthetic data."
    )

    st.sidebar.subheader("Reddit")
    st.sidebar.caption(
        "Download `main.csv` from [Reddit Popular]"
        "(https://www.kaggle.com/datasets/angelopimienta/reddit-popular) "
        f"and save it as `{DEFAULT_RAW_KAGGLE_PATH}`."
    )
    if st.sidebar.button("Load Reddit (Kaggle)"):
        _load_and_merge("Reddit", lambda: load_kaggle_reddit_popular(DEFAULT_RAW_KAGGLE_PATH))

    st.sidebar.subheader("Twitter / X")
    st.sidebar.caption(
        "Download `vaccination_tweets.csv` from [Pfizer Vaccine Tweets]"
        "(https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets) "
        f"and save it as `{DEFAULT_TWITTER_RAW_PATH}`."
    )
    if st.sidebar.button("Load Twitter (Kaggle)"):
        _load_and_merge(
            "Twitter", lambda: load_kaggle_twitter_vaccine_tweets(DEFAULT_TWITTER_RAW_PATH)
        )

    st.sidebar.subheader("YouTube")
    st.sidebar.caption(
        "Download the [YouTube Comments Sentiment Dataset]"
        "(https://www.kaggle.com/datasets/ansumansatapathy30/"
        "youtube-comments-sentiment-dataset-23k-comments) "
        f"and save it as `{DEFAULT_YOUTUBE_RAW_PATH}`."
    )
    if st.sidebar.button("Load YouTube (Kaggle)"):
        _load_and_merge("YouTube", lambda: load_kaggle_youtube_comments(DEFAULT_YOUTUBE_RAW_PATH))

    st.sidebar.subheader("Bluesky")
    st.sidebar.caption("Auto-fetched from Hugging Face; no manual download, needs network access.")
    if st.sidebar.button("Load Bluesky (auto-fetch)"):
        with st.spinner("Streaming Bluesky posts from Hugging Face..."):
            _load_and_merge("Bluesky", load_bluesky_sample)

    st.sidebar.subheader("Synthetic")
    n_per_topic = st.sidebar.slider("Posts per topic", 10, 200, 60, step=10)
    if st.sidebar.button("Generate synthetic data"):
        _load_and_merge(
            "synthetic", lambda: generate_synthetic_dataset(n_per_topic=n_per_topic)
        )


def _render_sidebar_mode_controls(bundle: ArtifactBundle | None) -> str:
    st.sidebar.header("0. Mode")
    options = ["Precomputed (Kaggle)", "Compute locally"]
    default_index = 0 if bundle is not None else 1
    if bundle is None:
        st.sidebar.caption(
            "No artifact bundle found in `data/artifacts/` - run the Kaggle "
            "notebook and download it there to enable precomputed mode."
        )
    mode = st.sidebar.radio("Data source", options, index=default_index)
    return mode


def _render_sidebar_timeline_controls() -> str:
    st.sidebar.header("2. Timeline")
    st.sidebar.caption(
        "Real: each platform keeps its actual, largely non-overlapping "
        "timestamps (honest, but a cross-platform trend can't co-occur). "
        "Aligned: a demo-only view that shifts every platform onto a "
        "shared window - not real timing, clearly labelled as such."
    )
    return st.sidebar.radio("Timeline", ["Real (per-platform)", "Aligned (demo)"], index=0)


def _render_sidebar_engine_controls(total_posts: int) -> dict:
    st.sidebar.header("3. Dynamic engine parameters")

    max_posts = total_posts
    if total_posts > 1000:
        max_posts = st.sidebar.slider(
            "Max posts for this run",
            min_value=500,
            max_value=total_posts,
            value=min(5000, total_posts),
            step=500,
            help=(
                "Both tabs below only process the N most recent posts, so a "
                "live demo doesn't have to embed/replay the entire corpus. "
                "Embeddings are cached to disk (see data/embeddings.npy) and "
                "merged rather than overwritten as this grows, so raising it "
                "later only embeds the newly-added posts."
            ),
        )

    similarity_threshold = st.sidebar.slider(
        "Similarity threshold", 0.1, 0.95, DEFAULT_SIMILARITY_THRESHOLD, step=0.05
    )
    half_life_hours = st.sidebar.slider(
        "Time-decay half-life (hours)", 1, 48, int(DEFAULT_HALF_LIFE_SECONDS / 3600)
    )
    auto_calibrate = st.sidebar.checkbox(
        "Auto-calibrate trending cutoff",
        value=True,
        help=(
            "Set the trending growth-rate cutoff from this corpus's own "
            "95th percentile instead of a fixed value, so the demo adapts "
            "to whatever data is loaded."
        ),
    )
    gemini_api_key = st.sidebar.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Leave blank to use the free offline heuristic labeler instead.",
    )
    return {
        "max_posts": max_posts,
        "similarity_threshold": similarity_threshold,
        "half_life_seconds": half_life_hours * 3600.0,
        "auto_calibrate": auto_calibrate,
        "gemini_api_key": gemini_api_key or None,
    }


def _run_dynamic_pipeline(df: pd.DataFrame, params: dict) -> None:
    vectorizer = _get_vectorizer()
    engine = DynamicClusteringEngine(
        similarity_threshold=params["similarity_threshold"],
        half_life_seconds=params["half_life_seconds"],
    )
    result = run_replay(df, vectorizer=vectorizer, engine=engine)
    summary = engine.active_cluster_summary(auto_calibrate=params["auto_calibrate"])

    # Only label the clusters actually rendered - labeling every active
    # cluster (previously hundreds) meant hundreds of Gemini calls per
    # button press.
    top_ids = summary.sort_values("growth_rate", ascending=False)["cluster_id"].head(TOP_N_LABEL)
    labels = {}
    for cluster_id in top_ids:
        cluster = engine.registry.clusters.get(cluster_id)
        if cluster is None:
            continue
        labels[cluster_id] = label_cluster(list(cluster.recent_titles), api_key=params["gemini_api_key"])

    st.session_state["engine"] = engine
    st.session_state["summary"] = summary
    st.session_state["labels"] = labels
    st.session_state["history"] = result.history


def _run_static_pipeline(df: pd.DataFrame) -> None:
    vectorizer = _get_vectorizer()
    df, embeddings, cluster_labels = run_static_pipeline(df=df, vectorizer=vectorizer)
    embeddings_2d = reduce_umap(embeddings, n_components=2)
    st.session_state["static_df"] = df
    st.session_state["static_labels"] = cluster_labels
    st.session_state["static_embeddings_2d"] = embeddings_2d


def _render_overview_tab(df: pd.DataFrame) -> None:
    if "platform" in df.columns:
        platform_counts = df["platform"].value_counts()
        with st.expander(f"Platform breakdown ({platform_counts.size} platforms)", expanded=True):
            col_chart, col_table = st.columns([2, 1])
            col_chart.bar_chart(platform_counts)
            col_table.dataframe(
                platform_counts.rename("posts").reset_index().rename(columns={"index": "platform"}),
                width="stretch",
                hide_index=True,
            )

        windows = platform_windows(df)
        if not windows.empty:
            st.caption(
                "Real per-platform time windows - note these are largely "
                "non-overlapping, so a genuine cross-platform trend can't "
                "co-occur in this corpus as collected."
            )
            st.dataframe(windows, width="stretch", hide_index=True)


def _render_evaluation_tab(bundle: ArtifactBundle | None) -> None:
    if bundle is None or not bundle.evaluation:
        st.info(
            "No evaluation results yet. Run the Kaggle notebook "
            "(`notebooks/04_kaggle_gpu_pipeline.ipynb`) and download its "
            "artifact bundle into `data/artifacts/` to see cluster-quality "
            "and burst-detection metrics here."
        )
        return

    evaluation = bundle.evaluation

    if "static" in evaluation or "dynamic" in evaluation:
        st.subheader("Cluster quality vs. `source` pseudo-labels")
        rows = []
        if "static" in evaluation:
            rows.append({"method": "static (HDBSCAN)", **evaluation["static"]})
        if "dynamic" in evaluation:
            rows.append({"method": "dynamic (online)", **evaluation["dynamic"]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if "burst" in evaluation:
        st.subheader("Burst-detection benchmark (semi-synthetic injection)")
        burst = evaluation["burst"]
        cols = st.columns(4)
        cols[0].metric("Detection rate", f"{burst.get('recall', 0):.0%}")
        cols[1].metric("Precision", f"{burst.get('precision', 0):.0%}")
        median_latency = burst.get("median_latency_seconds")
        cols[2].metric(
            "Median latency", f"{median_latency / 3600:.1f}h" if median_latency is not None else "n/a"
        )
        cols[3].metric("False-positive clusters", burst.get("n_false_positive_clusters", 0))

    if "sliding_window_burst" in evaluation:
        st.subheader("vs. sliding-window HDBSCAN baseline")
        st.caption(
            "The periodic 'recluster a trailing window from scratch' alternative to "
            "online micro-clustering, scored on the same injected bursts - this is "
            "the direct evidence for whether the online engine is actually better, "
            "not just whether it works in isolation."
        )
        dynamic_row = {"method": "dynamic (online)", **evaluation.get("burst", {})}
        sliding_row = {"method": "sliding-window HDBSCAN", **evaluation["sliding_window_burst"]}
        compare_df = pd.DataFrame([dynamic_row, sliding_row])
        display_cols = [
            c
            for c in [
                "method",
                "recall",
                "precision",
                "median_latency_seconds",
                "total_posts_reclustered",
                "n_refreshes",
            ]
            if c in compare_df.columns
        ]
        st.dataframe(compare_df[display_cols], width="stretch", hide_index=True)

    eval_dir = DEFAULT_ARTIFACTS_DIR
    per_burst_path = eval_dir / "eval_bursts.csv"
    if per_burst_path.exists():
        st.caption("Per-burst detail (dynamic engine)")
        st.dataframe(pd.read_csv(per_burst_path), width="stretch", hide_index=True)

    per_burst_sliding_path = eval_dir / "eval_bursts_sliding_window.csv"
    if per_burst_sliding_path.exists():
        st.caption("Per-burst detail (sliding-window baseline)")
        st.dataframe(pd.read_csv(per_burst_sliding_path), width="stretch", hide_index=True)

    sweep_path = eval_dir / "sweep_results.csv"
    if sweep_path.exists():
        st.caption("Parameter sweep (similarity x half-life x text mode)")
        st.dataframe(pd.read_csv(sweep_path), width="stretch", hide_index=True)

    if bundle.figures_dir is not None:
        st.caption("Figures generated by the Kaggle run")
        figure_cols = st.columns(2)
        for i, path in enumerate(sorted(bundle.figures_dir.glob("*.png"))):
            figure_cols[i % 2].image(str(path), caption=path.stem)


def _render_precomputed_dashboard(bundle: ArtifactBundle) -> None:
    st.success(
        f"Loaded precomputed artifacts generated {bundle.run_meta.get('generated_at', 'unknown time')} "
        f"({bundle.run_meta.get('n_posts', '?')} posts)."
    )

    tab_trending, tab_velocity, tab_static, tab_eval = st.tabs(
        ["Trending Now", "Velocity Over Time", "Static Baseline", "Evaluation"]
    )

    with tab_trending:
        if bundle.dynamic_summary is not None and not bundle.dynamic_summary.empty:
            labels = {
                row["cluster_id"]: {
                    "label": row.get("label", f"Cluster {row['cluster_id']}"),
                    "description": row.get("description", ""),
                }
                for _, row in bundle.dynamic_summary.iterrows()
            }
            render_trending_cards(bundle.dynamic_summary, labels)
        else:
            st.info("No dynamic_summary.csv in the artifact bundle.")

    with tab_velocity:
        if bundle.dynamic_history is not None and not bundle.dynamic_history.empty:
            top_ids = (
                bundle.dynamic_summary.sort_values("weight", ascending=False)["cluster_id"].head(6)
                if bundle.dynamic_summary is not None
                else bundle.dynamic_history["cluster_id"].unique()[:6]
            )
            histories = {}
            for cid in top_ids:
                hist = bundle.dynamic_history[bundle.dynamic_history["cluster_id"] == cid][
                    ["timestamp", "weight"]
                ].copy()
                hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, format="ISO8601")
                histories[f"#{cid}"] = hist
            render_velocity_chart(histories)
        else:
            st.info("No dynamic_history.csv in the artifact bundle.")

    with tab_static:
        if bundle.static_umap2d is not None and bundle.static_labels is not None:
            render_static_scatter(bundle.static_umap2d, bundle.static_labels["cluster"].to_numpy())
            if bundle.static_summary is not None:
                st.dataframe(bundle.static_summary, width="stretch")
        else:
            st.info("No static clustering artifacts in the bundle.")

    with tab_eval:
        _render_evaluation_tab(bundle)


def _render_local_dashboard(df: pd.DataFrame, bundle: ArtifactBundle | None) -> None:
    timeline_mode = _render_sidebar_timeline_controls()
    if timeline_mode == "Aligned (demo)":
        df = align_platform_timelines(df)
        st.warning(
            "Timeline is **aligned for demo purposes** - timestamps have been "
            "shifted so platforms overlap; this is not the real posting time. "
            "See `timestamp_original` for the real value."
        )

    params = _render_sidebar_engine_controls(total_posts=len(df))

    st.write(f"Loaded **{len(df)}** posts from `{DEFAULT_CSV_PATH.name}`.")

    if params["max_posts"] < len(df):
        df_run = (
            df.assign(_ts=pd.to_datetime(df["timestamp"], utc=True, format="ISO8601"))
            .sort_values("_ts")
            .tail(params["max_posts"])
            .drop(columns="_ts")
            .reset_index(drop=True)
        )
        st.caption(
            f"Using the {len(df_run)} most recent of {len(df)} posts for the tabs below "
            "(adjust 'Max posts for this run' in the sidebar)."
        )
    else:
        df_run = df

    _render_overview_tab(df)

    tab_trending, tab_velocity, tab_static, tab_eval = st.tabs(
        ["Trending Now", "Velocity Over Time", "Static Baseline", "Evaluation"]
    )

    with tab_trending:
        if st.button("Run dynamic trend detection", type="primary"):
            with st.spinner("Embedding posts and running the dynamic engine..."):
                _run_dynamic_pipeline(df_run, params)

        if "summary" in st.session_state:
            render_trending_cards(st.session_state["summary"], st.session_state["labels"])
        else:
            st.info("Click 'Run dynamic trend detection' to see active trends.")

    with tab_velocity:
        if "engine" in st.session_state:
            engine = st.session_state["engine"]
            labels = st.session_state["labels"]
            top_cluster_ids = st.session_state["summary"].sort_values(
                "weight", ascending=False
            )["cluster_id"].head(6)

            histories = {
                f"#{cid} {labels.get(cid, {}).get('label', '')}"[:40]: engine.velocity.history_df(cid)
                for cid in top_cluster_ids
            }
            render_velocity_chart(histories)
        else:
            st.info("Run dynamic trend detection first to see velocity history.")

    with tab_static:
        st.caption("Phase 2 baseline: one-shot UMAP + HDBSCAN clustering over the same corpus.")
        if st.button("Run static clustering baseline"):
            with st.spinner("Embedding posts and running HDBSCAN..."):
                _run_static_pipeline(df_run)

        if "static_df" in st.session_state:
            render_static_scatter(
                st.session_state["static_embeddings_2d"], st.session_state["static_labels"]
            )
            summary = cluster_summary(st.session_state["static_df"], st.session_state["static_labels"])
            st.dataframe(summary, width="stretch")
        else:
            st.info("Click 'Run static clustering baseline' to see results.")

    with tab_eval:
        _render_evaluation_tab(bundle)


def main() -> None:
    st.title("Dynamic Social Media Trend Detection")
    st.caption(
        "NLP + unsupervised ML pipeline that detects emerging trends from streams of "
        "social media posts. Kaggle GPU trains and evaluates; this dashboard serves it."
    )

    bundle = _get_bundle(str(DEFAULT_ARTIFACTS_DIR))
    mode = _render_sidebar_mode_controls(bundle)
    _render_sidebar_data_controls()

    df = _load_or_none(DEFAULT_CSV_PATH)

    if mode == "Precomputed (Kaggle)" and bundle is not None:
        if df is not None:
            fingerprint = corpus_fingerprint(df)
            bundle_fingerprint = bundle.run_meta.get("corpus_fingerprint")
            if bundle_fingerprint and bundle_fingerprint != fingerprint:
                st.warning(
                    "The artifact bundle's corpus fingerprint doesn't match the "
                    "local `master_dataset.csv` - the bundle may have been built "
                    "from a different dataset. Showing it anyway."
                )
        _render_precomputed_dashboard(bundle)
        return

    if df is None:
        st.info(
            "No dataset loaded yet. Use the sidebar to generate synthetic demo data "
            "or load the Kaggle dataset, then re-run."
        )
        return

    _render_local_dashboard(df, bundle)


if __name__ == "__main__":
    main()
