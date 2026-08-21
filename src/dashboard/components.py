"""Reusable Streamlit rendering helpers for the trend detection dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

STATE_EMOJI = {
    "trending": "\U0001F525",  # fire
    "rising": "\U0001F4C8",  # chart increasing
    "declining": "\U0001F4C9",  # chart decreasing
    "dormant": "\U0001F4A4",  # zzz
}

STATE_ORDER = ["trending", "rising", "declining", "dormant"]


def render_trend_card(row: pd.Series, label: dict[str, str]) -> None:
    """Render a single trending-cluster card."""
    emoji = STATE_EMOJI.get(row["state"], "")
    with st.container(border=True):
        st.markdown(f"### {emoji} {label['label']}")
        st.caption(label["description"])
        cols = st.columns(4)
        cols[0].metric("State", row["state"].title())
        cols[1].metric("Weight", f"{row['weight']:.1f}")
        growth = row.get("growth_rate")
        cols[2].metric("Growth / half-life", f"{growth:.2f}" if growth is not None else "n/a")
        burst_z = row.get("burst_z")
        cols[3].metric("Burst z", f"{burst_z:.2f}" if burst_z is not None else "n/a")
        if row.get("sample_titles"):
            with st.expander("Sample posts"):
                for title in str(row["sample_titles"]).split(" | "):
                    st.markdown(f"- {title}")


def render_trending_cards(summary: pd.DataFrame, labels: dict[int, dict[str, str]]) -> None:
    """Render all active clusters as cards, grouped by trending state."""
    if summary.empty:
        st.info("No active clusters yet. Try lowering the similarity threshold or loading more data.")
        return

    ordered = summary.copy()
    ordered["state_rank"] = ordered["state"].apply(
        lambda s: STATE_ORDER.index(s) if s in STATE_ORDER else len(STATE_ORDER)
    )
    sort_column = "growth_rate" if "growth_rate" in ordered.columns else "velocity"
    ordered = ordered.sort_values(["state_rank", sort_column], ascending=[True, False])

    for _, row in ordered.iterrows():
        label = labels.get(row["cluster_id"], {"label": "Unlabeled", "description": ""})
        render_trend_card(row, label)


def render_velocity_chart(histories: dict[str, pd.DataFrame]) -> None:
    """Line chart of weight-over-time for a handful of top clusters.

    ``histories`` maps a display label to a DataFrame with ``timestamp``
    and ``weight`` columns (see ``VelocityTracker.history_df``).
    """
    if not histories:
        st.info("No velocity history to show yet.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, history in histories.items():
        if history.empty:
            continue
        ax.plot(history["timestamp"], history["weight"], marker="o", markersize=3, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Decayed weight")
    ax.set_title("Cluster weight over time")
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()
    st.pyplot(fig)


def render_static_scatter(embeddings_2d, labels_array) -> None:
    """2D UMAP scatter colored by static cluster assignment."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_array, cmap="tab20", s=15
    )
    ax.set_title("Static clustering (UMAP 2D projection)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    st.pyplot(fig)
