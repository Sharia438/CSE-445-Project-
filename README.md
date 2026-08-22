# Dynamic Social Media Trend Detection

CSE 445 — end-to-end NLP and unsupervised ML pipeline that ingests historical and live social media posts, embeds them with transformers, and uses incremental clustering to surface emerging trends in near real time.

**Live demo:** https://76df-116-206-57-65.ngrok-free.app

## Pipeline

1. **Ingest** — load historical data from four platforms (Reddit, Twitter/X, YouTube, Bluesky) plus live Reddit submissions via PRAW, all merged into one `platform`-tagged master dataset
2. **Embed** — encode text with `sentence-transformers` into dense vectors
3. **Cluster** — baseline static clustering (HDBSCAN), then online micro-clustering with time decay
4. **Score** — a dimensionless growth-rate + burst z-score for rising/trending/declining topics
5. **Evaluate** — cluster quality (ARI/NMI vs. `source` pseudo-labels) and a semi-synthetic burst-detection benchmark (latency, precision, recall) — see [Evaluation](#evaluation)
6. **Summarize** — LLM-based trend labels
7. **Visualize** — Streamlit dashboard, reading a precomputed artifact bundle when one exists — see [Kaggle → Streamlit workflow](#kaggle--streamlit-workflow)

**Training runs on Kaggle (GPU), the dashboard runs locally** — the two are connected by an artifact
bundle Kaggle exports and Streamlit reads, so the local machine never needs a GPU or to re-embed the
full corpus. See [Kaggle → Streamlit workflow](#kaggle--streamlit-workflow) below.  

## Project structure

```text
CSE-445-Project-/
├── data/                         # Local datasets (gitignored)
│   └── artifacts/                # Kaggle's exported bundle (see Kaggle -> Streamlit workflow)
├── notebooks/                    # Exploration and experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_clustering_tests.ipynb
│   ├── 03_math_visuals.ipynb
│   └── 04_kaggle_gpu_pipeline.ipynb  # embed + sweep + cluster + evaluate on a Kaggle T4 GPU
├── src/
│   ├── data_ingestion/           # Multi-platform CSV/HF loaders, Reddit streamer, timeline alignment
│   ├── ml_engine/                # Embeddings, static + dynamic clustering, growth-rate/burst scoring
│   ├── evaluation/                # Cluster-quality metrics, burst injection, benchmark, sweeps
│   ├── artifacts.py               # Kaggle -> Streamlit artifact bundle (save/load)
│   ├── summarization/            # Trend labeling
│   └── dashboard/                # Streamlit UI
├── tests/                        # pytest suite (no GPU / model download required)
├── .env                          # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone

```bash
git clone https://github.com/Sharia438/CSE-445-Project-.git
cd CSE-445-Project-
```

### 2. Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

Create a `.env` file in the project root (never commit this file):

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=TrendDetectorBot by u/YourUsername
```

Get Reddit credentials from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) (create a **script** app).

Optional, for LLM-based trend labeling (falls back to a free offline TF-IDF
heuristic if unset):

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 5. Historical data (multiple platforms)

The pipeline is designed to ingest a **diverse, multi-platform** dataset. Each
platform below is optional and independent — the loader skips whatever isn't
present and falls back to synthetic data if nothing else is available, so the
pipeline is always runnable without any manual download.

| Platform | Dataset | Save as |
|----------|---------|---------|
| Reddit | [Reddit Popular](https://www.kaggle.com/datasets/angelopimienta/reddit-popular) — download `main.csv` (free Kaggle account required) | `data/raw_kaggle_data.csv` |
| Twitter / X | [Pfizer Vaccine Tweets](https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets) — download `vaccination_tweets.csv` | `data/raw_twitter_data.csv` |
| YouTube | [YouTube Comments Sentiment Dataset (23K Comments)](https://www.kaggle.com/datasets/ansumansatapathy30/youtube-comments-sentiment-dataset-23k-comments) | `data/raw_youtube_data.csv` |
| Bluesky | [alpindale/two-million-bluesky-posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts) on Hugging Face | **no download needed** — streamed automatically at run time (requires network access and the `datasets` package) |

(`data/` is gitignored, so none of these raw files are committed.)

- `data/raw_kaggle_data.csv`, `data/raw_twitter_data.csv`, `data/raw_youtube_data.csv` — raw platform downloads
- `data/master_dataset.csv` — cleaned / merged working dataset, schema: `post_id`, `title`, `text`, `source`, `platform`, `timestamp` (`platform` is the top-level origin — `reddit` / `twitter` / `youtube` / `bluesky` / `synthetic` — while `source` stays a finer-grained community/topic label, e.g. a subreddit name)

> **Note:** if you have a `master_dataset.csv` from before the `platform`
> column was added, delete it and re-run the historical loader to regenerate
> it under the new schema.

## How to run

### Live Reddit stream

Streams submissions from `r/technology`, `r/news`, and `r/worldnews` into `data/master_dataset.csv`:

```bash
python src/data_ingestion/reddit_streamer.py
```

Stop with `Ctrl+C`. The streamer reconnects automatically on network / API errors.

### Historical loader

Merges historical data from any combination of platforms into
`data/master_dataset.csv`. `--sources` accepts multiple values; each source
is attempted independently — a missing Kaggle file or a failed Bluesky fetch
just logs a warning and is skipped rather than failing the whole run. If
nothing could be loaded at all, it automatically falls back to synthetic data:

```bash
# default: try all four real platforms, merge whatever is available
python src/data_ingestion/historical_loader.py

# pick specific platforms
python src/data_ingestion/historical_loader.py --sources reddit twitter
python src/data_ingestion/historical_loader.py --sources bluesky

# add synthetic data on top of (or instead of) real data
python src/data_ingestion/historical_loader.py --sources reddit synthetic --n-per-topic 60
python src/data_ingestion/historical_loader.py --sources synthetic --n-per-topic 60

# Bluesky defaults to 5,000 streamed posts; adjust if needed
python src/data_ingestion/historical_loader.py --sources bluesky --bluesky-sample-size 10000
```

### Static clustering baseline (NLP + HDBSCAN/DBSCAN)

Embeds `data/master_dataset.csv` with `sentence-transformers` and prints a
cluster summary (size + sample titles per cluster):

```bash
python -m src.ml_engine.static_clustering
# or, to try the DBSCAN baseline instead of HDBSCAN
python -m src.ml_engine.static_clustering --method dbscan
```

Or explore interactively:

```bash
jupyter notebook notebooks/02_clustering_tests.ipynb
```

### Dynamic clustering engine (time-decay + growth-rate/burst scoring)

Replays `data/master_dataset.csv` in chronological order through the online
micro-clustering engine and prints the currently active trends (weight,
growth rate, burst z-score, trending state, sample titles):

```bash
python -m src.ml_engine.dynamic_engine
# YouTube rows especially benefit from embedding title+text, not just the comment:
python -m src.ml_engine.dynamic_engine --text-mode title_text
```

Every live cluster's weight is sampled on a regular time grid (not only when a
post reinforces it), so decay between reinforcements is actually visible —
this is what makes `declining` and `trending` reachable at all. Trending state
is decided from a **dimensionless growth rate** (fractional weight change per
half-life) rather than a raw weight/second slope, so the same threshold means
the same thing whether the corpus spans 48 simulated hours or several real
months: a cluster receiving no new posts always reads `-0.5` after exactly one
half-life (pure decay), `0` means it's exactly replacing its own decay, and
positive means real reinforcement. See [Evaluation](#evaluation) for how the
thresholds are chosen rather than guessed.

### Streamlit dashboard

The full experience — data loading, live trend cards, velocity charts, the
static baseline, and evaluation results — in one app:

```bash
streamlit run src/dashboard/app.py
```

**Mode** (top of the sidebar): **Precomputed (Kaggle)** reads the artifact
bundle a Kaggle GPU run exported to `data/artifacts/` — instant, and includes
the evaluation numbers below. This is the default whenever a bundle is
present; see [Kaggle → Streamlit workflow](#kaggle--streamlit-workflow). Without
a bundle, it falls back to **Compute locally**, the original click-to-run path:
load any combination of platforms (Reddit, Twitter, YouTube, Bluesky, and/or
synthetic demo data) — each button independently merges into the same
`master_dataset.csv` — then tune the engine's similarity threshold and decay
half-life (with an **auto-calibrate** option that sets the trending cutoff from
this corpus's own 95th percentile), and optionally paste a Gemini API key for
nicer trend labels on the top 10 clusters (only the top 10 are labeled per run,
to avoid hundreds of API calls per click). A **Timeline** toggle switches
between each platform's real, largely non-overlapping timestamps and a
demo-only **aligned** view that shifts every platform onto a shared window
(clearly flagged as synthetic timing wherever it's used). A **Max posts for
this run** slider caps how many (most recent) posts the two clustering tabs
process. Tabs: **Trending Now**, **Velocity Over Time**, **Static Baseline**,
**Evaluation**.

### Embedding cache

Embedding ~40k+ posts on a CPU is the slowest step in the pipeline, so both
`run_static_pipeline` and `run_replay` (and therefore the dashboard) go
through `embed_corpus_cached` (`src/ml_engine/vectorizer.py`), which persists
vectors to disk instead of recomputing them on every run:

| File | Contents |
|------|----------|
| `data/embeddings.npy` | the embedding matrix |
| `data/embeddings_ids.npy` | the `post_id` for each row, in the same order |
| `data/embeddings_meta.json` | `{"model_name": ..., "count": ...}` |

The cache is reused (rows are reordered to match, so it works regardless of
sort order or which entry point built it) as long as every `post_id` needed
is present in the cached id list **and** the model name matches. It's
automatically invalidated and rebuilt whenever the corpus grows past what's
cached, shrinks/changes, or the embedding model changes — no manual cache
clearing needed. Delete the three files above to force a full re-embed.

Because the cache format is just `post_id -> vector`, an `embeddings.npy` /
`embeddings_ids.npy` / `embeddings_meta.json` set computed elsewhere (e.g.
the Kaggle GPU notebook below) can be dropped straight into `data/` and the
local dashboard will pick it up automatically, skipping the model entirely.

### Notebooks

```bash
jupyter notebook notebooks/
```

## Kaggle → Streamlit workflow

Training and evaluation run on a Kaggle **T4 GPU**; the interactive dashboard
runs locally and reads what Kaggle computed. Kaggle notebooks have no public
ports, so there's no way to host Streamlit itself from there — this split is
the whole point of the artifact bundle (`src/artifacts.py`): everything
expensive is computed once on the GPU and exported, and the local machine
never re-embeds or re-clusters anything to render the demo.

1. **Locally** — populate the corpus, including Bluesky (it needs a live
   network fetch, so it's easy to end up without it — check
   `df["platform"].value_counts()` on `data/master_dataset.csv` if unsure):
   ```bash
   python src/data_ingestion/historical_loader.py --sources reddit twitter youtube bluesky
   ```
2. **Locally** — zip `src/` and `data/master_dataset.csv` together.
3. **Kaggle** — *Datasets → New Dataset* → upload the zip (e.g. name it
   `cse445-trend-data`).
4. **Kaggle** — *New Notebook → File → Import Notebook* → upload
   `notebooks/04_kaggle_gpu_pipeline.ipynb`.
5. **Kaggle** — *Add Input* → your dataset. *Settings* → Accelerator **GPU T4**,
   Internet **On** (the sentence-transformer model downloads from Hugging
   Face on first use).
6. **Kaggle** — *Run All*. The notebook: loads the corpus, embeds every post
   (GPU), runs a small parameter sweep (similarity × half-life × text mode,
   scored on cluster quality *and* burst detection — set `SWEEP_ON = False`
   in the config cell for a faster re-run once you've already found good
   values), runs the static HDBSCAN baseline and the dynamic engine with
   evaluation against `source` pseudo-labels, runs the semi-synthetic
   burst-detection benchmark, evaluates per-platform, generates figures, and
   exports everything as `artifacts.zip`.
7. **Kaggle** — *Output* tab → download `artifacts.zip`.
8. **Locally** — unzip into `data/artifacts/`.
9. **Locally** — `streamlit run src/dashboard/app.py`. It defaults to
   **Precomputed (Kaggle)** mode and renders instantly, evaluation tab
   included.

## Evaluation

There was previously no way to check whether the dynamic engine actually
detects trends, or how good the clusters are — this section is what answers
that, and it's what `notebooks/04_kaggle_gpu_pipeline.ipynb` computes and the
dashboard's **Evaluation** tab displays.

**Cluster quality** (`src/evaluation/metrics.py`) — ARI, NMI, homogeneity,
completeness, V-measure, and silhouette, scored against the `source` column
(subreddit / YouTube topic / synthetic topic label) as a free, imperfect but
genuinely informative pseudo-label ground truth. Computed for both the static
HDBSCAN baseline and the dynamic engine's per-post cluster assignments,
overall and per platform (`src/data_ingestion/timeline.py::platform_windows`).

**Burst detection** (`src/evaluation/injection.py` +
`src/evaluation/burst_benchmark.py`) — the primary evidence for the "identifies
emerging trends" half of the problem statement. Paraphrased burst topics are
injected into a real background corpus at known timestamps (not exact
duplicates — cosine-similarity-1.0 posts would make detection trivial), then
scored on:
- **detection rate** — the fraction of injected bursts whose majority cluster
  is ever flagged `trending` within the burst window (+ grace period)
- **precision** — of every cluster the engine ever calls `trending`, how many
  actually correspond to a real injected burst (a low number here means the
  thresholds are too sensitive for this corpus — exactly what the sweep tunes)
- **detection latency** — time from the burst's real start to the first
  `trending` flag

**Sliding-window HDBSCAN baseline** (`src/evaluation/sliding_window_baseline.py`)
— the natural "brute force" alternative: periodically re-cluster a trailing
window of recent posts from scratch (UMAP + HDBSCAN), track cluster
continuity across refreshes by nearest-centroid similarity, and flag a
cluster "surging" when it's new-and-large or growing fast. Scored on the
exact same injected corpus as the dynamic engine, so the comparison is
apples-to-apples. On a real 2,860-post Reddit slice (6 injected bursts, default
untuned parameters both sides):

| Method | Detection rate | Precision | Median latency | Posts processed |
|---|---|---|---|---|
| Dynamic engine (online) | 100% | 3% | **1.2h** | **2,860** (once each) |
| Sliding-window HDBSCAN | 100% | 8% | 3.4h | 10,284 (3.6× the corpus) |

Both catch every injected burst at these settings; the online engine gets there
~2.8× faster at 3.6× less compute, while the sliding-window baseline is
somewhat more precise (fewer false "trending" flags) — the parameter sweep
tunes both sides rather than leaving either at these defaults.

**Parameter sweep** (`src/evaluation/sweeps.py`) — grids over similarity
threshold, half-life, and text mode, scoring every combination on both cluster
quality and burst detection, so the shipped defaults come from measured
numbers rather than a guess.

Run any piece directly:

```bash
python -m src.evaluation.burst_benchmark --source reddit --n-bursts 6
```

## Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest -q
```

The suite (`tests/`) needs no GPU and no model download — engine/velocity/
loader/injection/vectorizer tests use small deterministic fixtures and a
fake, hash-seeded embedding function (`tests/conftest.py::FakeVectorizer`)
that stands in for the real sentence-transformer.

## Development roadmap

| Phase | Weeks | Focus | Status |
|-------|-------|--------|--------|
| 1 — Data infrastructure | 1–2 | Historical scaffolding, Reddit streaming | Done |
| 2 — NLP & static clustering | 3–4 | Vectorization, DBSCAN / HDBSCAN baseline | Done |
| 3 — Dynamic temporal engine | 5–7 | Micro-clustering, time decay, growth-rate/burst scoring | Done |
| 4 — Evaluation | 8–9 | Cluster-quality metrics, burst-detection benchmark, parameter sweep | Done |
| 5 — Output & visualization | 9–10 | LLM summarization, Kaggle-artifact-backed Streamlit UI, demo polish | Done |

## Known limitations

- **Platform time windows barely overlap** as collected (Reddit: Aug 2024;
  Twitter: Dec 2020–Nov 2021; YouTube: 2014–2026) — a genuine cross-platform
  trend can't co-occur in the real data. Evaluation always uses real
  timestamps per platform; the dashboard's **Aligned (demo)** timeline mode
  is explicitly a simulation for demo purposes, never used for evaluation.
- **Twitter is single-topic** (Pfizer vaccine discourse), so it contributes
  volume but not topic diversity.
- **`source` pseudo-labels are imperfect** (a subreddit or YouTube topic
  tag is not a ground-truth topic) — ARI/NMI against them are a useful signal,
  not an exact score.

## Tech stack

| Area | Libraries |
|------|-----------|
| Data | `pandas`, `praw`, `python-dotenv`, `datasets` (Hugging Face, for Bluesky) |
| ML / NLP | `numpy`, `scikit-learn`, `sentence-transformers`, `hdbscan`, `umap-learn`, `torch` |
| Viz / UI | `matplotlib`, `seaborn`, `streamlit` |
| Summarization | `google-genai` (Gemini), with an offline TF-IDF fallback |
| Notebooks | `jupyter` |

## License

Academic / course project for CSE 445.

## Live Demo
  https://76df-116-206-57-65.ngrok-free.app
