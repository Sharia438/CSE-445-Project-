# Dynamic Social Media Trend Detection

CSE 445 — end-to-end NLP and unsupervised ML pipeline that ingests historical and live social media posts, embeds them with transformers, and uses incremental clustering to surface emerging trends in near real time.

## Pipeline

1. **Ingest** — load a historical Kaggle corpus and stream live Reddit submissions via PRAW  
2. **Embed** — encode text with `sentence-transformers` into dense vectors  
3. **Cluster** — baseline static clustering (HDBSCAN), then online micro-clustering with time decay  
4. **Score** — velocity / burst detection for rising topics  
5. **Summarize** — LLM-based trend labels  
6. **Visualize** — Streamlit dashboard  

## Project structure

```text
CSE-445-Project-/
├── data/                         # Local datasets (gitignored)
├── notebooks/                    # Exploration and experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_clustering_tests.ipynb
│   └── 03_math_visuals.ipynb
├── src/
│   ├── data_ingestion/           # CSV loaders & Reddit streamer
│   ├── ml_engine/                # Embeddings, clustering, velocity
│   ├── summarization/            # Trend labeling
│   └── dashboard/                # Streamlit UI
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

Optional, for LLM summarization:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Historical data

Place the Kaggle dataset under `data/` (this folder is gitignored). Expected working files include:

- `data/raw_kaggle_data.csv` — raw download  
- `data/master_dataset.csv` — cleaned / streaming output schema: `post_id`, `title`, `text`, `source`, `timestamp`

## How to run

### Live Reddit stream

Streams submissions from `r/technology`, `r/news`, and `r/worldnews` into `data/master_dataset.csv`:

```bash
python src/data_ingestion/reddit_streamer.py
```

Stop with `Ctrl+C`. The streamer reconnects automatically on network / API errors.

### Historical loader

```bash
python src/data_ingestion/historical_loader.py
```

### Streamlit dashboard

```bash
streamlit run src/dashboard/app.py
```

### Notebooks

```bash
jupyter notebook notebooks/
```

## Development roadmap

| Phase | Weeks | Focus |
|-------|-------|--------|
| 1 — Data infrastructure | 1–2 | Historical scaffolding, Reddit streaming |
| 2 — NLP & static clustering | 3–4 | Vectorization, DBSCAN / HDBSCAN baseline |
| 3 — Dynamic temporal engine | 5–7 | Micro-clustering, time decay, velocity scoring |
| 4 — Output & visualization | 8–10 | LLM summarization, Streamlit UI, demo polish |

## Tech stack

| Area | Libraries |
|------|-----------|
| Data | `pandas`, `praw`, `python-dotenv` |
| ML / NLP | `numpy`, `scikit-learn`, `sentence-transformers`, `hdbscan`, `umap-learn` |
| Viz / UI | `matplotlib`, `seaborn`, `streamlit` |
| Summarization | `openai` |
| Notebooks | `jupyter` |

## License

Academic / course project for CSE 445.
