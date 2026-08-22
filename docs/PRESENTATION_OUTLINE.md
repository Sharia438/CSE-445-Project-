# Presentation structure — Dynamic Social Media Trend Detection

Outline for a ~12–15 minute CSE 445 presentation, built from the actual pipeline in this
repo. Each slide lists its purpose, the content to put on it, and which figure/number to
pull in. Figures referenced as `data/artifacts/figures/NN_*.png` come from
[`notebooks/04_kaggle_gpu_pipeline.ipynb`](../notebooks/04_kaggle_gpu_pipeline.ipynb) — run
that notebook once on Kaggle before building slides, so every number below is the real one
from your run, not a placeholder. Where this outline shows a number from local testing
during development, it's marked **(dev sample — replace with your Kaggle run)**.

**Target length:** 13 slides, ~1 minute each, +2–3 min for the live demo.

---

## 1. Title slide

- Project name: **Dynamic Social Media Trend Detection**
- Course: CSE 445, team members
- One-line hook: *"An online clustering engine that watches a stream of social media posts
  and flags emerging topics as they burst — not after the fact."*

---

## 2. Problem statement

- Verbatim prompt: *"Collect a diverse dataset of public social media posts from various
  platforms. Develop a dynamic clustering model that identifies emerging trends and topics
  in these posts."*
- Two deliverables, restated as questions your talk will answer:
  1. Is the dataset actually diverse (multiple platforms, topics, time)?
  2. Does the model actually detect emerging trends — with a number attached, not a claim?
- This framing sets up slides 4 (dataset) and 9–11 (evaluation) as the two halves of the
  talk that answer it directly.

---

## 3. Why "dynamic" is the hard part

- Contrast in one slide: static clustering (HDBSCAN once, over everything) vs. online
  micro-clustering (each post processed once, in one pass, as it arrives)
- Why static isn't enough: no notion of *time* — can't tell "trending now" from "trending
  last month," can't run on a live stream without re-clustering from scratch
- Sets up the two methods you'll present (slides 6–7) as complementary, not either/or —
  the static baseline is still in the pipeline as a sanity check

---

## 4. Dataset: diversity across platforms

- Table: platforms × row counts × time window. From this project's real corpus:

  | Platform | Posts | Time window |
  |---|---|---|
  | Reddit | 9,982 | Jul 27 – Aug 24, 2024 |
  | Twitter/X | 11,020 | Dec 2020 – Nov 2021 |
  | YouTube | 23,751 | 2014 – 2026 |
  | Bluesky | *(run the loader — see README)* | — |

  *(dev sample — replace with your Kaggle run's `df["platform"].value_counts()` and
  `platform_windows(df)` output, Section 3 of the notebook)*
- Figure: `01_platform_breakdown.png`
- **Say this out loud, don't hide it:** the platforms barely overlap in time — a real
  cross-platform trend can't co-occur in this data as collected. Show `02_posts_over_time.png`
  as the visual proof. This is a limitation you own, not a bug you missed — see slide 12.

---

## 5. Pipeline overview

- One architecture diagram, five stages: **Ingest → Embed → Cluster (static + dynamic) →
  Score → Evaluate → Serve**
- Call out where Kaggle vs. local runs: GPU-heavy stages (embed, cluster, evaluate) run once
  on Kaggle; the Streamlit dashboard reads the exported result locally, instantly
- Good spot for a simple boxes-and-arrows diagram (Mermaid or hand-drawn), not code

---

## 6. Method: static baseline

- `sentence-transformers` (`all-MiniLM-L6-v2`) → UMAP (5D) → HDBSCAN
- One sentence on why UMAP first: HDBSCAN needs density structure that's easier to find in a
  reduced space than in 384 raw dimensions
- Figure: `03_umap_scatter.png` (colored by cluster) — this is usually your most visually
  convincing slide, use it big
- One line on evaluation: scored against the `source` column (subreddit / YouTube topic) as
  a free pseudo-label — not a claim of ground truth, a useful signal

---

## 7. Method: the dynamic engine

- Each post, once, in arrival order:
  1. Decay every existing cluster's weight (time-based forgetting)
  2. Find the most similar existing cluster (cosine similarity on the embedding)
  3. Above threshold → merge in; below → spawn a new micro-cluster
- The key design choice: decayed **weight** is a recency-weighted "how much is this topic
  still being talked about," not a raw post count
- Keep the algorithm on this slide to ~4 bullets — the math goes on slide 8

---

## 8. The half-life trick (this is your differentiator slide)

- State the formula in plain language: a cluster that gets **no new posts** loses exactly
  half its weight every half-life, by construction
- That gives a **dimensionless growth rate**: `-0.5` = abandoned, `0` = holding steady,
  positive = real reinforcement — the same number means the same thing whether your data
  spans 48 hours or 8 months
- One sentence on why this mattered: an earlier version of this engine sampled weight only
  when a cluster got a new post, so decay between posts was invisible — on real data that
  produced **0 trending clusters out of 2,616** and only 1 ever marked declining. Fixing the
  sampling (a regular time grid) and switching to this growth rate made all four states —
  dormant / rising / trending / declining — actually reachable.
- This is the single best "we understood the problem, not just the library calls" slide —
  don't cut it for time

---

## 9. Evaluation methodology

- Two questions, two protocols:
  1. **Are the clusters any good?** ARI / NMI / silhouette vs. `source` pseudo-labels
     (`src/evaluation/metrics.py`)
  2. **Does it actually catch a trend, and how fast?** Inject paraphrased "burst" topics into
     real background posts at known timestamps, then measure detection latency,
     precision, recall (`src/evaluation/injection.py` + `burst_benchmark.py`)
- Why injection instead of hand-labeling real bursts: nobody agrees on exactly when a real
  trend "started," so hand labels would be arguable — injected timestamps are ground truth
  by construction
- Keep this slide conceptual; the numbers go on the next slide

---

## 10. Results: cluster quality

- One table, static vs. dynamic, overall and per-platform:

  | Method | ARI | NMI | Silhouette |
  |---|---|---|---|
  | Static (HDBSCAN) | 0.41 | 0.79 | 0.37 |
  | Dynamic (online) | 0.57 | 0.85 | — |

  *(dev sample, 1,500-post real subset — replace with your Kaggle run's full-corpus numbers,
  `evaluation.json` → `static`/`dynamic` in the artifact bundle)*
- One line of interpretation: the dynamic engine's per-post assignment scored *higher* than
  the one-shot static baseline on this sample — call out that it's still one pass, no
  re-clustering, no lookahead

---

## 11. Results: burst detection (the headline slide)

- Head-to-head table, dynamic engine vs. the sliding-window HDBSCAN baseline (periodically
  re-clustering a trailing window from scratch — the natural "brute force" alternative):

  | Method | Detection rate | Median latency | Posts processed |
  |---|---|---|---|
  | Dynamic engine (online) | 100% | **1.2h** | **2,860** (once each) |
  | Sliding-window HDBSCAN | 100% | 3.4h | 10,284 (3.6× the corpus) |

  *(dev sample, 6 injected bursts on a 2,860-post real Reddit slice — replace with your
  Kaggle run's numbers, `eval_bursts.csv` / `eval_bursts_sliding_window.csv`)*
- Figure: `09_burst_latency.png` and `10_dynamic_vs_sliding_window.png`
- The message: both catch every injected burst here; the online engine gets there ~3×
  faster at a fraction of the compute — that's the actual case for "dynamic," not just that
  it works in isolation

---

## 12. Limitations (own them before someone asks)

- Platform time windows barely overlap as collected — no real cross-platform co-occurrence
  is possible in this data (slide 4's `02_posts_over_time.png` is the proof)
- Twitter is single-topic (COVID vaccine discourse) — volume, not topic diversity
- `source` pseudo-labels are a convenience, not verified ground truth
- Precision on the burst benchmark is low at default/untuned thresholds — the parameter
  sweep (`sweeps.py`) exists precisely because the right threshold depends on the corpus
- One slide, stated plainly, is worth more than a Q&A ambush

---

## 13. Conclusion + what's next

- Recap the two questions from slide 2, answered:
  - Diversity: three-to-four platforms, [N] posts, explicit about the time-window caveat
  - Dynamic detection: measurable, faster and cheaper than the periodic-recluster
    alternative, at comparable or better cluster quality
- Future work, pick 2–3: an aligned-timeline cross-platform demo scored honestly; richer
  LLM-based trend summarization beyond the offline TF-IDF fallback; a truly live Reddit
  stream running against the dashboard instead of a historical replay
- Thanks / questions

---

## Demo slot (after slide 11, or as a live cutaway)

Run `streamlit run src/dashboard/app.py` in **Precomputed (Kaggle)** mode (instant, once
`data/artifacts/` is populated from the notebook). Walk through, in order:

1. **Overview** tab — platform breakdown + the honest time-window table
2. **Trending Now** — point at 2–3 real trend cards (e.g. a sports/news event that actually
   showed up as `trending` in your run) and read the growth-rate / burst-z numbers off the
   card
3. **Evaluation** tab — the same ARI/NMI and burst-detection tables as slides 10–11, live

If presenting somewhere without reliable internet/screen-share for a live app, take
screenshots of these three views ahead of time as a fallback — don't let a live-demo failure
eat your time slot.

## Building the actual slide deck

This file is the outline, not the deck. To turn it into slides:

- Use the `pptx` skill (`/pptx` or ask for a "PowerPoint deck") and hand it this file as the
  structure to follow — one slide per numbered section above.
- Pull figures directly from `data/artifacts/figures/*.png` (generated by the Kaggle
  notebook) rather than recreating charts by hand.
- Replace every **(dev sample)** number with your own Kaggle run's numbers before presenting
  — they're realistic placeholders from local testing during development, not your project's
  final results.
