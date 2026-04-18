# Customer Churn Survival Analysis & CLV Prediction

End-to-end customer analytics across two retail panels:

- **Online Retail II** — UK e-commerce, Dec 2009 – Dec 2011, ~5,800 customers.
- **Dunnhumby "The Complete Journey"** — US grocery, 2 years, 2,500 households, ~2.6M transactions.

Covers empirical churn-window selection, survival modelling (landmark design), CLV prediction (ML vs BTYD), and a BTYD library comparison (Lifetimes vs PyMC-Marketing).

> **New to survival analysis or this repo?** Start with [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) — a short, practical checklist of the mistakes caught in two rounds of code review. Saves you from repeating them.

---

## Project structure

```
├── scripts/                          # Online Retail Python pipeline (authoritative)
│   ├── 01_survival.py                # Landmark survival on the alive-at-t0 cohort
│   ├── 02_stage1.py                  # First-invoice conversion classifier (val-based XGB)
│   ├── 03_scorecard.py               # Full-coverage scorecard (Cox + KM baseline)
│   ├── patch_notebooks.py            # Adds banners + patches stage-1 cell in notebooks
│   ├── artifacts/                    # Generated metrics / plots / predictions
│   └── README.md
│
├── customer-survival-analysis.ipynb  # Original exploratory notebook (+ methodology banner)
├── clv-prediction-benchmark.ipynb    # ML vs BTYD — clean in review, unchanged
├── churn-window-analysis.ipynb       # Kneedle elbow for the churn window
├── stage1-conversion-model.ipynb     # Original conversion model (+ banner, cell 15 patched)
│
├── btyd_analysis/                    # BTYD library comparison (Lifetimes vs PyMC-Marketing)
│   ├── 01_data_prep.py
│   ├── 02_lifetimes_analysis.py
│   ├── 03_pymc_analysis.py
│   └── 04_comparison.py
│
├── dunnhumby/                        # Dunnhumby pipeline (see dunnhumby/README.md)
│   ├── scripts/                      # 9 stages, landmark-based
│   ├── tests/test_leakage_and_smoke.py   # 18 regression assertions
│   └── 00_EDA_and_Business_Problem.ipynb # Pedagogical front door
│
├── LESSONS_LEARNED.md                # Practical do-not-repeat checklist
├── SURVIVAL_ANALYSIS_GUIDE.md        # Glossary for readers new to survival analysis
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Additional for the Dunnhumby pipeline
pip install kneed shap
```

Place `online_retail_II.csv` and the Dunnhumby CSVs in `../data/` relative to this folder.

---

## Quick start

### Online Retail II pipeline

```bash
source venv/bin/activate
cd scripts
python 01_survival.py
python 02_stage1.py
python 03_scorecard.py
python patch_notebooks.py      # One-shot: updates notebooks with banners + stage-1 cell fix
```

Metrics land in `scripts/artifacts/`. Full per-customer scorecard at `scripts/artifacts/online_retail_scorecard.csv`.

### Dunnhumby pipeline

```bash
source venv/bin/activate
cd dunnhumby/scripts
for s in 00_data_prep 01_eda 02_churn_window_analysis 03_feature_engineering \
         04_btyd_benchmark 05_early_dropout_model 06_customer_survival_analysis \
         07_clv_prediction_benchmark 08_household_scorecard; do
  python $s.py
done
cd .. && python tests/test_leakage_and_smoke.py     # expects 18/18 passing
```

Final scorecard at `../data/household_survival_scorecard.csv`.

---

## The notebooks (Online Retail)

### `customer-survival-analysis.ipynb`

Original survival exploration — PCA+K-Means segmentation, Kaplan-Meier per tier, five survival models, scorecard. Retained for historical context. A banner at the top points to the authoritative `scripts/` pipeline, which fixes three issues caught in review:

1. Event/censoring time misalignment (should be `last + window` for churners, `study_end` for censored).
2. Retrospective features with random split (should be landmark + held-out test).
3. Scorecard `S(30d/60d/90d)` were unconditional from first purchase, not conditional forward-from-now.

### `clv-prediction-benchmark.ipynb`

Compares ML (Linear / RF / XGBoost) against probabilistic BTYD for 6-month CLV. Clean in review: real temporal cutoff, train-only feature building, BTYD fit on training only, `cross_val_predict` for ML. **Lifetimes wins** (MAE £494 vs XGBoost £553).

### `churn-window-analysis.ipynb`

Kneedle elbow on the inter-purchase gap CDF → 43 days → validates a **45-day** churn window for Online Retail. The same methodology applied to Dunnhumby yields **14 days**; see [`dunnhumby/scripts/02_churn_window_analysis.py`](dunnhumby/scripts/02_churn_window_analysis.py).

### `stage1-conversion-model.ipynb`

First-invoice → will-they-return binary classifier. 25 features, temporal split. Original XGBoost AUC 0.62. The stage-1 XGBoost cell (cell 15) has been patched in place to early-stop on a validation fold rather than the test fold (run `scripts/patch_notebooks.py` to apply).

### `btyd_analysis/` (Lifetimes vs PyMC-Marketing)

Four-script head-to-head BG/NBD + Gamma-Gamma comparison. Near-identical results: CLV correlation r=0.999, top-50 customer overlap 96%. Lifetimes is ~130× faster; PyMC provides posterior uncertainty.

---

## The Dunnhumby pipeline (`dunnhumby/`)

Nine-stage landmark pipeline with its own [`README.md`](dunnhumby/README.md). Headline numbers (grocery panel is much denser, so features and thresholds differ):

| Metric | Value |
|---|---|
| Churn window (Kneedle) | **14 days** (vs 45 for Online Retail) |
| Landmark | DAY 500 (211-day follow-up) |
| Alive-at-landmark cohort | 1,794 / 2,500 households |
| Best survival C-index (test) | **0.738** (CoxPH) |
| Stage-5 early-dropout classifier AUC | **0.742** (LogReg) |
| CLV best model (180d horizon) | **Linear Regression** — MAE $339 (ML beats BTYD here, the opposite of Online Retail) |
| Leakage + smoke tests | **18 / 18** pass |

The Dunnhumby work sits in its own folder because the dataset is richer (campaigns, coupons, demographics, product hierarchy) and the modelling decisions diverge from Online Retail at several points. See [`dunnhumby/00_EDA_and_Business_Problem.ipynb`](dunnhumby/00_EDA_and_Business_Problem.ipynb) for a narrative walkthrough.

---

## Before you start modifying code

1. **Read [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md).** It lists the specific traps this repo has already fallen into. Five minutes now saves a day later.
2. **Read [`SURVIVAL_ANALYSIS_GUIDE.md`](SURVIVAL_ANALYSIS_GUIDE.md)** if you haven't worked with survival models before.
3. **Run the Dunnhumby smoke tests (`dunnhumby/tests/test_leakage_and_smoke.py`) before and after your change.** They assert 18 invariants (no leakage, monotone survival, disjoint splits, correct event-time formulas). If you break one, the test name tells you what contract you violated.
4. **Don't re-introduce `duration = last - first` or `recency_ratio = duration / T`.** Both were real bugs that inflated metrics by ~0.25 C-index points.

---

## Output artefacts

**Generated, gitignored:**

- `scripts/artifacts/online_retail_scorecard.csv` — full 5,878-customer scorecard with `s_source` provenance column
- `scripts/artifacts/online_retail_survival_metrics.json` — test metrics for all survival models
- `scripts/artifacts/online_retail_stage1_*` — stage-1 metrics and predictions
- `dunnhumby/processed/*.parquet` — canonical Dunnhumby intermediates
- `dunnhumby/artifacts/<stage>/` — per-stage plots and metrics.json
- `../data/household_survival_scorecard.csv` — final Dunnhumby scorecard (2,500 × 34)
- `../data/customer_survival_scorecard.csv` — final Online Retail scorecard (generated by the notebook)
- `../data/clv_benchmark_results.csv` — CLV comparison per customer
- `btyd_analysis/*.parquet` / `*.png` — BTYD comparison artefacts

**Risk labels** used consistently across both projects:

- **Churned - Loss/Winback** — already inactive > `CHURN_WINDOW` at observation end
- **High Risk** — active but low survival probability
- **Medium Risk** — active with warning signs
- **Low Risk** — healthy active customer

---

## Data

- [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 1M+ transactions from a UK online retailer. After cleaning: ~780K transactions, ~5,800 customers.
- [Dunnhumby "The Complete Journey"](https://www.dunnhumby.com/source-files/) — 2-year panel of 2,500 US households, ~2.6M transactions, 92K products, 801 households with demographics, 30 campaigns.
