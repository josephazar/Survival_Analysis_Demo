# Customer Churn Survival Analysis & CLV Prediction

End-to-end customer analytics pipeline using the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (~5,800 customers, Dec 2009 – Dec 2011). Covers churn definition, survival modeling, CLV prediction, and BTYD library comparison.

## Project Structure

```
├── customer-survival-analysis.ipynb   # Main survival analysis (7 parts)
├── clv-prediction-benchmark.ipynb     # ML vs probabilistic CLV comparison
├── churn-window-analysis.ipynb        # Empirical churn window selection
├── stage1-conversion-model.ipynb      # First-purchase conversion classifier
│
├── btyd_analysis/                     # BTYD library comparison pipeline
│   ├── 01_data_prep.py                # Cleaning & RFM aggregation
│   ├── 02_lifetimes_analysis.py       # Lifetimes BG/NBD + Gamma-Gamma
│   ├── 03_pymc_analysis.py            # PyMC-Marketing (MAP + MCMC)
│   └── 04_comparison.py               # Head-to-head evaluation
│
├── requirements.txt
└── .gitignore
```

## Notebooks

### 1. Customer Survival Analysis (`customer-survival-analysis.ipynb`)

The core notebook. Frames churn as a **survival problem** — "how long does a customer stay active?" rather than binary yes/no.

**Pipeline:**
- RFM summary + BG/NBD & Gamma-Gamma (lifetimes)
- PCA + K-Means segmentation (Gold / Silver / Bronze) using behavioral features only
- Kaplan-Meier survival curves per segment
- **5 survival models**: Cox PH, CoxNet (elastic net), Random Survival Forest, Gradient Boosting Survival, XGBoost AFT
- Comprehensive evaluation: IPCW C-index, time-dependent AUC, Brier score, IBS
- Customer Survival Scorecard with personalized S(t) predictions

**Key design decisions:**
- **One-timers excluded** from survival model training (handled via population KM curve in the scorecard)
- **45-day churn window** validated empirically in the churn-window notebook
- Clustering uses only behavioral features (frequency, recency, monetary, T) — no forward-looking lifetimes predictions that would leak churn status

**Results (repeat customers only):**

| Model | C-index (IPCW) | Mean TD-AUC | IBS |
|-------|----------------|-------------|-----|
| Random Survival Forest | 0.938 | 0.984 | 0.034 |
| CoxNet (Elastic Net) | 0.925 | 0.986 | 0.035 |
| Gradient Boosting | 0.922 | 0.968 | 0.118 |
| Cox PH (lifelines) | 0.921 | 0.985 | 0.041 |
| XGBoost AFT | 0.891 | 0.939 | 0.124 |

---

### 2. CLV Prediction Benchmark (`clv-prediction-benchmark.ipynb`)

Compares ML models against probabilistic BTYD models for 6-month CLV prediction using a temporal train/test split.

**Models compared:** Linear Regression, Random Forest, XGBoost vs Lifetimes (BG/NBD + Gamma-Gamma), PyMC-Marketing

**Key finding:** Lifetimes wins with MAE £494 vs XGBoost £553 — probabilistic models outperform ML with fair out-of-sample evaluation (cross_val_predict). Lifetimes also better detects churned customers (predicts £0 revenue).

---

### 3. Churn Window Analysis (`churn-window-analysis.ipynb`)

Empirically determines the optimal inactivity threshold for defining churn by analyzing the inter-purchase gap distribution.

**Method:** CDF of 30,918 inter-purchase gaps → elbow detection (Kneedle algorithm + max curvature) → cross-window validation at 30/40/45/50/60 days.

**Result:** Elbow at 43 days (captures 66% of returns). Label stability peaks at 45→50 days (98.4% agreement). Validates the **45-day churn window** used throughout the project.

---

### 4. Stage 1: Conversion Model (`stage1-conversion-model.ipynb`)

Binary classifier predicting whether a first-time buyer will ever return, using **only first-invoice features** (zero future leakage).

**Design:**
- 90-day observation window with censoring-aware labels (customers without enough observation time excluded)
- Temporal train/test split (train on earlier cohorts, test on later)
- 25 features: monetary, basket composition, temporal, geographic, product keywords

**Results:** XGBoost AUC-ROC = 0.62 — modest but honest signal from a single purchase. This model feeds into a two-stage framework: Stage 1 predicts conversion, Stage 2 (survival notebook) predicts churn timing for repeat customers.

---

## BTYD Library Comparison (`btyd_analysis/`)

Four-script pipeline comparing **lifetimes** vs **PyMC-Marketing** for BG/NBD + Gamma-Gamma modeling.

Run sequentially:
```bash
python btyd_analysis/01_data_prep.py
python btyd_analysis/02_lifetimes_analysis.py
python btyd_analysis/03_pymc_analysis.py
python btyd_analysis/04_comparison.py
```

**Finding:** Near-identical results (CLV correlation r=0.999, top-50 customer overlap 96%). Lifetimes is faster (0.06s vs 7.7s), PyMC provides posterior uncertainty.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place `online_retail_II.csv` in `../data/` relative to the project folder.

---

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 1M+ transactions from a UK online retailer. After cleaning: ~780K transactions, ~5,800 customers.

**Outputs** (generated, not tracked in git):
- `../data/customer_survival_scorecard.csv` — personalized survival predictions for all customers
- `../data/clv_benchmark_results.csv` — CLV model comparison per customer
- `btyd_analysis/*.parquet` — intermediate data and model results
- `btyd_analysis/*.png` — visualizations
