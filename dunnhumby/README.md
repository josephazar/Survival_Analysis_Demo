# Dunnhumby Survival Analysis

Landmark-based survival modeling on the
[Dunnhumby "The Complete Journey"](https://www.dunnhumby.com/source-files/) panel:
2,500 US grocery households, ~2.6M transactions over two years.

**Start here if you're new:** [00_EDA_and_Business_Problem.ipynb](00_EDA_and_Business_Problem.ipynb) walks through the business problem, dataset, and plan of attack with embedded EDA plots.

For the methodology traps this pipeline is designed to avoid, see
[`../LESSONS_LEARNED.md`](../LESSONS_LEARNED.md).

## Quick start

```bash
cd scripts
for s in 00_data_prep 01_eda 02_churn_window_analysis 03_feature_engineering \
         04_btyd_benchmark 05_early_dropout_model 06_customer_survival_analysis \
         07_clv_prediction_benchmark 08_household_scorecard; do
  ../../venv/bin/python $s.py
done
cd .. && ../venv/bin/python tests/test_leakage_and_smoke.py   # expects 20/20 assertions
```

## Key numbers

| Stage | Headline |
|---|---|
| Churn window | **14 days** (Kneedle elbow = 16d, nearest candidate; robust to collapsing same-day baskets) |
| Landmark | DAY 500 (211-day follow-up) |
| Alive-at-landmark cohort | **1,794 / 2,500** households (event rate 23.8%) |
| Survival best model (selected on **validation** C-index) | CoxNet — test C-index **0.733**, TD-AUC 0.771, IBS 0.042 |
| Hazard-ratio inference | separate unpenalized Cox on a curated 9-feature subset; PH assumption checked (no violations at p<0.05) |
| Early-dropout classifier (val-selected) | LogReg — test AUC **0.743** |
| CLV best model, 180d horizon (val-selected) | LinearRegression — test MAE **$339** (BTYD $355) |
| BG/NBD caveat | dropout parameter a ≈ 0.01 on this dense panel → `p_alive` ≈ 1 for nearly everyone; flagged in stage 04 and not used as a churn signal |
| Risk-tier split | Low 51% / Medium 12% / High 5% / Churned 32% |
| Leakage / smoke tests | **20/20 pass** (includes recomputing features from raw pre-landmark data) |

Model selection happens on the validation fold; the test fold is scored
once and reported for all five models (CoxPH's test C-index of 0.738 is
slightly higher than CoxNet's, but CoxNet won on validation — reporting the
val-selected model is the honest protocol).

## Outputs

- `processed/*.parquet` — canonical intermediates
- `artifacts/<stage>/` — per-stage plots and `metrics.json`
- `../../data/household_survival_scorecard.csv` — final 2,500-household risk export
