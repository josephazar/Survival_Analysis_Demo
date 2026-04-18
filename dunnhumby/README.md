# Dunnhumby Survival Analysis (v2, review-fixed)

Landmark-based survival modeling on the
[Dunnhumby "The Complete Journey"](https://www.dunnhumby.com/source-files/) panel.

**Start here if you're new:** [00_EDA_and_Business_Problem.ipynb](00_EDA_and_Business_Problem.ipynb) walks through the business problem, dataset, and plan of attack with embedded EDA plots.

**For the full design and v1 → v2 fix log**, see [DUNNHUMBY_PLAN.md](DUNNHUMBY_PLAN.md).

## Quick start

```bash
cd scripts
../../venv/bin/python 00_data_prep.py
../../venv/bin/python 01_eda.py
../../venv/bin/python 02_churn_window_analysis.py
../../venv/bin/python 03_feature_engineering.py
../../venv/bin/python 04_btyd_benchmark.py
../../venv/bin/python 05_early_dropout_model.py
../../venv/bin/python 06_customer_survival_analysis.py
../../venv/bin/python 07_clv_prediction_benchmark.py
../../venv/bin/python 08_household_scorecard.py
cd .. && ../venv/bin/python tests/test_leakage_and_smoke.py   # 14/14 assertions
```

## Key numbers

| Stage | Headline |
|---|---|
| Churn window (Kneedle) | 14 days |
| Landmark | DAY 500 (211-day follow-up) |
| Alive-at-landmark cohort | 1,767 / 2,500 households |
| Survival best model | CoxPH — test C-index **0.725**, TD-AUC 0.771, IBS 0.039 |
| CLV best model (180d) | LinearRegression — MAE **$338** |
| Early-dropout classifier | LogReg AUC **0.742** |
| Risk-tier split | Low 52% / Medium 11% / High 5% / Churned 32% |
| Leakage / smoke tests | **14/14 pass** |

## Outputs

- `processed/*.parquet` — canonical intermediates
- `artifacts/<stage>/` — per-stage plots and `metrics.json`
- `../../data/household_survival_scorecard.csv` — final 2,500-household risk export
