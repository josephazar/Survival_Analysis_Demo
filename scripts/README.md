# Online Retail II — Python pipeline

The notebooks at `../` are the exploratory, narrative versions of this work.
The scripts in this folder are the **authoritative pipeline**: a landmark
survival design with aligned event times, strictly pre-landmark features,
validation-based tuning, and a conditional scorecard.

## Scripts

| File | Role |
|---|---|
| `01_survival.py` | Landmark survival on the alive-at-t0 cohort (CoxPH / CoxNet / RSF / GBSA), IPCW-evaluated on a held-out test fold; persists the split assignment |
| `02_stage1.py` | First-invoice conversion classifier with a fixed 90-day label horizon; validation drives early stopping, thresholds, and model selection |
| `03_scorecard.py` | Full-coverage scorecard: Cox landmark S(Δ) on the cohort, population-KM baseline for active post-landmark joiners, NaN (not a fake number) for churned out-of-cohort customers |

## Run

```bash
cd scripts
../venv/bin/python 01_survival.py
../venv/bin/python 02_stage1.py
../venv/bin/python 03_scorecard.py
```

## Design guarantees

- **Landmark = 2011-06-01** (192-day follow-up), churn window = 45 days
  (empirically derived in `../churn-window-analysis.ipynb`).
- Event time for churners is `last_purchase + 45d`, censoring at study end —
  both measured from the landmark, so `S(Δ)` is conditional on being alive
  at the landmark by construction.
- Features come strictly from transactions on or before the landmark.
- Evaluation: IPCW C-index / time-dependent AUC / integrated Brier on a
  locked test fold; hyperparameters chosen on validation.
- The stage-1 label is "second invoice within 90 days of the first" for
  every cohort — a fixed horizon, so train and test labels are comparable
  and no label uses information from after the split cutoff.

## Headline numbers (retrospective design vs this pipeline)

| Metric | Retrospective notebook | This pipeline |
|---|---|---|
| Landmark | none (features span the full window) | 2011-06-01 (192d follow-up) |
| Alive-at-landmark cohort | n/a | 1,269 / 5,878 customers |
| Best survival C-index on test | ~0.89 | **0.758** (RSF; CoxPH 0.750) |
| Stage-1 test AUC | 0.62 (ever-converted label, test-fold early stopping) | **0.62** (fixed 90-day label, val-based tuning — RandomForest) |
| Scorecard S(Δ) | unconditional from first purchase | conditional forward-from-landmark, dated by `s_asof` |
| Scorecard S-coverage | — | cohort 1,269 via Cox + active joiners via KM baseline; churned customers get NaN by design |

The survival C-index drop is not a regression — it is the size of the
retrospective bias the landmark design removes. The scorecard carries an
`s_source` column (`cox_landmark` / `km_baseline` / `none_churned`) so
consumers can filter by provenance, and a `New/Unproven` tier for one-time
buyers, for whom BG/NBD's `p_alive` equals 1.0 by construction and is
therefore masked rather than reported.

## CLV benchmark

The CLV comparison lives in `../clv-prediction-benchmark.ipynb`: real
temporal cutoff, features from the training window only, BTYD fit on
training data only, `cross_val_predict` (with in-pipeline scaling) for the
ML models. Lifetimes wins on MAE (£494 vs £553 for XGBoost).

## Leakage notes

The Online Retail feature `recency_gap_ratio = last_gap / avg_gap` is built
from **pre-landmark gap statistics** — unlike a `duration / T` ratio, it does
not embed the survival target. Every feature is additionally screened by a
correlation tripwire (`|corr| > 0.7` against event or event time fails the
run); the real defence is the landmark construction itself.
