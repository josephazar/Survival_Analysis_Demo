# Online Retail II — Python pipeline

The four top-level notebooks at `../` are the original exploratory work. The
Python scripts in this folder are the authoritative pipeline — they fix the
methodology issues raised in the code review (misaligned event time,
retrospective feature engineering, test-set early stopping, unconditional
scorecard survival, missing KM backfill).

The notebooks carry a banner pointing here and the stage-1 XGBoost cell has
been surgically patched in place by `patch_notebooks.py`.

## Scripts

| File | Role |
|---|---|
| `01_survival.py` | Landmark survival on the alive-at-t0 cohort (CoxPH / CoxNet / RSF / GBSA) |
| `02_stage1.py` | Stage-1 conversion model with the full 25-feature notebook set and val-based XGBoost early stopping |
| `03_scorecard.py` | Full-coverage scorecard: Cox landmark S(Δ) on the cohort, population KM baseline everywhere else |
| `patch_notebooks.py` | Adds methodology banners + patches the stage-1 XGBoost cell in the notebooks |

## Run

```bash
cd scripts
../venv/bin/python 01_survival.py
../venv/bin/python 02_stage1.py
../venv/bin/python 03_scorecard.py
../venv/bin/python patch_notebooks.py
```

## Headline numbers

| Metric | Original notebook | This pipeline |
|---|---|---|
| Landmark | none (retrospective) | 2011-06-01 (192d follow-up) |
| Alive-at-landmark cohort | n/a | 1,269 / 5,878 customers |
| Best survival C-index on test | 0.894 (CoxNet) | 0.758 (RSF) |
| Stage-1 XGBoost test AUC (25-feature set, isolated early-stop fix) | 0.6204 (test leak) | 0.6097 (val-based) |
| Scorecard S(Δ) | Unconditional from first purchase | Conditional from landmark |
| Scorecard coverage on S columns | — | 100% (cohort 1,269 + KM baseline 4,609) |

The C-index / AUC drops are the honest cost of removing the misaligned time
axis, retrospective features, and test peek. The scorecard is now fully
populated and carries a `s_source` column (`cox_landmark` or `km_baseline`)
so consumers can filter by provenance.

## CLV benchmark

The CLV comparison in `../clv-prediction-benchmark.ipynb` was not flagged
in review and is unchanged. It uses a real temporal cutoff, features from
training data only, BTYD fit on training only, and `cross_val_predict` for
fair ML evaluation.

## Leakage notes

The original Online Retail `recency_ratio = last_gap / avg_gap` is **not**
the same feature as Dunnhumby v1's `recency_ratio = duration / T_days` —
the former uses gap statistics, the latter embeds the survival target in
the numerator. The correlation of the Online Retail feature with
`event_observed` is moderate, not near-identity, so this pipeline didn't
suffer the 0.99-C-index pathology that Dunnhumby did. Even so, the time-axis
and scorecard issues still warranted this rewrite.
