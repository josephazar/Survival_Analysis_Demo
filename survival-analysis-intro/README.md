# Survival Analysis Intro

A self-contained, six-notebook introduction to survival analysis and Cox
proportional hazards modeling, written for a reader who may be new to the
field but needs professional instincts: how to frame the business question,
define the event and time axis, handle censoring, avoid leakage, interpret
Kaplan-Meier curves, compare groups, fit Cox models, check assumptions, and
turn predictions into an action table whose probabilities mean "from today
forward."

It doubles as the conceptual on-ramp to the two full pipelines in this
repository (Online Retail II and Dunnhumby): every idea used there —
landmarking, censoring-aware labels, PH diagnostics, leakage tripwires,
and age-conditioned survival scoring — is introduced here from first
principles on small synthetic datasets.

## Notebook path

1. [`notebooks/01_what_is_survival_analysis.ipynb`](notebooks/01_what_is_survival_analysis.ipynb)
   Defines time zero, event, duration, censoring, and Kaplan-Meier from first principles.

2. [`notebooks/02_kaplan_meier_and_log_rank.ipynb`](notebooks/02_kaplan_meier_and_log_rank.ipynb)
   Uses a synthetic product-activation story to teach survival curves, cumulative event curves, group comparisons, and log-rank tests.

3. [`notebooks/03_cox_model_driver_analysis.ipynb`](notebooks/03_cox_model_driver_analysis.ipynb)
   Introduces Cox proportional hazards as driver analysis with a clock, including hazard ratios and persona-level survival curves.

4. [`notebooks/04_assumptions_validation_and_leakage.ipynb`](notebooks/04_assumptions_validation_and_leakage.ipynb)
   Covers held-out validation, proportional-hazards diagnostics, horizon calibration, and a deliberate leakage demonstration.

5. [`notebooks/05_end_to_end_retention_project.ipynb`](notebooks/05_end_to_end_retention_project.ipynb)
   Builds a day-30 landmark retention model with a time-based train/test split and exports a risk scorecard (written to `outputs/`, regenerated on each run).

6. [`notebooks/06_age_conditioned_forward_survival_scoring.ipynb`](notebooks/06_age_conditioned_forward_survival_scoring.ipynb)
   Shows why `S(Δ)` from a signup-time model is not a future probability for existing customers, then scores `S(age + Δ) / S(age)` with plots, `lifelines` checks, and a planning scorecard.

## Data

The CSV files under `data/` are **fully synthetic** — generated
programmatically for this module. No real customer data, and no third-party
material.

| File | Used by | Story |
|---|---|---|
| `tiny_customer_churn_timeline.csv` | 01 | A 12-customer timeline to make censoring tangible |
| `product_activation_survival.csv` | 02 | Product activation cohorts for KM + log-rank |
| `service_churn_drivers.csv` | 03, 04 | Service churn with known drivers for Cox + diagnostics |
| `subscription_retention_project.csv` | 05, 06 | Subscription panel for the end-to-end landmark project and age-conditioned scoring lesson |

## Run

From the repository root (after the `pip install -r requirements.txt` setup):

```bash
cd survival-analysis-intro
../venv/bin/python -m ipykernel install --user \
  --name survival-analysis-venv \
  --display-name "Python (Survival Analysis venv)"
../venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=survival-analysis-venv notebooks/*.ipynb
```

The notebooks are committed fully executed, so you can also just read them
top to bottom on GitHub.
