"""Online Retail II — Customer scorecard.

Replaces `customer-survival-analysis.ipynb`'s cell-47 scorecard with:

    - S(30d), S(60d), S(90d) as CONDITIONAL forward-from-landmark
      probabilities (not unconditional S(t from first purchase)).
    - Landmark model for the alive-at-t0 cohort (1,269 customers).
    - Population Kaplan-Meier baseline filling in the remaining 4,609
      customers; provenance tracked in the `s_source` column.
    - Scorecard risk tiering that uses CoxPH landmark S_30d when
      available, falling back to BTYD p_alive otherwise.

Depends on output of 01_survival.py. Refits CoxPH on train+val before
scoring everyone.

Outputs:
    scripts/artifacts/online_retail_scorecard.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA_CSV = ROOT.parent.parent / "data" / "online_retail_II.csv"

LANDMARK_DATE = pd.Timestamp("2011-06-01")
CHURN_WINDOW = 45
RANDOM_STATE = 42

# Imports shared with 01_survival.py
import importlib.util
SPEC = importlib.util.spec_from_file_location("survival_mod", ROOT / "01_survival.py")
FS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FS)


def risk_tier(row):
    if row["already_churned"]:
        return "Churned - Loss/Winback"
    if row["not_in_cohort"]:
        if pd.isna(row["p_alive"]):
            return "Unknown"
        if row["p_alive"] < 0.4:
            return "High Risk"
        if row["p_alive"] < 0.7:
            return "Medium Risk"
        return "Low Risk"
    if row["S_30d"] < 0.6:
        return "High Risk"
    if row["S_30d"] < 0.85:
        return "Medium Risk"
    return "Low Risk"


def main():
    print("=" * 64)
    print("ONLINE RETAIL II — SCORECARD (conditional S(Δ))")
    print("=" * 64)

    df = FS.load_and_clean()
    obs_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # --- Per-customer summary (full window) ---
    summary = df.groupby("Customer ID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        num_invoices=("Invoice", "nunique"),
        total_spend=("LineTotal", "sum"),
    ).reset_index()
    summary["days_since_last"] = (obs_end - summary["last_purchase"]).dt.days
    summary["already_churned"] = (summary["days_since_last"] > CHURN_WINDOW).astype(bool)

    # --- Lifetimes BG/NBD on all customers (for p_alive fallback) ---
    bgf_rfm = summary_data_from_transaction_data(
        df, customer_id_col="Customer ID", datetime_col="InvoiceDate",
        monetary_value_col="LineTotal", observation_period_end=obs_end, freq="D",
    ).reset_index()
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(bgf_rfm["frequency"], bgf_rfm["recency"], bgf_rfm["T"])
    bgf_rfm["p_alive"] = bgf.conditional_probability_alive(
        bgf_rfm["frequency"], bgf_rfm["recency"], bgf_rfm["T"]
    )
    summary = summary.merge(bgf_rfm[["Customer ID", "p_alive"]], on="Customer ID", how="left")

    # --- Landmark survival model (CoxPH, refit on train+val) ---
    target = FS.compute_survival_target(df, LANDMARK_DATE, obs_end, CHURN_WINDOW)
    feats = FS.build_landmark_features(df, LANDMARK_DATE)
    data = target.merge(feats, on="Customer ID", how="inner")
    feat_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                 if c not in ("Customer ID", "event_time_days", "event_observed",
                               "last_pre_t0", "last_overall")]
    data = data.reset_index(drop=True)

    y = data[["event_time_days", "event_observed"]]
    X = data[feat_cols]
    idx_tv, idx_te = train_test_split(data.index, test_size=0.20, random_state=RANDOM_STATE,
                                        stratify=data["event_observed"])
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.25, random_state=RANDOM_STATE,
                                        stratify=data.loc[idx_tv, "event_observed"])
    idx_fit = np.concatenate([idx_tr, idx_va])
    imputer = SimpleImputer(strategy="median").fit(X.loc[idx_fit])
    scaler = StandardScaler().fit(imputer.transform(X.loc[idx_fit]))

    X_fit = scaler.transform(imputer.transform(X.loc[idx_fit]))
    X_all = scaler.transform(imputer.transform(X))
    cph_df_fit = pd.concat([pd.DataFrame(X_fit, columns=feat_cols, index=idx_fit),
                             y.loc[idx_fit]], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df_fit, duration_col="event_time_days", event_col="event_observed", show_progress=False)
    print(f"[cph refit] C-index on training data = {cph.concordance_index_:.4f}")

    # Conditional S(Δ): since event_time_days is measured from landmark,
    # S(Δ) is directly the survival function at Δ (i.e. P(alive at landmark+Δ | alive at landmark))
    deltas = [30, 60, 90]
    delta_cols = [f"S_{d}d" for d in deltas]
    # Cap deltas at the modelled range to avoid extrapolation
    eval_times = np.linspace(1, int(y["event_time_days"].max()) - 1, 50)
    t_max = float(eval_times.max())
    capped_deltas = [min(d, t_max) for d in deltas]
    if capped_deltas != deltas:
        print(f"[caution] capped deltas to {capped_deltas} (max modelled t = {t_max:.1f})")

    surv_df = cph.predict_survival_function(pd.DataFrame(X_all, columns=feat_cols), times=eval_times)
    # surv_df index = eval_times, columns = customer rows
    surv_arr = surv_df.values.T  # shape (n_customers, n_times)

    out_landmark = data[["Customer ID"]].copy()
    out_landmark["in_cohort"] = True
    out_landmark["risk_score"] = cph.predict_partial_hazard(
        pd.DataFrame(X_all, columns=feat_cols)
    ).values.ravel()
    for delta, cap in zip(deltas, capped_deltas):
        out_landmark[f"S_{delta}d"] = [np.interp(cap, eval_times, row) for row in surv_arr]

    # --- Population Kaplan-Meier baseline on the TRAINING fold ---
    # We use this to fill S_30d/S_60d/S_90d for customers outside the landmark
    # cohort (4,609 of 5,878 in the current run). Those customers are either
    # long-inactive or joined too recently, so a single conservative baseline
    # survival curve from the training cohort is the honest fallback.
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    y_train = y.loc[idx_tr]
    kmf.fit(y_train["event_time_days"], y_train["event_observed"], label="population")
    km_baseline = {delta: float(kmf.survival_function_at_times(min(delta, t_max)).iloc[0])
                   for delta in deltas}
    print(f"[km baseline (training cohort)] S_30d={km_baseline[30]:.3f}  "
          f"S_60d={km_baseline[60]:.3f}  S_90d={km_baseline[90]:.3f}")

    # Merge with full summary
    scorecard = summary.merge(out_landmark, on="Customer ID", how="left")
    scorecard["in_cohort"] = scorecard["in_cohort"].fillna(False)
    scorecard["not_in_cohort"] = ~scorecard["in_cohort"]
    scorecard["risk_percentile"] = scorecard["risk_score"].rank(pct=True)

    # Backfill S(Δ) for out-of-cohort customers with the KM baseline.
    # Separately track provenance in `s_source` so downstream consumers can
    # filter on it if they want model-based estimates only.
    scorecard["s_source"] = np.where(scorecard["in_cohort"], "cox_landmark", "km_baseline")
    for delta in deltas:
        col = f"S_{delta}d"
        scorecard[col] = scorecard[col].fillna(km_baseline[delta])

    scorecard["risk_label"] = scorecard.apply(risk_tier, axis=1)

    out_path = ART / "online_retail_scorecard.csv"
    scorecard.to_csv(out_path, index=False)
    print(f"\n[output] {out_path}  shape={scorecard.shape}")

    print("\n[risk distribution]")
    print(scorecard["risk_label"].value_counts().to_string())
    print("\n[cohort coverage]")
    print(f"  in_cohort (landmark survival):   {int(scorecard['in_cohort'].sum()):,} / {len(scorecard):,}")
    print(f"  already_churned:                {int(scorecard['already_churned'].sum()):,}")

    print("\n[per-tier averages]")
    print(scorecard.groupby("risk_label").agg(
        n=("Customer ID", "count"),
        avg_total_spend=("total_spend", "mean"),
        avg_p_alive=("p_alive", "mean"),
        avg_S_30d=("S_30d", "mean"),
    ).round(2))


if __name__ == "__main__":
    main()
