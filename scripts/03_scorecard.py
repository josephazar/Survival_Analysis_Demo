"""Online Retail II — Customer scorecard.

Full-coverage scorecard where:

    - S(30d), S(60d), S(90d) are survival probabilities measured FORWARD
      FROM THE LANDMARK (see `s_asof` column), conditional on being alive
      at the landmark. They are NOT forecasts from the end of the data.
      Status fields (`days_since_last`, `already_churned`, `p_alive`) are
      anchored at observation end; the tier logic gates on `already_churned`
      first, so a stale landmark forecast never drives the tier of a
      customer who has lapsed since the landmark.
    - The landmark CoxPH model (refit on train+val, never test) scores the
      alive-at-landmark cohort. A population Kaplan-Meier baseline covers
      post-landmark joiners who are still active. Customers who are outside
      the cohort AND already churned get NaN survival columns — a healthy
      baseline number would be misleading for them.
    - Provenance tracked in `s_source`: cox_landmark / km_baseline /
      none_churned.

Depends on output of 01_survival.py (including its persisted split
assignment, so the test fold is never used for refitting).

Outputs:
    scripts/artifacts/online_retail_scorecard.csv
"""
from __future__ import annotations

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    # Thresholds are business-chosen operating points, not fitted quantities.
    # Note the two fallback scales (Cox S_30d vs BG/NBD p_alive) are not
    # calibrated against each other; `s_source` records which one applied.
    if row["already_churned"]:
        return "Churned - Loss/Winback"
    if row["not_in_cohort"]:
        # p_alive is NaN here for one-time buyers: BG/NBD returns exactly 1.0
        # by construction at frequency=0 (a model artifact, not an estimate),
        # so it is masked upstream and those customers are labelled
        # New/Unproven instead of a fake "Low Risk".
        if pd.isna(row["p_alive"]):
            return "New/Unproven"
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
    summary = summary.merge(bgf_rfm[["Customer ID", "frequency", "p_alive"]],
                             on="Customer ID", how="left")
    # BG/NBD's p_alive is exactly 1.0 by construction for frequency-0 (no
    # repeat purchase) customers. Mask it: no estimate is better than a fake
    # perfectly-healthy one.
    summary.loc[summary["frequency"] == 0, "p_alive"] = np.nan

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
    # Reuse the exact split persisted by 01_survival.py so the test fold is
    # never used for refitting. Fall back to re-deriving it with the same
    # seed only if 01 has not been run.
    split_path = ART / "online_retail_split_assignment.parquet"
    if split_path.exists():
        split_map = (pd.read_parquet(split_path)
                       .set_index("Customer ID")["split"])
        splits = data["Customer ID"].map(split_map)
        idx_tr = data.index[splits == "train"].to_numpy()
        idx_fit = data.index[splits.isin(["train", "val"])].to_numpy()
        print(f"[split] loaded from {split_path.name}: "
              f"fit(train+val)={len(idx_fit)}  held-out test={int((splits == 'test').sum())}")
    else:
        print("[split] WARNING: split assignment not found; re-deriving with the 01_survival.py seed")
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
    print(f"[cph refit] in-sample C-index (fit diagnostic only — test metrics live in "
          f"online_retail_survival_metrics.json) = {cph.concordance_index_:.4f}")

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
    # Fills S_30d/S_60d/S_90d ONLY for customers outside the landmark cohort
    # who are still active at obs_end (post-landmark joiners). Out-of-cohort
    # customers who already churned keep NaN — the KM curve describes
    # alive-at-landmark customers and would hand a long-dead customer a
    # healthy-looking S(Δ).
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

    # Backfill S(Δ) with the KM baseline only for active out-of-cohort
    # customers; churned out-of-cohort customers keep NaN. Provenance in
    # `s_source` so downstream consumers can filter to model-based estimates.
    scorecard["s_source"] = np.select(
        [scorecard["in_cohort"].to_numpy(), (~scorecard["already_churned"]).to_numpy()],
        ["cox_landmark", "km_baseline"],
        default="none_churned",
    )
    backfill = scorecard["s_source"] == "km_baseline"
    for delta in deltas:
        col = f"S_{delta}d"
        scorecard.loc[backfill, col] = scorecard.loc[backfill, col].fillna(km_baseline[delta])

    # All S(Δ) columns are anchored at the landmark — date them explicitly.
    scorecard["s_asof"] = np.where(scorecard["s_source"] == "none_churned",
                                    "", LANDMARK_DATE.date().isoformat())

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
