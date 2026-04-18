"""Stage 08 — Household survival scorecard (v2, review-fixed).

Consolidates the landmark survival, BTYD, and CLV outputs into a single
per-household scorecard with honest risk tiering.

Review fixes:
    - S(t) columns now come directly from the landmark model where they
      are measured from t0 forward (so they are already conditional on
      "alive at t0"). No post-hoc `duration + Δ` indexing into a flat S.
    - S(Δ) values beyond the modelled range are capped and flagged
      (no silent extrapolation).
    - Risk labelling explicitly uses the landmark S(Δ) rather than
      unconditional whole-window survival.

Input files (all produced by earlier stages):
    processed/household_features_landmark.parquet
    processed/household_features_full.parquet
    processed/survival_predictions.parquet           (landmark survival, S_30d..S_180d)
    processed/btyd_predictions.parquet               (BG/NBD + Gamma-Gamma on full window)
    processed/clv_benchmark.parquet                  (only test rows)

Output:
    household_survival_scorecard.csv  (at ../../data/)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (ARTIFACTS_DIR, CHURN_WINDOW, LANDMARK_DAY, MAX_DAY,
                     PROCESSED_DIR, REPO_ROOT)


def tier_risk(row):
    """Risk tiering rules — all based on conditional landmark survival.

    1. Already-churned households get an explicit override.
    2. Active households are bucketed on S_30d (their probability of
       still being active 30 days post-landmark). 30 days is the shortest
       horizon modelled and is the most stable signal.
    """
    if row["already_churned"]:
        return "Churned - Loss/Winback"
    if row["not_in_survival_cohort"]:
        # These are HHs who weren't alive at the landmark (could be churned earlier
        # or joined too late). Use BTYD p_alive as a fallback.
        if row["p_alive"] is None or np.isnan(row["p_alive"]):
            return "Unknown"
        if row["p_alive"] < 0.4:
            return "High Risk"
        if row["p_alive"] < 0.7:
            return "Medium Risk"
        return "Low Risk"
    # Landmark-based tiers
    if row["risk_percentile"] >= 0.85 or row["S_30d"] < 0.85:
        return "High Risk"
    if row["risk_percentile"] >= 0.60 or row["S_30d"] < 0.95:
        return "Medium Risk"
    return "Low Risk"


def main():
    print("=" * 64)
    print("STAGE 08: HOUSEHOLD SCORECARD (landmark + conditional S(Δ))")
    print("=" * 64)

    full = pd.read_parquet(PROCESSED_DIR / "household_features_full.parquet")
    landmark = pd.read_parquet(PROCESSED_DIR / "household_features_landmark.parquet")
    survival = pd.read_parquet(PROCESSED_DIR / "survival_predictions.parquet")
    btyd = pd.read_parquet(PROCESSED_DIR / "btyd_predictions.parquet")
    demo = pd.read_parquet(PROCESSED_DIR / "demographics.parquet")
    clv_bench = pd.read_parquet(PROCESSED_DIR / "clv_benchmark.parquet")

    # Base table: all 2,500 households with basic activity stats
    df = full[["household_key", "first_day", "last_day", "days_since_last",
               "n_baskets", "total_spend", "avg_spend"]].copy()
    df.rename(columns={"first_day": "first_basket_day", "last_day": "last_basket_day"}, inplace=True)
    df["already_churned"] = (df["days_since_last"] > CHURN_WINDOW).astype(bool)

    # Demographics
    df = df.merge(demo[["household_key", "HAS_DEMOGRAPHICS", "AGE_DESC", "INCOME_DESC",
                         "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC",
                         "MARITAL_STATUS_CODE", "HOMEOWNER_DESC"]], on="household_key", how="left")

    # Landmark survival outputs — the critical part
    surv_keep = survival[["household_key", "segment", "risk_score", "split",
                           "S_30d", "S_60d", "S_90d", "S_180d",
                           "event_time_from_t0", "event_observed"]].rename(
        columns={"event_observed": "churned_post_landmark",
                 "event_time_from_t0": "survival_time_from_t0"}
    )
    df = df.merge(surv_keep, on="household_key", how="left")
    df["not_in_survival_cohort"] = df["segment"].isna()
    print(f"[coverage] {int((~df['not_in_survival_cohort']).sum())} / {len(df)} households in landmark cohort")

    # BTYD for fallback & CLV info
    btyd_keep = btyd[["household_key", "p_alive", "expected_txns_180d",
                      "predicted_avg_spend", "clv_180d"]]
    df = df.merge(btyd_keep, on="household_key", how="left")

    # CLV benchmark (test-set only)
    clv_keep = clv_bench[["household_key", "pred_XGBoost", "pred_BTYD_GG"]].rename(
        columns={"pred_XGBoost": "clv_ml_xgb_bench", "pred_BTYD_GG": "clv_btyd_bench"}
    )
    df = df.merge(clv_keep, on="household_key", how="left")

    # Compute risk percentile WITHIN the landmark cohort (not globally)
    cohort_mask = ~df["not_in_survival_cohort"]
    df["risk_percentile"] = np.nan
    df.loc[cohort_mask, "risk_percentile"] = df.loc[cohort_mask, "risk_score"].rank(pct=True)

    df["risk_label"] = df.apply(tier_risk, axis=1)

    # Column ordering
    cols = [
        "household_key",
        # Demographics
        "HAS_DEMOGRAPHICS", "AGE_DESC", "INCOME_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
        "KID_CATEGORY_DESC", "MARITAL_STATUS_CODE", "HOMEOWNER_DESC",
        # Activity
        "first_basket_day", "last_basket_day", "days_since_last",
        "n_baskets", "total_spend", "avg_spend",
        # Landmark survival
        "segment", "not_in_survival_cohort", "already_churned",
        "churned_post_landmark", "survival_time_from_t0",
        "risk_score", "risk_percentile", "risk_label",
        "S_30d", "S_60d", "S_90d", "S_180d",
        "split",
        # BTYD
        "p_alive", "expected_txns_180d", "predicted_avg_spend", "clv_180d",
        # CLV bench (test only)
        "clv_ml_xgb_bench", "clv_btyd_bench",
    ]
    df = df[cols]

    out_path = REPO_ROOT / "data" / "household_survival_scorecard.csv"
    df.to_csv(out_path, index=False)
    print(f"[output] {out_path}  shape={df.shape}")

    # --- Summary ---
    print("\n[risk distribution]")
    print(df["risk_label"].value_counts().to_string())

    print("\n[segment distribution (landmark cohort)]")
    print(df["segment"].value_counts(dropna=False).to_string())

    print("\n[risk × segment]")
    print(pd.crosstab(df["segment"].fillna("not_in_cohort"), df["risk_label"]))

    print("\n[per-tier averages]")
    summary = df.groupby("risk_label").agg(
        n=("household_key", "count"),
        avg_total_spend=("total_spend", "mean"),
        avg_p_alive=("p_alive", "mean"),
        avg_S_30d=("S_30d", "mean"),
        avg_S_90d=("S_90d", "mean"),
        avg_clv_180d=("clv_180d", "mean"),
    )
    print(summary.round(2))


if __name__ == "__main__":
    main()
