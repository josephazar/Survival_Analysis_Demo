"""Stage 03 — Landmark-based household feature engineering.

REWRITE (v2). The original implementation had two methodology problems flagged
in review:

    1. `recency_ratio = duration / T_days` embedded the survival target in a
       feature (duration is the y-axis).
    2. Features like `*_last_90d` and `spend_trend` were windowed to MAX_DAY,
       so they implicitly encoded whether the household had stopped shopping
       by the end of the observation window — which is exactly the churn
       label.

This version uses a **landmark analysis**:

    - Pick a landmark day t0 (default: 500, leaving 211 days of follow-up).
    - Keep only households that are *alive at t0* — i.e. had a basket in
      (t0 - CHURN_WINDOW, t0].
    - All features use transactions on DAY <= t0. Zero contamination from
      post-landmark behaviour.
    - Survival target is measured from t0:
        event_observed = 1 if the household is inactive for more than
          CHURN_WINDOW days before MAX_DAY is reached.
        event_time = (last_basket_overall + CHURN_WINDOW) - t0     if event=1
                   = MAX_DAY - t0                                  if censored
      i.e. the event *is declared* when the inactivity threshold is crossed,
      not when the last basket occurred — this fixes the misaligned time
      axis issue.

Non-landmark (full-window) features are still emitted separately for the
BTYD benchmark and exploratory uses, clearly labelled `household_features_full.parquet`.

Outputs:
    processed/household_features_landmark.parquet    # for Stage 06 survival
    processed/household_features_full.parquet        # pre-existing, for BTYD / CLV
    artifacts/features/leakage_report.json           # correlation sanity check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ARTIFACTS_DIR, MAX_DAY, CHURN_WINDOW, LANDMARK_DAY

OUT = ARTIFACTS_DIR / "features"
OUT.mkdir(exist_ok=True, parents=True)


# ---------- Survival target (correct event times, landmark-anchored) -------
def compute_survival_target(baskets: pd.DataFrame, t0: int, max_day: int, window: int) -> pd.DataFrame:
    """Return household_key + event_time (days from t0) + event_observed.

    A household is eligible if its LAST pre-t0 basket is within `window` days of t0
    (i.e. they are still "alive" at t0 under the inactivity-based churn definition).

    Event time convention (the fix for the review P1):
        churned:   L_overall + window - t0   (the day the churn was DECLARED)
        censored:  max_day   - t0            (censored at the study end)
    """
    pre = baskets.loc[baskets["DAY"] <= t0, ["household_key", "DAY"]]
    last_pre = pre.groupby("household_key")["DAY"].max().rename("last_pre_t0")

    # Alive-at-t0 filter. Use `>= t0 - window` (not strictly >), because churn
    # is defined as inactivity *strictly greater than* CHURN_WINDOW days — so
    # a household whose last pre-t0 basket is exactly `t0 - window` has 14 days
    # of inactivity at t0, which is NOT > window and therefore still alive.
    alive_hh = last_pre[last_pre >= t0 - window].index

    # Overall last basket (<=MAX_DAY) for each alive household
    last_overall = (
        baskets[baskets["household_key"].isin(alive_hh)]
        .groupby("household_key")["DAY"].max().rename("last_overall")
    )
    out = pd.DataFrame(last_overall).join(last_pre, how="left")
    out["gap_to_end"] = max_day - out["last_overall"]
    out["event_observed"] = (out["gap_to_end"] > window).astype(int)
    out["event_day"] = np.where(
        out["event_observed"] == 1,
        out["last_overall"] + window,
        max_day,
    )
    out["event_time"] = (out["event_day"] - t0).clip(lower=1)  # strictly positive
    out = out.reset_index()
    return out[["household_key", "event_time", "event_observed", "last_pre_t0", "last_overall"]]


# ---------- Feature blocks (all computed from DAY <= t0) -------------------
def rfm_and_gaps_pre(baskets: pd.DataFrame, t0: int) -> pd.DataFrame:
    b = baskets[baskets["DAY"] <= t0].sort_values(["household_key", "DAY"]).copy()
    b["prev_day"] = b.groupby("household_key")["DAY"].shift(1)
    b["gap"] = b["DAY"] - b["prev_day"]

    agg = b.groupby("household_key").agg(
        first_day=("DAY", "min"),
        last_pre_day=("DAY", "max"),
        n_baskets_pre=("BASKET_ID", "nunique"),
        total_spend_pre=("SALES_VALUE", "sum"),
        avg_spend_pre=("SALES_VALUE", "mean"),
        std_spend_pre=("SALES_VALUE", "std"),
        max_spend_pre=("SALES_VALUE", "max"),
        avg_gap_pre=("gap", "mean"),
        std_gap_pre=("gap", "std"),
        median_gap_pre=("gap", "median"),
        p90_gap_pre=("gap", lambda s: s.dropna().quantile(0.9) if s.notna().any() else np.nan),
    ).reset_index()

    # Pre-landmark activity span & rate (NOT anchored to MAX_DAY)
    agg["activity_span_pre"] = agg["last_pre_day"] - agg["first_day"]  # days active before t0
    agg["T_pre"] = t0 - agg["first_day"] + 1                           # pre-landmark observation days
    agg["frequency_pre"] = agg["n_baskets_pre"] - 1
    agg["purchase_rate_pre"] = agg["n_baskets_pre"] / agg["T_pre"]
    agg["activity_span_ratio"] = agg["activity_span_pre"] / agg["T_pre"]

    # Last observed pre-t0 gap (most recent inter-basket interval)
    last_gap = b.dropna(subset=["gap"]).groupby("household_key")["gap"].last().rename("last_gap_pre")
    agg = agg.merge(last_gap, on="household_key", how="left")

    # Inactivity-at-landmark signal (days between last pre-t0 basket and t0)
    agg["days_inactive_at_t0"] = t0 - agg["last_pre_day"]

    # Windowed features anchored at t0 (NOT at MAX_DAY)
    last_30 = b[b["DAY"] > t0 - 30].groupby("household_key").agg(
        spend_last_30d_pre_t0=("SALES_VALUE", "sum"),
        baskets_last_30d_pre_t0=("BASKET_ID", "nunique"),
    )
    prior_30 = b[(b["DAY"] > t0 - 60) & (b["DAY"] <= t0 - 30)].groupby("household_key").agg(
        spend_prior_30d_pre_t0=("SALES_VALUE", "sum"),
        baskets_prior_30d_pre_t0=("BASKET_ID", "nunique"),
    )
    last_90 = b[b["DAY"] > t0 - 90].groupby("household_key").agg(
        spend_last_90d_pre_t0=("SALES_VALUE", "sum"),
        baskets_last_90d_pre_t0=("BASKET_ID", "nunique"),
    )
    agg = (agg
           .merge(last_30, on="household_key", how="left")
           .merge(prior_30, on="household_key", how="left")
           .merge(last_90, on="household_key", how="left"))
    for c in ["spend_last_30d_pre_t0", "baskets_last_30d_pre_t0",
              "spend_prior_30d_pre_t0", "baskets_prior_30d_pre_t0",
              "spend_last_90d_pre_t0", "baskets_last_90d_pre_t0"]:
        agg[c] = agg[c].fillna(0)
    agg["spend_trend_pre_t0"] = (agg["spend_last_30d_pre_t0"] + 1) / (agg["spend_prior_30d_pre_t0"] + 1)
    agg["basket_trend_pre_t0"] = (agg["baskets_last_30d_pre_t0"] + 1) / (agg["baskets_prior_30d_pre_t0"] + 1)

    return agg


def basket_composition_pre(baskets: pd.DataFrame, tx: pd.DataFrame, t0: int) -> pd.DataFrame:
    b = baskets[baskets["DAY"] <= t0]
    basket_agg = b.groupby("household_key").agg(
        avg_basket_qty=("QUANTITY", "mean"),
        avg_basket_lines=("N_LINES", "mean"),
        avg_unique_products=("N_UNIQUE_PRODUCTS", "mean"),
        avg_departments=("N_DEPARTMENTS", "mean"),
        avg_private_label_ratio=("PRIVATE_LABEL_RATIO", "mean"),
    ).reset_index()
    t = tx[tx["DAY"] <= t0]
    unique_all = t.groupby("household_key")["PRODUCT_ID"].nunique().rename("unique_products_total_pre").reset_index()
    return basket_agg.merge(unique_all, on="household_key", how="left")


def promo_responsiveness_pre(baskets: pd.DataFrame, campaigns: pd.DataFrame,
                             coupon_redempt: pd.DataFrame, campaign_desc: pd.DataFrame, t0: int) -> pd.DataFrame:
    b = baskets[baskets["DAY"] <= t0].copy()
    b["has_coupon_basket"] = (b["COUPON_DISC"].abs() > 0).astype(int)
    promo = b.groupby("household_key").agg(
        total_disc_pre=("TOTAL_DISC", lambda s: float(s.abs().sum())),
        avg_discount_depth_pre=("DISCOUNT_DEPTH", "mean"),
        pct_baskets_with_coupon_pre=("has_coupon_basket", "mean"),
    ).reset_index()

    # Coupon redemptions before t0
    cr = coupon_redempt[coupon_redempt["DAY"] <= t0]
    redempt = cr.groupby("household_key").agg(
        n_coupon_redemptions_pre=("COUPON_UPC", "count"),
        n_distinct_coupons_pre=("COUPON_UPC", "nunique"),
    ).reset_index()

    # Campaigns received before t0 — restrict to campaigns whose START_DAY <= t0
    cd = campaign_desc[campaign_desc["START_DAY"] <= t0]
    pre_campaigns = campaigns.merge(cd[["CAMPAIGN"]], on="CAMPAIGN", how="inner")
    camp = pre_campaigns.groupby("household_key").agg(
        n_campaigns_pre=("CAMPAIGN", "nunique"),
        n_typeA_pre=("DESCRIPTION", lambda s: (s == "TypeA").sum()),
        n_typeB_pre=("DESCRIPTION", lambda s: (s == "TypeB").sum()),
        n_typeC_pre=("DESCRIPTION", lambda s: (s == "TypeC").sum()),
    ).reset_index()

    out = promo.merge(redempt, on="household_key", how="left").merge(camp, on="household_key", how="left")
    for c in ["n_coupon_redemptions_pre", "n_distinct_coupons_pre",
              "n_campaigns_pre", "n_typeA_pre", "n_typeB_pre", "n_typeC_pre"]:
        out[c] = out[c].fillna(0)
    out["coupon_redempt_per_campaign_pre"] = (out["n_coupon_redemptions_pre"]
                                              / out["n_campaigns_pre"].replace(0, np.nan)).fillna(0)
    return out


def department_mix_pre(tx: pd.DataFrame, products: pd.DataFrame, t0: int) -> pd.DataFrame:
    t = tx[tx["DAY"] <= t0].merge(products[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="left")
    t["DEPARTMENT"] = t["DEPARTMENT"].fillna("UNKNOWN")
    top_depts = (
        t.groupby("DEPARTMENT")["SALES_VALUE"].sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    t["DEPT_GRP"] = np.where(t["DEPARTMENT"].isin(top_depts), t["DEPARTMENT"], "OTHER")
    spend_by_dept = t.groupby(["household_key", "DEPT_GRP"])["SALES_VALUE"].sum().unstack(fill_value=0.0)
    total = spend_by_dept.sum(axis=1)
    shares = spend_by_dept.div(total, axis=0)
    shares.columns = [f"share_{c.lower().replace(' ', '_').replace('-', '_')}_pre" for c in shares.columns]
    hhi = (shares ** 2).sum(axis=1).rename("dept_hhi_pre")
    unique_depts = t.groupby("household_key")["DEPARTMENT"].nunique().rename("unique_departments_pre")
    return shares.join(hhi).join(unique_depts).reset_index()


def demographic_features(demographics: pd.DataFrame) -> pd.DataFrame:
    d = demographics.copy()
    age_map = {"19-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6, "Unknown": np.nan}
    income_map = {
        "Under 15K": 1, "15-24K": 2, "25-34K": 3, "35-49K": 4,
        "50-74K": 5, "75-99K": 6, "100-124K": 7, "125-149K": 8,
        "150-174K": 9, "175-199K": 10, "200-249K": 11, "250K+": 12,
        "Unknown": np.nan,
    }
    size_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5, "Unknown": np.nan}
    d["age_ord"] = d["AGE_DESC"].map(age_map).fillna(-1)
    d["income_ord"] = d["INCOME_DESC"].map(income_map).fillna(-1)
    d["hh_size_ord"] = d["HOUSEHOLD_SIZE_DESC"].map(size_map).fillna(-1)
    d["homeowner_flag"] = (d["HOMEOWNER_DESC"].str.strip().str.lower() == "homeowner").astype(int)
    d["kids_flag"] = (~d["KID_CATEGORY_DESC"].isin(["None/Unknown", "Unknown"])).astype(int)
    return d[["household_key", "HAS_DEMOGRAPHICS", "age_ord", "income_ord", "hh_size_ord",
              "homeowner_flag", "kids_flag",
              "AGE_DESC", "INCOME_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
              "KID_CATEGORY_DESC", "MARITAL_STATUS_CODE", "HOMEOWNER_DESC"]]


# ---------- Legacy full-window feature matrix (retained for BTYD / EDA) ----
def build_full_window_features(baskets: pd.DataFrame, tx: pd.DataFrame,
                               products: pd.DataFrame, demographics: pd.DataFrame,
                               campaigns: pd.DataFrame, coupon_redempt: pd.DataFrame,
                               campaign_desc: pd.DataFrame) -> pd.DataFrame:
    """Original whole-window feature set. Kept for BTYD benchmark & scorecard joins,
    NOT for the survival model. Only exposes 'safe' summary stats here."""
    b = baskets.sort_values(["household_key", "DAY"]).copy()
    b["prev_day"] = b.groupby("household_key")["DAY"].shift(1)
    b["gap"] = b["DAY"] - b["prev_day"]
    agg = b.groupby("household_key").agg(
        first_day=("DAY", "min"),
        last_day=("DAY", "max"),
        n_baskets=("BASKET_ID", "nunique"),
        total_spend=("SALES_VALUE", "sum"),
        avg_spend=("SALES_VALUE", "mean"),
        total_qty=("QUANTITY", "sum"),
    ).reset_index()
    agg["T_days"] = MAX_DAY - agg["first_day"] + 1
    agg["days_since_last"] = MAX_DAY - agg["last_day"]
    agg["duration"] = agg["last_day"] - agg["first_day"]
    return agg


def main():
    print("=" * 64)
    print("STAGE 03: LANDMARK-BASED FEATURE ENGINEERING")
    print(f"  landmark day t0 = {LANDMARK_DAY},  churn window = {CHURN_WINDOW}d")
    print(f"  follow-up post-t0 = {MAX_DAY - LANDMARK_DAY} days")
    print("=" * 64)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet",
                          columns=["household_key", "PRODUCT_ID", "SALES_VALUE", "DAY"])
    products = pd.read_parquet(PROCESSED_DIR / "product.parquet")
    demographics = pd.read_parquet(PROCESSED_DIR / "demographics.parquet")
    campaigns = pd.read_parquet(PROCESSED_DIR / "campaign_exposure.parquet")
    coupon_redempt = pd.read_parquet(PROCESSED_DIR / "coupon_redempt.parquet")

    # Rebuild campaign_desc in-process (light file)
    from config import DATA_DIR
    campaign_desc = pd.read_csv(DATA_DIR / "campaign_desc.csv")

    # --- Survival target -----------------------------------------------------
    target = compute_survival_target(baskets, LANDMARK_DAY, MAX_DAY, CHURN_WINDOW)
    print(f"\n[eligibility] alive-at-t0 households: {len(target):,} / {baskets['household_key'].nunique():,}")
    print(f"[event rate]  {target['event_observed'].mean() * 100:.2f}%")
    print(f"[event_time]  min={target['event_time'].min()}  median={target['event_time'].median():.0f}  max={target['event_time'].max()}")

    # --- Pre-landmark features ----------------------------------------------
    rfm = rfm_and_gaps_pre(baskets, LANDMARK_DAY)
    comp = basket_composition_pre(baskets, tx, LANDMARK_DAY)
    promo = promo_responsiveness_pre(baskets, campaigns, coupon_redempt, campaign_desc, LANDMARK_DAY)
    dept = department_mix_pre(tx, products, LANDMARK_DAY)
    demo = demographic_features(demographics)

    features = (target
                .merge(rfm, on="household_key", how="left")
                .merge(comp, on="household_key", how="left")
                .merge(promo, on="household_key", how="left")
                .merge(dept, on="household_key", how="left")
                .merge(demo, on="household_key", how="left"))

    # NaN handling
    for col in ["last_gap_pre", "avg_gap_pre", "median_gap_pre", "std_gap_pre", "p90_gap_pre", "std_spend_pre"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    fill_neutral = {"spend_trend_pre_t0": 1.0, "basket_trend_pre_t0": 1.0}
    for col, v in fill_neutral.items():
        if col in features.columns:
            features[col] = features[col].fillna(v)

    features.to_parquet(PROCESSED_DIR / "household_features_landmark.parquet", index=False)
    print(f"\n[landmark features] shape={features.shape}")

    # --- Full-window features (for BTYD, CLV benchmark, scorecard join) -----
    full_feats = build_full_window_features(baskets, tx, products, demographics,
                                             campaigns, coupon_redempt, campaign_desc)
    full_feats.to_parquet(PROCESSED_DIR / "household_features_full.parquet", index=False)
    print(f"[full-window features] shape={full_feats.shape}")

    # --- Leakage sanity check ----------------------------------------------
    # Correlations between every candidate feature and the target time/event.
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("household_key", "event_time", "event_observed",
                                                   "last_pre_t0", "last_overall")]
    corr_time = features[num_cols + ["event_time"]].corr()["event_time"].drop("event_time").sort_values(key=lambda s: s.abs(), ascending=False)
    corr_event = features[num_cols + ["event_observed"]].corr()["event_observed"].drop("event_observed").sort_values(key=lambda s: s.abs(), ascending=False)

    leakage_report = {
        "landmark_day": LANDMARK_DAY,
        "eligible_households": int(len(features)),
        "event_rate": float(features["event_observed"].mean()),
        "event_time_stats": {
            "min": float(features["event_time"].min()),
            "median": float(features["event_time"].median()),
            "max": float(features["event_time"].max()),
        },
        "top_abs_corr_with_event_time": corr_time.head(10).round(4).to_dict(),
        "top_abs_corr_with_event_observed": corr_event.head(10).round(4).to_dict(),
        "features_with_abs_corr_gt_0p80": [
            c for c in num_cols
            if abs(features[c].corr(features["event_observed"])) > 0.80
               or abs(features[c].corr(features["event_time"])) > 0.80
        ],
    }
    with open(OUT / "leakage_report.json", "w") as fh:
        json.dump(leakage_report, fh, indent=2)
    print(f"\n[leakage report] top 5 |corr| with event_observed:")
    print(corr_event.head(5).round(4).to_string())
    print(f"\n[leakage report] features with |corr| > 0.80: {leakage_report['features_with_abs_corr_gt_0p80']}")
    print(f"\n[outputs]")
    print(f"  {PROCESSED_DIR / 'household_features_landmark.parquet'}")
    print(f"  {PROCESSED_DIR / 'household_features_full.parquet'}")
    print(f"  {OUT / 'leakage_report.json'}")


if __name__ == "__main__":
    main()
