"""Stage 07 — CLV prediction benchmark (v2, review-fixed).

Addresses the review findings:

    P1 — BTYD previously fit on ALL households before the train/test split,
         giving it population information from test rows.
    P1 — XGBoost previously used the held-out test fold for early stopping.

Fixes:
    1. Split households FIRST (stratified 70/30 by spend quartile).
    2. Carve a validation slice from train (60/15/25 of the full dataset).
    3. Fit BTYD (BG/NBD + Gamma-Gamma) ONLY on training households' RFM
       data from the calibration window.
    4. Fit ML models on training features only. XGBoost uses the
       VALIDATION fold for early stopping. Test set is touched once at
       final evaluation.

Temporal split (unchanged): calibration DAY 1..450, holdout DAY 451..630,
buffer 631..711.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import ARTIFACTS_DIR, MAX_DAY, PROCESSED_DIR, RANDOM_STATE, SYNTHETIC_EPOCH

warnings.filterwarnings("ignore")
OUT = ARTIFACTS_DIR / "clv"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110

CALIBRATION_END = 450
HOLDOUT_END = 630


def build_ml_features(baskets: pd.DataFrame, tx: pd.DataFrame, products: pd.DataFrame,
                      cal_end: int) -> pd.DataFrame:
    b = baskets[baskets["DAY"] <= cal_end].copy()
    feats = b.groupby("household_key").agg(
        n_baskets=("BASKET_ID", "nunique"),
        total_spend=("SALES_VALUE", "sum"),
        avg_spend=("SALES_VALUE", "mean"),
        std_spend=("SALES_VALUE", "std"),
        first_day=("DAY", "min"),
        last_day=("DAY", "max"),
        avg_basket_lines=("N_LINES", "mean"),
        avg_unique_products=("N_UNIQUE_PRODUCTS", "mean"),
        avg_departments=("N_DEPARTMENTS", "mean"),
        avg_private_label_ratio=("PRIVATE_LABEL_RATIO", "mean"),
        total_disc=("TOTAL_DISC", lambda s: float(s.abs().sum())),
    ).reset_index()

    for days, suffix in [(30, "30d"), (90, "90d")]:
        sub = b[b["DAY"] > cal_end - days].groupby("household_key").agg(
            **{f"spend_last_{suffix}": ("SALES_VALUE", "sum"),
                f"baskets_last_{suffix}": ("BASKET_ID", "nunique")}
        ).reset_index()
        feats = feats.merge(sub, on="household_key", how="left")
    for c in [c for c in feats.columns if c.startswith(("spend_last_", "baskets_last_"))]:
        feats[c] = feats[c].fillna(0)

    bb = b.sort_values(["household_key", "DAY"]).drop_duplicates(["household_key", "BASKET_ID"])
    bb["prev"] = bb.groupby("household_key")["DAY"].shift(1)
    bb["gap"] = bb["DAY"] - bb["prev"]
    gap = bb.groupby("household_key")["gap"].agg(avg_gap="mean", std_gap="std", last_gap="last").reset_index()
    feats = feats.merge(gap, on="household_key", how="left")
    for c in ["avg_gap", "std_gap", "last_gap"]:
        feats[c] = feats[c].fillna(0)

    feats["T_days"] = cal_end - feats["first_day"] + 1
    feats["days_since_last_at_cutoff"] = cal_end - feats["last_day"]
    feats["std_spend"] = feats["std_spend"].fillna(0)
    return feats


def build_holdout_target(baskets: pd.DataFrame, cal_end: int, hold_end: int) -> pd.DataFrame:
    h = baskets[(baskets["DAY"] > cal_end) & (baskets["DAY"] <= hold_end)]
    return h.groupby("household_key")["SALES_VALUE"].sum().rename("holdout_spend").reset_index()


def fit_btyd_on_hhset(baskets: pd.DataFrame, train_hh: set, cal_end: int):
    """Fit BG/NBD + Gamma-Gamma on TRAINING households only, using their
    calibration-window transactions. Returns fitted models and the training
    RFM table (fitted parameters are population-level, not household-specific)."""
    cal = baskets[(baskets["DAY"] <= cal_end) & (baskets["household_key"].isin(train_hh))].copy()
    cal_end_date = pd.to_datetime(SYNTHETIC_EPOCH) + pd.Timedelta(days=cal_end - 1)
    rfm_train = summary_data_from_transaction_data(
        cal, customer_id_col="household_key",
        datetime_col="INVOICE_DATE",
        monetary_value_col="SALES_VALUE",
        observation_period_end=cal_end_date,
        freq="D",
    ).reset_index()

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm_train["frequency"], rfm_train["recency"], rfm_train["T"])
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    repeat = rfm_train[(rfm_train["frequency"] > 0) & (rfm_train["monetary_value"] > 0)]
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return bgf, ggf


def btyd_predict_for(baskets: pd.DataFrame, hh_set: set, bgf, ggf, cal_end: int,
                     horizon_days: int) -> pd.DataFrame:
    """Score arbitrary households (train or test) using the TRAIN-fitted BTYD models.
    Each household's own calibration RFM is passed in to generate the prediction."""
    cal = baskets[(baskets["DAY"] <= cal_end) & (baskets["household_key"].isin(hh_set))].copy()
    cal_end_date = pd.to_datetime(SYNTHETIC_EPOCH) + pd.Timedelta(days=cal_end - 1)
    rfm = summary_data_from_transaction_data(
        cal, customer_id_col="household_key",
        datetime_col="INVOICE_DATE",
        monetary_value_col="SALES_VALUE",
        observation_period_end=cal_end_date,
        freq="D",
    ).reset_index()

    can_score = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
    pred_txns = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, rfm["frequency"], rfm["recency"], rfm["T"],
    )
    pred_avg = pd.Series(rfm["monetary_value"].values, index=rfm.index)
    if can_score.sum() > 0:
        pred_avg.loc[can_score] = ggf.conditional_expected_average_profit(
            rfm.loc[can_score, "frequency"], rfm.loc[can_score, "monetary_value"]
        ).values
    pred_spend = np.maximum((pred_txns * pred_avg).values, 0)
    return pd.DataFrame({"household_key": rfm["household_key"].values,
                         "pred_clv_btyd": pred_spend})


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 else np.nan,
    }


def main():
    print("=" * 64)
    print("STAGE 07: CLV BENCHMARK (split-first, train-only fitting)")
    print(f"  calibration 1..{CALIBRATION_END}, holdout {CALIBRATION_END + 1}..{HOLDOUT_END}")
    print("=" * 64)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    baskets["INVOICE_DATE"] = pd.to_datetime(baskets["INVOICE_DATE"])
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet",
                          columns=["household_key", "DAY", "PRODUCT_ID", "SALES_VALUE"])
    products = pd.read_parquet(PROCESSED_DIR / "product.parquet")

    feats = build_ml_features(baskets, tx, products, CALIBRATION_END)
    target = build_holdout_target(baskets, CALIBRATION_END, HOLDOUT_END)
    df = feats.merge(target, on="household_key", how="left")
    df["holdout_spend"] = df["holdout_spend"].fillna(0)
    print(f"[dataset] {len(df)} households")

    # ----- STEP 1: SPLIT FIRST (households) -----
    bins = pd.qcut(df["total_spend"].rank(method="first"), q=4, labels=False)
    trainval_df, test_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=bins,
    )
    bins_tv = pd.qcut(trainval_df["total_spend"].rank(method="first"), q=4, labels=False)
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.20, random_state=RANDOM_STATE, stratify=bins_tv,
    )
    train_hh = set(train_df["household_key"])
    val_hh = set(val_df["household_key"])
    test_hh = set(test_df["household_key"])
    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ----- STEP 2: FIT BTYD ON TRAINING HOUSEHOLDS ONLY -----
    bgf, ggf = fit_btyd_on_hhset(baskets, train_hh, CALIBRATION_END)
    horizon = HOLDOUT_END - CALIBRATION_END
    btyd_test = btyd_predict_for(baskets, test_hh, bgf, ggf, CALIBRATION_END, horizon)
    test_df = test_df.merge(btyd_test, on="household_key", how="left")
    test_df["pred_clv_btyd"] = test_df["pred_clv_btyd"].fillna(0)

    # ----- STEP 3: FIT ML ON TRAINING, VALIDATE ON VAL, TEST ONCE -----
    ml_features = [c for c in feats.columns if c != "household_key"]
    X_train, y_train = train_df[ml_features], train_df["holdout_spend"]
    X_val, y_val = val_df[ml_features], val_df["holdout_spend"]
    X_test, y_test = test_df[ml_features], test_df["holdout_spend"]

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_va = scaler.transform(X_val)
    Xs_te = scaler.transform(X_test)

    results = {}
    preds = {}

    lr = LinearRegression()
    lr.fit(Xs_tr, y_train)
    preds["LinearRegression"] = lr.predict(Xs_te)

    rf = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds["RandomForest"] = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        random_state=RANDOM_STATE, early_stopping_rounds=30,
    )
    # Early stopping on VALIDATION (not test)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[xgb] best iter on val = {xgb.best_iteration}")
    preds["XGBoost"] = xgb.predict(X_test)

    preds["BTYD_GG"] = test_df["pred_clv_btyd"].values

    for name, p in preds.items():
        results[name] = evaluate(y_test.values, p)
        r = results[name]
        print(f"  [{name:18s}] MAE=${r['mae']:7.2f}  RMSE=${r['rmse']:7.2f}  R²={r['r2']:6.3f}  r={r['pearson_r']:.3f}")

    # --- Plot scatter ---
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y_test.values, p, s=8, alpha=0.35, color="#3b82f6")
        lim = max(y_test.max(), p.max())
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_title(f"{name}\nMAE=${results[name]['mae']:.0f}")
        ax.set_xlabel("Actual spend ($)")
    axes[0].set_ylabel("Predicted spend ($)")
    fig.tight_layout(); fig.savefig(OUT / "scatter.png"); plt.close(fig)

    # --- Persist ---
    out_df = test_df[["household_key", "total_spend", "holdout_spend"]].copy()
    for name, p in preds.items():
        out_df[f"pred_{name}"] = p
    out_df.to_parquet(PROCESSED_DIR / "clv_benchmark.parquet", index=False)

    # Also persist the full train / val / test household assignment so downstream
    # tests can directly verify disjointness rather than inferring from sizes.
    assignment = pd.concat([
        train_df[["household_key"]].assign(split="train"),
        val_df[["household_key"]].assign(split="val"),
        test_df[["household_key"]].assign(split="test"),
    ], ignore_index=True)
    assignment.to_parquet(PROCESSED_DIR / "clv_split_assignment.parquet", index=False)

    with open(OUT / "metrics.json", "w") as fh:
        json.dump({
            "calibration_end_day": CALIBRATION_END,
            "holdout_end_day": HOLDOUT_END,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "results": results,
        }, fh, indent=2)

    best = min(results, key=lambda k: results[k]["mae"])
    print(f"\n[best] {best}  (MAE ${results[best]['mae']:.2f})")
    print(f"[output] {PROCESSED_DIR / 'clv_benchmark.parquet'}")


if __name__ == "__main__":
    main()
