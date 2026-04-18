"""Stage 05 — Early-behavior churn classifier (v2, review-fixed).

The original Online-Retail Stage-1 "conversion model" (one-timers vs.
repeaters) doesn't fit Dunnhumby — only 3 of 2,500 households are one-timers.
We pose a more data-appropriate question:

    Given a household's FIRST 90 DAYS of shopping behaviour, can we
    predict whether they will eventually be flagged as churned at the
    end of the 711-day observation window (14-day inactivity rule)?

Review fixes:
    1. XGBoost now uses a VALIDATION split carved from the training set
       for early stopping — no more test-set contamination.
    2. Docstring now matches the actual code (percentile-based temporal
       split, not a fixed day).

Design:
    - Eligibility: first_day <= MAX_DAY − 90 − 180 (= 441) so we can label
      churn at the end of the window.
    - Features: only transactions where DAY <= first_day + 90.
    - Label: eventual churn flag from `household_features_full.parquet`
      — re-derived here as `(MAX_DAY - last_day) > CHURN_WINDOW`.
    - Temporal split on `first_day`:
        train : first_day <= p48   (~60% of eligible)
        val   : p48 < first_day <= p60   (~12%)
        test  : first_day > p60   (~28%)
    - Models: LogReg, Random Forest, XGBoost (val-based early stopping).
    - Metrics: ROC-AUC, PR-AUC, F1 at Youden-optimal threshold.

Outputs:
    processed/early_dropout_predictions.parquet
    artifacts/early_dropout/metrics.json, roc.png, pr_curve.png, feature_importance.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                              precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (ARTIFACTS_DIR, CHURN_WINDOW, MAX_DAY, PROCESSED_DIR,
                     RANDOM_STATE)

OUT = ARTIFACTS_DIR / "early_dropout"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110

EARLY_WINDOW_DAYS = 90
MIN_OBS_AFTER_WINDOW = 180


def build_early_features(baskets_all, tx, products, demo_numeric):
    first_days = baskets_all.groupby("household_key")["DAY"].min().rename("first_day").reset_index()
    b = baskets_all.merge(first_days, on="household_key")
    b["days_since_first"] = b["DAY"] - b["first_day"]
    b_early = b[b["days_since_first"] <= EARLY_WINDOW_DAYS].copy()

    early = b_early.groupby("household_key").agg(
        first_day=("first_day", "first"),
        early_baskets=("BASKET_ID", "nunique"),
        early_spend=("SALES_VALUE", "sum"),
        early_avg_basket_spend=("SALES_VALUE", "mean"),
        early_std_basket_spend=("SALES_VALUE", "std"),
        early_total_qty=("QUANTITY", "sum"),
        early_avg_qty=("QUANTITY", "mean"),
        early_avg_lines=("N_LINES", "mean"),
        early_avg_unique_products=("N_UNIQUE_PRODUCTS", "mean"),
        early_avg_departments=("N_DEPARTMENTS", "mean"),
        early_avg_private_ratio=("PRIVATE_LABEL_RATIO", "mean"),
        early_avg_discount_depth=("DISCOUNT_DEPTH", "mean"),
        early_pct_coupon_baskets=("HAS_COUPON", "mean"),
        early_last_day=("DAY", "max"),
    ).reset_index()
    early["early_active_range"] = early["early_last_day"] - early["first_day"]
    early["early_basket_rate"] = early["early_baskets"] / (early["early_active_range"] + 1)

    tmp = b_early.sort_values(["household_key", "DAY"])[["household_key", "BASKET_ID", "DAY"]].drop_duplicates(["household_key", "BASKET_ID"])
    tmp["prev_day"] = tmp.groupby("household_key")["DAY"].shift(1)
    tmp["gap"] = tmp["DAY"] - tmp["prev_day"]
    gap_stats = tmp.groupby("household_key")["gap"].agg(
        early_avg_gap="mean", early_std_gap="std", early_max_gap="max",
    ).reset_index()
    early = early.merge(gap_stats, on="household_key", how="left")

    tx_first = tx.merge(first_days, on="household_key")
    tx_first = tx_first[tx_first["DAY"] - tx_first["first_day"] <= EARLY_WINDOW_DAYS]
    n_unique_total = tx_first.groupby("household_key")["PRODUCT_ID"].nunique().rename("early_unique_products_total").reset_index()
    early = early.merge(n_unique_total, on="household_key", how="left")

    tx_prod = tx_first.merge(products[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="left")
    dept_diversity = tx_prod.groupby("household_key")["DEPARTMENT"].nunique().rename("early_unique_departments").reset_index()
    early = early.merge(dept_diversity, on="household_key", how="left")

    early = early.merge(demo_numeric, on="household_key", how="left")
    for c in ["early_avg_gap", "early_std_gap", "early_max_gap", "early_std_basket_spend"]:
        early[c] = early[c].fillna(0)
    return early


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler = StandardScaler()
    Xs_tr, Xs_va, Xs_te = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    results, models, probs = {}, {}, {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(Xs_tr, y_train)
    models["LogReg"] = lr
    probs["LogReg"] = lr.predict_proba(Xs_te)[:, 1]

    rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                 class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf
    probs["RandomForest"] = rf.predict_proba(X_test)[:, 1]

    pos, neg = y_train.sum(), (y_train == 0).sum()
    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=1,
        scale_pos_weight=max(neg / max(pos, 1), 1),
        random_state=RANDOM_STATE, eval_metric="auc",
        early_stopping_rounds=30,
    )
    # Early stopping on VALIDATION (not test) — P1 fix
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"       xgb best iter on val = {xgb.best_iteration}")
    models["XGBoost"] = xgb
    probs["XGBoost"] = xgb.predict_proba(X_test)[:, 1]

    for name, p in probs.items():
        auc = roc_auc_score(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        fpr, tpr, thr = roc_curve(y_test, p)
        youden = int(np.argmax(tpr - fpr))
        preds = (p >= thr[youden]).astype(int)
        results[name] = {
            "auc": float(auc), "pr_auc": float(pr_auc),
            "f1_at_youden": float(f1_score(y_test, preds)),
            "optimal_threshold": float(thr[youden]),
        }

    return models, results, probs


def main():
    print("=" * 64)
    print("STAGE 05: EARLY-BEHAVIOUR CHURN CLASSIFIER (val-based early stopping)")
    print("=" * 64)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet",
                          columns=["household_key", "DAY", "PRODUCT_ID", "SALES_VALUE"])
    products = pd.read_parquet(PROCESSED_DIR / "product.parquet")
    # Demographics: pull directly from the canonical demographics.parquet and encode
    # ordinals in-place, so this stage is independent of the landmark cohort.
    demo_raw = pd.read_parquet(PROCESSED_DIR / "demographics.parquet")
    age_map = {"19-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6, "Unknown": -1}
    income_map = {
        "Under 15K": 1, "15-24K": 2, "25-34K": 3, "35-49K": 4, "50-74K": 5,
        "75-99K": 6, "100-124K": 7, "125-149K": 8, "150-174K": 9,
        "175-199K": 10, "200-249K": 11, "250K+": 12, "Unknown": -1,
    }
    size_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5, "Unknown": -1}
    demo_raw["age_ord"] = demo_raw["AGE_DESC"].map(age_map).fillna(-1)
    demo_raw["income_ord"] = demo_raw["INCOME_DESC"].map(income_map).fillna(-1)
    demo_raw["hh_size_ord"] = demo_raw["HOUSEHOLD_SIZE_DESC"].map(size_map).fillna(-1)
    demo_raw["homeowner_flag"] = (demo_raw["HOMEOWNER_DESC"].str.strip().str.lower() == "homeowner").astype(int)
    demo_raw["kids_flag"] = (~demo_raw["KID_CATEGORY_DESC"].isin(["None/Unknown", "Unknown"])).astype(int)
    demo_numeric = demo_raw[["household_key", "HAS_DEMOGRAPHICS", "age_ord",
                              "income_ord", "hh_size_ord", "homeowner_flag", "kids_flag"]]
    # Need full-window label (eventual churn by MAX_DAY) — compute directly
    obs = baskets.groupby("household_key").agg(
        first_day=("DAY", "min"), last_day=("DAY", "max"), n_baskets=("BASKET_ID", "nunique"),
    ).reset_index()
    obs["event_observed"] = ((MAX_DAY - obs["last_day"]) > CHURN_WINDOW).astype(int)

    early = build_early_features(baskets, tx, products, demo_numeric)
    labeled = early.merge(obs[["household_key", "event_observed"]], on="household_key")
    print(f"[early features] shape={early.shape}")

    eligible = labeled[labeled["first_day"] <= MAX_DAY - EARLY_WINDOW_DAYS - MIN_OBS_AFTER_WINDOW].copy()
    excluded = len(labeled) - len(eligible)
    print(f"[eligibility] kept {len(eligible)}/{len(labeled)} households (excluded {excluded} late arrivals)")
    print(f"[class balance] churn rate = {eligible['event_observed'].mean() * 100:.2f}%")

    # --- Temporal 3-way split on first_day ---
    p48 = eligible["first_day"].quantile(0.48)
    p60 = eligible["first_day"].quantile(0.60)
    train = eligible[eligible["first_day"] <= p48].copy()
    val = eligible[(eligible["first_day"] > p48) & (eligible["first_day"] <= p60)].copy()
    test = eligible[eligible["first_day"] > p60].copy()
    print(f"[split] train <= day {p48:.0f} -> {len(train)}  |  val <= day {p60:.0f} -> {len(val)}  |  test > day {p60:.0f} -> {len(test)}")
    print(f"[split rates] train={train['event_observed'].mean() * 100:.1f}%  val={val['event_observed'].mean() * 100:.1f}%  test={test['event_observed'].mean() * 100:.1f}%")

    feature_cols = [c for c in eligible.columns
                    if c not in ("household_key", "event_observed", "first_day", "early_last_day")]
    X_train, y_train = train[feature_cols], train["event_observed"]
    X_val, y_val = val[feature_cols], val["event_observed"]
    X_test, y_test = test[feature_cols], test["event_observed"]

    models, results, probs = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n[results on test]")
    print(pd.DataFrame(results).T.round(4))

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — early-dropout classifier (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "roc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        ax.plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "pr_curve.png"); plt.close(fig)

    best_name = max(results, key=lambda k: results[k]["auc"])
    best = models[best_name]
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp.index, imp.values, color="#10b981")
        ax.set_title(f"Top features — {best_name}")
        fig.tight_layout(); fig.savefig(OUT / "feature_importance.png"); plt.close(fig)

    out_df = test[["household_key", "first_day", "event_observed"]].copy()
    for name, p in probs.items():
        out_df[f"pred_churn_proba_{name}"] = p
    out_df.to_parquet(PROCESSED_DIR / "early_dropout_predictions.parquet", index=False)

    with open(OUT / "metrics.json", "w") as fh:
        json.dump({
            "train_size": int(len(train)),
            "val_size": int(len(val)),
            "test_size": int(len(test)),
            "train_churn_rate": float(train["event_observed"].mean()),
            "val_churn_rate": float(val["event_observed"].mean()),
            "test_churn_rate": float(test["event_observed"].mean()),
            "n_features": len(feature_cols),
            "results": results,
            "best_model": best_name,
        }, fh, indent=2)

    print(f"\n[best] {best_name}: AUC={results[best_name]['auc']:.4f}")
    print(f"[output] {OUT}")


if __name__ == "__main__":
    main()
