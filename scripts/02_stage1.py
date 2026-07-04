"""Online Retail II — Stage-1 conversion model.

First-invoice features -> will the customer place a second order within 90 days?

Methodology guarantees:
    - Fixed 90-day conversion horizon for BOTH train and test labels. Without a
      fixed horizon, train customers (up to ~24 months of follow-up) and test
      customers (~3-6 months) would carry incomparable labels, and train labels
      would use outcomes from after the split cutoff.
    - Temporal split: every test customer retains >= 90 days of follow-up
      before the data ends, so no test label is right-censored.
    - A validation slice carved from train drives XGBoost early stopping,
      classification-threshold selection (Youden's J), and best-model
      selection. The test fold is scored exactly once.

Feature set (shared with stage1-conversion-model.ipynb):
    - Monetary:          first_total_spend_log, first_avg_price, first_max_price,
                          first_price_range, first_price_std
    - Basket:            first_num_lines, first_unique_items, first_total_qty,
                          first_avg_qty, first_qty_per_unique
    - Temporal:          first_hour, first_day_of_week, first_is_weekend,
                          first_month, first_is_holiday_season
    - Geographic:        is_uk, geo_Europe_Major, geo_Europe_Other
    - Product keywords:  has_christmas, has_gift, has_vintage, has_heart
    - Diversity:         first_desc_diversity
    - Concentration:     first_spend_concentration

Temporal split:
    train : first_purchase <= 2011-06-01
    test  : 2011-06-01 < first_purchase <= 2011-09-10
Validation slice: 15% of train (random 85/15, stratified). All models fit on the
remaining 85% so the validation fold is genuinely held out from fitting.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
DATA_CSV = ROOT.parent.parent / "data" / "online_retail_II.csv"

TRAIN_END = pd.Timestamp("2011-06-01")       # matches notebook cutoff
TEST_END = pd.Timestamp("2011-09-10")        # must have 90d observation remaining
VAL_FRACTION = 0.15                          # of training set; early stopping + threshold + model selection
CONVERSION_WINDOW_DAYS = 90                  # fixed label horizon for all cohorts
RANDOM_STATE = 42

EUROPE_MAJOR = {"France", "Germany", "Spain", "Italy", "Netherlands", "Belgium",
                 "Switzerland", "Austria", "Portugal", "Ireland"}
EUROPE_OTHER = {"Sweden", "Denmark", "Finland", "Norway", "Poland", "Czech Republic",
                 "Greece", "Cyprus", "Malta"}
CHRISTMAS_WORDS = ("CHRISTMAS", "XMAS", "SANTA", "NOEL", "ADVENT", "NATIVITY")
GIFT_WORDS = ("GIFT", "PRESENT")
VINTAGE_WORDS = ("VINTAGE", "RETRO", "ANTIQUE")
HEART_WORDS = ("HEART",)


def load_and_clean():
    df = pd.read_csv(DATA_CSV, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["Invoice"].astype(str).str.startswith("C") == False]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]
    return df


def build_first_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 25-feature notebook set from each customer's FIRST invoice."""
    first_invoice = (df.sort_values("InvoiceDate")
                       .groupby("Customer ID")["Invoice"].first()
                       .rename("first_invoice"))
    first_df = df.merge(first_invoice.reset_index(), on="Customer ID")
    first_df = first_df[first_df["Invoice"] == first_df["first_invoice"]].copy()
    first_df["Description"] = first_df["Description"].fillna("").astype(str).str.upper()

    feat = first_df.groupby("Customer ID").agg(
        first_total_spend=("LineTotal", "sum"),
        first_avg_price=("Price", "mean"),
        first_max_price=("Price", "max"),
        first_min_price=("Price", "min"),
        first_price_std=("Price", "std"),
        first_num_lines=("StockCode", "count"),
        first_unique_items=("StockCode", "nunique"),
        first_total_qty=("Quantity", "sum"),
        first_avg_qty=("Quantity", "mean"),
        first_date=("InvoiceDate", "first"),
        Country=("Country", "first"),
    )
    feat["first_price_std"] = feat["first_price_std"].fillna(0)
    feat["first_price_range"] = feat["first_max_price"] - feat["first_min_price"]
    feat["first_qty_per_unique"] = feat["first_total_qty"] / feat["first_unique_items"]
    feat["first_total_spend_log"] = np.log1p(feat["first_total_spend"])
    feat["first_hour"] = feat["first_date"].dt.hour
    feat["first_day_of_week"] = feat["first_date"].dt.dayofweek
    feat["first_is_weekend"] = (feat["first_day_of_week"] >= 5).astype(int)
    feat["first_month"] = feat["first_date"].dt.month
    feat["first_is_holiday_season"] = feat["first_month"].isin([10, 11, 12]).astype(int)

    # Geography (one-hot matches notebook)
    feat["is_uk"] = (feat["Country"] == "United Kingdom").astype(int)
    feat["geo_Europe_Major"] = feat["Country"].isin(EUROPE_MAJOR).astype(int)
    feat["geo_Europe_Other"] = feat["Country"].isin(EUROPE_OTHER).astype(int)

    # Description-based product flags & diversity. Compute on first_df then roll up.
    desc_flags = first_df.groupby("Customer ID").agg(
        first_desc_diversity=("Description", "nunique"),
        has_christmas=("Description", lambda s: int(any(any(w in d for w in CHRISTMAS_WORDS) for d in s))),
        has_gift=("Description", lambda s: int(any(any(w in d for w in GIFT_WORDS) for d in s))),
        has_vintage=("Description", lambda s: int(any(any(w in d for w in VINTAGE_WORDS) for d in s))),
        has_heart=("Description", lambda s: int(any(any(w in d for w in HEART_WORDS) for d in s))),
    )
    feat = feat.join(desc_flags)

    # Spend concentration — Herfindahl over line items (one line = one product × qty)
    first_df["LineShare"] = first_df["LineTotal"] / first_df.groupby("Customer ID")["LineTotal"].transform("sum")
    first_df["LineShareSq"] = first_df["LineShare"] ** 2
    conc = first_df.groupby("Customer ID")["LineShareSq"].sum().rename("first_spend_concentration")
    feat = feat.join(conc)

    # Drop helper columns before return
    feat = feat.drop(columns=["Country", "first_min_price"])
    return feat.reset_index()


def build_label(df: pd.DataFrame, observation_end: pd.Timestamp,
                conversion_window_days: int = CONVERSION_WINDOW_DAYS) -> pd.DataFrame:
    """Label = second invoice within `conversion_window_days` of the first.

    A fixed horizon keeps labels comparable across cohorts: every included
    customer is observed for at least the full window, and conversions after
    the window do not count regardless of how long the customer was tracked.
    """
    inv = (df.groupby(["Customer ID", "Invoice"], as_index=False)["InvoiceDate"].min()
             .sort_values(["Customer ID", "InvoiceDate"]))
    inv["inv_rank"] = inv.groupby("Customer ID").cumcount()
    first = inv.loc[inv["inv_rank"] == 0].set_index("Customer ID")["InvoiceDate"].rename("first_date")
    second = inv.loc[inv["inv_rank"] == 1].set_index("Customer ID")["InvoiceDate"].rename("second_date")
    n_inv = inv.groupby("Customer ID").size().rename("n_invoices")
    agg = pd.concat([first, second, n_inv], axis=1).reset_index()

    agg["obs_days"] = (observation_end - agg["first_date"]).dt.days
    # Everyone must have the full window of follow-up, or a "no conversion"
    # label could just mean "not observed long enough".
    agg = agg[agg["obs_days"] >= conversion_window_days].copy()
    days_to_second = (agg["second_date"] - agg["first_date"]).dt.days
    agg["label"] = (days_to_second <= conversion_window_days).fillna(False).astype(int)
    return agg[["Customer ID", "first_date", "n_invoices", "obs_days", "label"]]


def main():
    print("=" * 64)
    print("ONLINE RETAIL II — STAGE-1 CONVERSION (val-based early stopping)")
    print("=" * 64)

    df = load_and_clean()
    obs_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    labels = build_label(df, obs_end)
    feats = build_first_invoice_features(df)

    data = labels.merge(feats.drop(columns=["first_date"]), on="Customer ID", how="inner")
    print(f"[data] {len(data)} customers, label mean = {data['label'].mean():.3f}")

    # Temporal split: train <= 2011-06-01, test in (2011-06-01, 2011-09-10]
    train_full = data[data["first_date"] <= TRAIN_END].copy()
    test = data[(data["first_date"] > TRAIN_END) & (data["first_date"] <= TEST_END)].copy()

    # Carve a validation slice from TRAIN ONLY. All models fit on the remaining
    # 85%, so validation stays out of fitting and can be reused for early
    # stopping, threshold selection, and model selection.
    from sklearn.model_selection import train_test_split as _tts
    train, val = _tts(train_full, test_size=VAL_FRACTION, random_state=RANDOM_STATE,
                       stratify=train_full["label"])
    print(f"[split] train_full={len(train_full)} -> train={len(train)}  val={len(val)}   test={len(test)}")
    print(f"[split rates] train={train['label'].mean():.3f}  val={val['label'].mean():.3f}  test={test['label'].mean():.3f}")

    feat_cols = [c for c in feats.columns if c not in ("Customer ID", "first_date")]
    X_train, y_train = train[feat_cols], train["label"]
    X_val, y_val = val[feat_cols], val["label"]
    X_test, y_test = test[feat_cols], test["label"]

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_va = scaler.transform(X_val)
    Xs_te = scaler.transform(X_test)

    results, probs, val_probs = {}, {}, {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(Xs_tr, y_train)
    val_probs["LogReg"] = lr.predict_proba(Xs_va)[:, 1]
    probs["LogReg"] = lr.predict_proba(Xs_te)[:, 1]

    rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                 class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_probs["RandomForest"] = rf.predict_proba(X_val)[:, 1]
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
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[xgb] best iter on val = {xgb.best_iteration}")
    val_probs["XGBoost"] = xgb.predict_proba(X_val)[:, 1]
    probs["XGBoost"] = xgb.predict_proba(X_test)[:, 1]

    for name, p in probs.items():
        # Youden's J threshold chosen on VALIDATION, applied frozen to test.
        fpr_v, tpr_v, thr_v = roc_curve(y_val, val_probs[name])
        j = int(np.argmax(tpr_v - fpr_v))
        threshold = float(thr_v[j])
        results[name] = {
            "auc": float(roc_auc_score(y_test, p)),
            "pr_auc": float(average_precision_score(y_test, p)),
            "val_auc": float(roc_auc_score(y_val, val_probs[name])),
            "f1_at_val_threshold": float(f1_score(y_test, (p >= threshold).astype(int))),
            "threshold_from_val": threshold,
        }

    print("\n[test results]")
    print(pd.DataFrame(results).T.round(4))

    # Winner is picked on validation AUC; its test AUC is the reported number.
    best = max(results, key=lambda k: results[k]["val_auc"])
    print(f"\n[best by val AUC] {best}: val={results[best]['val_auc']:.4f}  test={results[best]['auc']:.4f}")

    out = test[["Customer ID", "first_date", "label"]].copy()
    for name, p in probs.items():
        out[f"pred_{name}"] = p
    out.to_parquet(ART / "online_retail_stage1_predictions.parquet", index=False)

    with open(ART / "online_retail_stage1_metrics.json", "w") as fh:
        json.dump({
            "train_end": str(TRAIN_END.date()),
            "test_end": str(TEST_END.date()),
            "train_full_size": int(len(train_full)),
            "train_size": int(len(train)),
            "val_size": int(len(val)),
            "test_size": int(len(test)),
            "train_label_rate": float(train_full["label"].mean()),
            "val_label_rate": float(val["label"].mean()),
            "test_label_rate": float(test["label"].mean()),
            "feature_set": sorted(feat_cols),
            "n_features": len(feat_cols),
            "conversion_window_days": CONVERSION_WINDOW_DAYS,
            "results": results,
            "best_model_by_val_auc": best,
            "methodology_note": ("Fixed 90-day conversion horizon for all cohorts; "
                                 "temporal split (train <= 2011-06-01, test in "
                                 "(2011-06-01, 2011-09-10]); all models fit on 85% of "
                                 "train; validation drives early stopping, Youden "
                                 "threshold, and model selection; test scored once."),
        }, fh, indent=2)


if __name__ == "__main__":
    main()
