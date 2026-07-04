"""Online Retail II — landmark survival analysis.

Replaces the original `customer-survival-analysis.ipynb` survival stage
with a landmark-based design:

    - Features are built strictly from transactions on or before the
      landmark date — no retrospective contamination.
    - Event time is measured from the landmark forward (churn declared
      at last_purchase + window; censored at study end).
    - Evaluation uses a 60/20/20 train/val/test split. The val fold is
      used for hyperparameter selection (CoxNet alpha, GBSA n_estimators);
      the test fold is touched once. The split is random (event-stratified)
      rather than temporal because this is a single-landmark design: all
      customers share the same calendar follow-up window.

Caveat worth knowing: with inactivity-defined churn, events declared in the
first `window` days after the landmark are partly mechanical — a customer
already inactive at t0 who never returns gets event_time = last_pre + window.
Features may legitimately predict this (it is not leakage), but very early
event times should not be read as fresh post-landmark behaviour.

Data:   ../data/online_retail_II.csv  (relative to the repo root)
Dates:  Dec-2009..Dec-2011
Landmark t0:   2011-06-01  (observation_end - 192 days)
Follow-up:     2011-06-01..2011-12-09
Churn window:  45 days (empirically validated in the churn-window notebook)

Outputs:
    scripts/artifacts/online_retail_survival_metrics.json
    scripts/artifacts/online_retail_km.png
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc,
                             integrated_brier_score)
from sksurv.util import Surv

# Silence known-noisy deprecation chatter only; model warnings (e.g. Cox
# convergence) stay visible — they are diagnostics, not noise.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
DATA_CSV = ROOT.parent.parent / "data" / "online_retail_II.csv"

LANDMARK_DATE = pd.Timestamp("2011-06-01")
CHURN_WINDOW = 45
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 110


def load_and_clean():
    df = pd.read_csv(DATA_CSV, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Clean: remove cancellations, zero/negative qty/price, null customer ID
    df = df[df["Invoice"].astype(str).str.startswith("C") == False]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]
    print(f"[data] {len(df):,} rows, {df['Customer ID'].nunique():,} customers")
    print(f"[data] date range {df['InvoiceDate'].min()} .. {df['InvoiceDate'].max()}")
    return df


def compute_survival_target(df: pd.DataFrame, landmark: pd.Timestamp, obs_end: pd.Timestamp,
                             window_days: int) -> pd.DataFrame:
    """Same correct formulation used on Dunnhumby: alive-at-landmark filter,
    event time is the day churn is DECLARED (last + window), censored at study end."""
    pre = df[df["InvoiceDate"] <= landmark]
    last_pre = pre.groupby("Customer ID")["InvoiceDate"].max().rename("last_pre_t0")
    # Use `>=` not `>` — a customer whose last pre-landmark purchase is exactly
    # `landmark - window_days` is still alive (inactivity = window_days is not > window_days).
    alive = last_pre[last_pre >= landmark - pd.Timedelta(days=window_days)].index

    last_overall = (df[df["Customer ID"].isin(alive)]
                    .groupby("Customer ID")["InvoiceDate"].max().rename("last_overall"))
    out = pd.DataFrame(last_overall).join(last_pre, how="left")
    out["gap_to_end_days"] = (obs_end - out["last_overall"]).dt.days
    out["event_observed"] = (out["gap_to_end_days"] > window_days).astype(int)
    event_date = np.where(out["event_observed"] == 1,
                           out["last_overall"] + pd.Timedelta(days=window_days),
                           obs_end)
    event_date = pd.to_datetime(pd.Series(event_date, index=out.index))
    out["event_time_days"] = ((event_date - landmark) / np.timedelta64(1, "D")).astype(int).clip(lower=1)
    return out.reset_index()[["Customer ID", "event_time_days", "event_observed",
                                "last_pre_t0", "last_overall"]]


def build_landmark_features(df: pd.DataFrame, landmark: pd.Timestamp) -> pd.DataFrame:
    """Features built strictly from transactions with InvoiceDate <= landmark."""
    pre = df[df["InvoiceDate"] <= landmark].copy()
    # Invoice-level aggregation
    invoice_agg = (pre.groupby(["Customer ID", "Invoice"], as_index=False)
                      .agg(InvoiceDate=("InvoiceDate", "min"),
                            TotalSpend=("LineTotal", "sum"),
                            TotalQty=("Quantity", "sum"),
                            NumItems=("StockCode", "nunique"))
                      .sort_values(["Customer ID", "InvoiceDate"]))

    feat = invoice_agg.groupby("Customer ID").agg(
        num_transactions_pre=("Invoice", "count"),
        total_spend_pre=("TotalSpend", "sum"),
        avg_spend_pre=("TotalSpend", "mean"),
        std_spend_pre=("TotalSpend", "std"),
        total_qty_pre=("TotalQty", "sum"),
        avg_basket_size_pre=("TotalQty", "mean"),
        avg_items_per_visit_pre=("NumItems", "mean"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase_pre=("InvoiceDate", "max"),
    )
    feat["std_spend_pre"] = feat["std_spend_pre"].fillna(0)

    invoice_agg["PrevDate"] = invoice_agg.groupby("Customer ID")["InvoiceDate"].shift(1)
    invoice_agg["Gap"] = (invoice_agg["InvoiceDate"] - invoice_agg["PrevDate"]).dt.days
    gap = invoice_agg.dropna(subset=["Gap"]).groupby("Customer ID")["Gap"].agg(
        avg_gap_pre="mean", std_gap_pre="std",
        min_gap_pre="min", max_gap_pre="max", last_gap_pre="last",
    )
    gap["std_gap_pre"] = gap["std_gap_pre"].fillna(0)
    feat = feat.merge(gap, left_index=True, right_index=True, how="left")
    for c in ["avg_gap_pre", "std_gap_pre", "min_gap_pre", "max_gap_pre", "last_gap_pre"]:
        feat[c] = feat[c].fillna(0)

    # Recency as gap-based (not target-based) ratio
    feat["recency_gap_ratio"] = np.where(
        feat["avg_gap_pre"] > 0, feat["last_gap_pre"] / feat["avg_gap_pre"], 0
    )
    feat["gap_acceleration_pre"] = feat["last_gap_pre"] - feat["avg_gap_pre"]

    last_spend = invoice_agg.groupby("Customer ID")["TotalSpend"].last()
    feat["monetary_trend_pre"] = np.where(
        feat["avg_spend_pre"] > 0, last_spend / feat["avg_spend_pre"], 1
    )
    feat["total_spend_log_pre"] = np.log1p(feat["total_spend_pre"])

    # Windowed features anchored at t0 (not at obs_end)
    for w in [30, 90]:
        sub = invoice_agg[invoice_agg["InvoiceDate"] > landmark - pd.Timedelta(days=w)]
        agg = sub.groupby("Customer ID").agg(
            **{f"spend_last_{w}d_pre": ("TotalSpend", "sum"),
                f"invoices_last_{w}d_pre": ("Invoice", "count")}
        )
        feat = feat.merge(agg, left_index=True, right_index=True, how="left")
        for c in [f"spend_last_{w}d_pre", f"invoices_last_{w}d_pre"]:
            feat[c] = feat[c].fillna(0)

    # Pre-landmark observation length in days
    feat["T_pre_days"] = (landmark - feat["first_purchase"]).dt.days.clip(lower=1)
    feat["purchase_rate_pre"] = feat["num_transactions_pre"] / feat["T_pre_days"]

    # Drop non-numeric columns (dates)
    feat = feat.drop(columns=["first_purchase", "last_purchase_pre"])
    return feat.reset_index()


def main():
    print("=" * 64)
    print("ONLINE RETAIL II — LANDMARK SURVIVAL")
    print(f"  landmark = {LANDMARK_DATE.date()}  churn window = {CHURN_WINDOW}d")
    print("=" * 64)

    df = load_and_clean()
    obs_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    print(f"[obs_end] {obs_end}")

    target = compute_survival_target(df, LANDMARK_DATE, obs_end, CHURN_WINDOW)
    print(f"[eligible] alive-at-t0 customers: {len(target):,}")
    print(f"[event rate] {target['event_observed'].mean() * 100:.2f}%")
    print(f"[event_time] min={target['event_time_days'].min()}  median={target['event_time_days'].median():.0f}  max={target['event_time_days'].max()}")

    feats = build_landmark_features(df, LANDMARK_DATE)

    data = target.merge(feats, on="Customer ID", how="inner")
    feat_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                 if c not in ("Customer ID", "event_time_days", "event_observed",
                               "last_pre_t0", "last_overall")]
    # Drop the raw "last_*" columns from features if they snuck in
    feat_cols = [c for c in feat_cols if "last_pre_t0" not in c and "last_overall" not in c]

    print(f"[features] {len(feat_cols)} cols, n={len(data)}")

    # --- Leakage smoke check (linear correlation with the event flag only;
    # it will not catch nonlinear or timing leakage — the design guarantees
    # above are the real defence, this is a tripwire) ---
    leaks = []
    for c in feat_cols:
        if abs(data[c].corr(data["event_observed"])) > 0.7:
            leaks.append((c, float(data[c].corr(data["event_observed"]))))
    print(f"[leakage check] features with |corr(event)| > 0.7: {leaks}")

    # --- 60/20/20 split, event-stratified ---
    y = data[["event_time_days", "event_observed"]]
    X = data[feat_cols]
    idx_tv, idx_te = train_test_split(data.index, test_size=0.20, random_state=RANDOM_STATE,
                                        stratify=data["event_observed"])
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.25, random_state=RANDOM_STATE,
                                        stratify=data.loc[idx_tv, "event_observed"])
    print(f"[split] train={len(idx_tr)}  val={len(idx_va)}  test={len(idx_te)}")

    # Persist the split so downstream stages (03_scorecard.py) reuse the exact
    # same folds instead of re-deriving them.
    split_assign = pd.DataFrame({
        "Customer ID": data["Customer ID"],
        "split": np.select([data.index.isin(idx_tr), data.index.isin(idx_va)],
                            ["train", "val"], default="test"),
    })
    split_assign.to_parquet(ART / "online_retail_split_assignment.parquet", index=False)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X.loc[idx_tr]))
    Xva = scaler.transform(imputer.transform(X.loc[idx_va]))
    Xte = scaler.transform(imputer.transform(X.loc[idx_te]))

    y_tr = y.loc[idx_tr]
    y_va = y.loc[idx_va]
    y_te = y.loc[idx_te]

    def to_surv(yd):
        return Surv.from_arrays(event=yd["event_observed"].astype(bool).values,
                                 time=yd["event_time_days"].astype(float).values)
    y_tr_s, y_va_s, y_te_s = to_surv(y_tr), to_surv(y_va), to_surv(y_te)

    t_min = int(max(y_te["event_time_days"].min(), y_tr["event_time_days"].min()) + 1)
    t_max = int(min(y_te["event_time_days"].max(), y_tr["event_time_days"].max()) - 1)
    eval_times = np.linspace(t_min, t_max, 25)
    print(f"[eval_times] {eval_times.round(1).tolist()}")

    metrics = {}

    # CoxPH (lifelines)
    print("\n[1/4] CoxPH")
    cph_df = pd.concat([pd.DataFrame(Xtr, columns=feat_cols, index=idx_tr), y_tr], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df, duration_col="event_time_days", event_col="event_observed", show_progress=False)
    risk_te = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=feat_cols)).values.ravel()
    # predict_survival_function(times=eval_times) is already on the eval grid
    surv_mat = cph.predict_survival_function(pd.DataFrame(Xte, columns=feat_cols), times=eval_times).values.T

    c = concordance_index_ipcw(y_tr_s, y_te_s, risk_te)[0]
    _, td_mean = cumulative_dynamic_auc(y_tr_s, y_te_s, risk_te, eval_times)
    ibs = integrated_brier_score(y_tr_s, y_te_s, surv_mat, eval_times)
    metrics["CoxPH"] = {"c_index_ipcw": float(c), "mean_td_auc": float(td_mean), "ibs": float(ibs)}
    print(f"  CoxPH  C={c:.4f}  TD-AUC={td_mean:.4f}  IBS={ibs:.4f}")

    # CoxNet
    print("[2/4] CoxNet")
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=5000, fit_baseline_model=True)
    coxnet.fit(Xtr, y_tr_s)
    val_c = [concordance_index_ipcw(y_tr_s, y_va_s, coxnet.predict(Xva, alpha=a))[0] for a in coxnet.alphas_]
    best_alpha = coxnet.alphas_[int(np.argmax(val_c))]
    risk_te = coxnet.predict(Xte, alpha=best_alpha)
    surv_fns = coxnet.predict_survival_function(Xte, alpha=best_alpha)
    surv_mat = np.asarray([fn(eval_times) for fn in surv_fns])
    c = concordance_index_ipcw(y_tr_s, y_te_s, risk_te)[0]
    _, td_mean = cumulative_dynamic_auc(y_tr_s, y_te_s, risk_te, eval_times)
    ibs = integrated_brier_score(y_tr_s, y_te_s, surv_mat, eval_times)
    metrics["CoxNet"] = {"c_index_ipcw": float(c), "mean_td_auc": float(td_mean), "ibs": float(ibs)}
    print(f"  CoxNet C={c:.4f}  TD-AUC={td_mean:.4f}  IBS={ibs:.4f}  alpha={best_alpha:.4f}")

    # RSF
    print("[3/4] RSF")
    rsf = RandomSurvivalForest(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rsf.fit(Xtr, y_tr_s)
    risk_te = rsf.predict(Xte)
    surv_fns = rsf.predict_survival_function(Xte)
    surv_mat = np.asarray([fn(eval_times) for fn in surv_fns])
    c = concordance_index_ipcw(y_tr_s, y_te_s, risk_te)[0]
    _, td_mean = cumulative_dynamic_auc(y_tr_s, y_te_s, risk_te, eval_times)
    ibs = integrated_brier_score(y_tr_s, y_te_s, surv_mat, eval_times)
    metrics["RSF"] = {"c_index_ipcw": float(c), "mean_td_auc": float(td_mean), "ibs": float(ibs)}
    print(f"  RSF    C={c:.4f}  TD-AUC={td_mean:.4f}  IBS={ibs:.4f}")

    # GBSA
    print("[4/4] GBSA")
    best_gbs, best_val = None, -np.inf
    for n_est in [100, 200, 300]:
        gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_est, max_depth=3, learning_rate=0.05,
                                                random_state=RANDOM_STATE)
        gbs.fit(Xtr, y_tr_s)
        val = concordance_index_ipcw(y_tr_s, y_va_s, gbs.predict(Xva))[0]
        if val > best_val:
            best_gbs, best_val = gbs, val
    risk_te = best_gbs.predict(Xte)
    surv_fns = best_gbs.predict_survival_function(Xte)
    surv_mat = np.asarray([fn(eval_times) for fn in surv_fns])
    c = concordance_index_ipcw(y_tr_s, y_te_s, risk_te)[0]
    _, td_mean = cumulative_dynamic_auc(y_tr_s, y_te_s, risk_te, eval_times)
    ibs = integrated_brier_score(y_tr_s, y_te_s, surv_mat, eval_times)
    metrics["GBSA"] = {"c_index_ipcw": float(c), "mean_td_auc": float(td_mean), "ibs": float(ibs),
                       "n_estimators": int(best_gbs.n_estimators)}
    print(f"  GBSA   C={c:.4f}  TD-AUC={td_mean:.4f}  IBS={ibs:.4f}  n_est={best_gbs.n_estimators}")

    # --- Kaplan-Meier plot on training cohort ---
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    sub = data.loc[idx_tr]
    kmf.fit(sub["event_time_days"], sub["event_observed"], label="Train (all)")
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel(f"Days from landmark = {LANDMARK_DATE.date()}")
    ax.set_ylabel("S(t)")
    ax.set_title(f"Online Retail II — KM, {CHURN_WINDOW}d inactivity churn")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ART / "online_retail_km.png"); plt.close(fig)

    # --- Save metrics + predictions ---
    metrics["_meta"] = {
        "landmark": str(LANDMARK_DATE.date()),
        "obs_end": str(obs_end.date()),
        "churn_window_days": CHURN_WINDOW,
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "n_test": int(len(idx_te)),
        "n_features": len(feat_cols),
        "event_rate_test": float(y_te["event_observed"].mean()),
        "eval_times": eval_times.tolist(),
        "leakage_corr_smoke_check_pass": len(leaks) == 0,
    }
    with open(ART / "online_retail_survival_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[output] {ART / 'online_retail_survival_metrics.json'}")
    summary_line = ", ".join(f"{k}: {v['c_index_ipcw']:.4f}" for k, v in metrics.items() if k != "_meta")
    print(f"[summary] {summary_line}")


if __name__ == "__main__":
    main()
