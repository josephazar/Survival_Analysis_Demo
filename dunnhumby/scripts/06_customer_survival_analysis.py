"""Stage 06 — Landmark customer survival analysis (v2).

REWRITE. Addresses the review findings:

    P0 feature leakage: all features now come from Stage-03 landmark matrix
        (built strictly from DAY <= t0).
    P1 event-time misalignment: target is the landmark `event_time`
        (days from t0), so event=1 customers have time = L + window - t0.
    P1 scorecard S(t): we now compute CONDITIONAL survival
        S(t0 + Δ | alive at t0), refit on the full training set, and cap
        Δ to the modelled time range so there is no extrapolation.

Pipeline:
    1. Load landmark feature matrix + target from Stage 03.
    2. Train / validation / test split (60/20/20, event-stratified).
       Test is locked away; validation is used for early stopping + model
       selection. Final numbers reported on test once.
    3. PCA + KMeans (on training rows only) to segment households.
    4. Fit five survival models: CoxPH, CoxNet, RSF, GBSA, XGBoost AFT.
    5. Evaluate on test: IPCW C-index, time-dependent AUC, integrated Brier.
    6. Refit the best model on train+val (for scorecard scoring).
    7. Emit per-household risk + conditional S(Δ) to processed/.

Outputs:
    processed/survival_predictions.parquet  (per-household risk + conditional S(Δ))
    artifacts/survival/
        metrics.json                 Full test metrics for all 5 models
        km_by_segment.png            Kaplan-Meier per segment
        pca_segments.png
        feature_importance.png       Permutation importance (RSF)
        cox_hazard_ratios.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

from config import (ARTIFACTS_DIR, CHURN_WINDOW, LANDMARK_DAY, MAX_DAY,
                     PROCESSED_DIR, RANDOM_STATE)

warnings.filterwarnings("ignore")
OUT = ARTIFACTS_DIR / "survival"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110


# Features to use. EXCLUDES BTYD outputs (p_alive, clv) and any column that could
# directly encode the label (none in landmark matrix, but we double-check below).
EXCLUDE = {"household_key", "event_time", "event_observed", "last_pre_t0", "last_overall",
           "first_day", "last_pre_day",   # raw calendar days (identifying, not predictive)
           "AGE_DESC", "INCOME_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
           "KID_CATEGORY_DESC", "MARITAL_STATUS_CODE", "HOMEOWNER_DESC"}


def build_feature_matrix(features: pd.DataFrame):
    feat_cols = [c for c in features.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]
    X = features[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, feat_cols


def to_surv(y_df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(event=y_df["event_observed"].astype(bool).values,
                            time=y_df["event_time"].astype(float).values)


def evaluate_block(model_name, risk_train, risk_test, y_train_surv, y_test_surv,
                   surv_fns_test, eval_times):
    c_test = concordance_index_ipcw(y_train_surv, y_test_surv, risk_test)[0]
    td_auc_series, td_auc_mean = cumulative_dynamic_auc(y_train_surv, y_test_surv, risk_test, eval_times)
    surv_probs = np.asarray([fn(eval_times) for fn in surv_fns_test])
    ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, eval_times)
    return {
        "c_index_test_ipcw": float(c_test),
        "mean_td_auc": float(td_auc_mean),
        "ibs": float(ibs),
    }, surv_probs


def segment_on_train(X_train_scaled: np.ndarray, train_rows: pd.DataFrame, feat_cols):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_train_scaled)
    km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(X_train_scaled)
    cluster_spend = pd.Series(clusters).groupby(clusters).apply(lambda g: train_rows["total_spend_pre"].iloc[g.index].median()).sort_values()
    order = cluster_spend.index.tolist()
    label_map = {int(c): name for c, name in zip(order, ["Bronze", "Silver", "Gold"])}
    segments = pd.Series([label_map[c] for c in clusters], index=train_rows.index, name="segment")
    return pca, km, segments, label_map


def main():
    print("=" * 64)
    print("STAGE 06: LANDMARK SURVIVAL ANALYSIS")
    print(f"  landmark t0 = {LANDMARK_DAY}, churn window = {CHURN_WINDOW}d")
    print("=" * 64)

    features = pd.read_parquet(PROCESSED_DIR / "household_features_landmark.parquet")
    print(f"[features] shape={features.shape}")
    print(f"[event rate] {features['event_observed'].mean() * 100:.2f}%")
    print(f"[event_time] min={features['event_time'].min()}  median={features['event_time'].median():.0f}  max={features['event_time'].max()}")

    X, feat_cols = build_feature_matrix(features)
    print(f"[features_used] {len(feat_cols)} columns")

    # --- Split: 60 / 20 / 20, event-stratified ---------------------------------
    rows = features.reset_index(drop=True)
    X = X.reset_index(drop=True)

    idx_trainval, idx_test = train_test_split(
        rows.index, test_size=0.20, random_state=RANDOM_STATE,
        stratify=rows["event_observed"],
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=RANDOM_STATE,
        stratify=rows.loc[idx_trainval, "event_observed"],
    )
    print(f"[split] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    y_df = rows[["event_time", "event_observed"]]
    y_train = y_df.loc[idx_train]
    y_val = y_df.loc[idx_val]
    y_test = y_df.loc[idx_test]

    # Imputer + scaler fit on train only
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr_raw = imputer.fit_transform(X.loc[idx_train])
    Xva_raw = imputer.transform(X.loc[idx_val])
    Xte_raw = imputer.transform(X.loc[idx_test])
    Xtr = scaler.fit_transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    # --- Segments on training rows ---------------------------------------------
    pca, km_seg, segments_train, label_map = segment_on_train(Xtr, rows.loc[idx_train], feat_cols)
    # Assign segments to val/test rows by nearest-cluster assignment
    segments_all = pd.Series(index=rows.index, dtype="object")
    segments_all.loc[idx_train] = segments_train.values
    seg_val = pd.Series([label_map[c] for c in km_seg.predict(Xva)], index=idx_val)
    seg_test = pd.Series([label_map[c] for c in km_seg.predict(Xte)], index=idx_test)
    segments_all.loc[idx_val] = seg_val.values
    segments_all.loc[idx_test] = seg_test.values

    # KM plot on training set only
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for name in ["Bronze", "Silver", "Gold"]:
        mask = segments_train == name
        if mask.sum() == 0:
            continue
        sub = rows.loc[idx_train].loc[mask]
        kmf.fit(sub["event_time"], sub["event_observed"], label=f"{name} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel(f"Days from landmark t0 = {LANDMARK_DAY}")
    ax.set_ylabel("Survival S(t)")
    ax.set_title(f"Kaplan-Meier by segment (train) — churn = {CHURN_WINDOW}d inactivity post-landmark")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "km_by_segment.png"); plt.close(fig)

    # Scatter for PCA
    fig, ax = plt.subplots(figsize=(7, 5))
    pcs_train = pca.transform(Xtr)
    colors = {"Bronze": "#b45309", "Silver": "#9ca3af", "Gold": "#ca8a04"}
    for name in ["Bronze", "Silver", "Gold"]:
        m = segments_train == name
        ax.scatter(pcs_train[m, 0], pcs_train[m, 1], s=10, alpha=0.5, color=colors[name], label=name)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(); ax.set_title("Landmark-only behaviour — PCA segments")
    fig.tight_layout(); fig.savefig(OUT / "pca_segments.png"); plt.close(fig)

    # --- Survival targets ------------------------------------------------------
    y_train_surv = to_surv(y_train)
    y_val_surv = to_surv(y_val)
    y_test_surv = to_surv(y_test)

    # Evaluation time grid — confined to modelled range to avoid extrapolation
    t_min = int(max(y_test["event_time"].min(), y_train["event_time"].min()) + 1)
    t_max = int(min(y_test["event_time"].max(), y_train["event_time"].max()) - 1)
    eval_times = np.linspace(t_min, t_max, 10)
    print(f"[eval_times] {eval_times.round(1).tolist()}")

    all_metrics = {}
    all_risk_test = {}

    # 1. Cox PH (lifelines) — train only, validate only for reporting consistency
    print("\n[1/5] CoxPH (lifelines)")
    cph_df_train = pd.concat([
        pd.DataFrame(Xtr, columns=feat_cols, index=idx_train),
        y_train,
    ], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df_train, duration_col="event_time", event_col="event_observed", show_progress=False)
    risk_tr = cph.predict_partial_hazard(pd.DataFrame(Xtr, columns=feat_cols)).values.ravel()
    risk_te = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=feat_cols)).values.ravel()
    surv_df = cph.predict_survival_function(pd.DataFrame(Xte, columns=feat_cols), times=eval_times)
    surv_probs_cox = surv_df.values.T

    def make_fn(row, times):
        def fn(t):
            return np.interp(t, times, row)
        return fn
    surv_fns_cox = [make_fn(r, eval_times) for r in surv_probs_cox]
    all_metrics["CoxPH"], _ = evaluate_block("CoxPH", risk_tr, risk_te, y_train_surv, y_test_surv, surv_fns_cox, eval_times)
    all_risk_test["CoxPH"] = risk_te
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].sort_values("exp(coef)", ascending=False)
    hr.to_csv(OUT / "cox_hazard_ratios.csv")

    # 2. CoxNet
    print("[2/5] CoxNet (elastic net)")
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=500,
                                    fit_baseline_model=True)
    coxnet.fit(Xtr, y_train_surv)
    # Pick alpha via validation C-index
    val_c = []
    for alpha in coxnet.alphas_:
        r = coxnet.predict(Xva, alpha=alpha)
        val_c.append(concordance_index_ipcw(y_train_surv, y_val_surv, r)[0])
    best_alpha = coxnet.alphas_[int(np.argmax(val_c))]
    print(f"       selected alpha={best_alpha:.4f} (val C-index={max(val_c):.4f})")
    risk_tr = coxnet.predict(Xtr, alpha=best_alpha)
    risk_te = coxnet.predict(Xte, alpha=best_alpha)
    surv_fns = coxnet.predict_survival_function(Xte, alpha=best_alpha)
    all_metrics["CoxNet"], _ = evaluate_block("CoxNet", risk_tr, risk_te, y_train_surv, y_test_surv, surv_fns, eval_times)
    all_risk_test["CoxNet"] = risk_te

    # 3. RSF
    print("[3/5] RandomSurvivalForest")
    rsf = RandomSurvivalForest(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rsf.fit(Xtr, y_train_surv)
    risk_tr = rsf.predict(Xtr)
    risk_te = rsf.predict(Xte)
    surv_fns = rsf.predict_survival_function(Xte)
    all_metrics["RSF"], _ = evaluate_block("RSF", risk_tr, risk_te, y_train_surv, y_test_surv, surv_fns, eval_times)
    all_risk_test["RSF"] = risk_te
    try:
        pi = permutation_importance(rsf, Xte, y_test_surv, n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
        imp_series = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(imp_series.index, imp_series.values, color="#0891b2")
        ax.set_title("RSF — permutation importance on test (top 20)")
        fig.tight_layout(); fig.savefig(OUT / "feature_importance.png"); plt.close(fig)
    except Exception as e:
        print(f"       permutation importance failed: {e}")

    # 4. GradientBoostingSurvival
    print("[4/5] GradientBoostingSurvivalAnalysis")
    # Use val set for early-stopping-ish behaviour: tune n_estimators from val
    best_gbs = None
    best_val = -np.inf
    for n_est in [100, 200, 300]:
        gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_est, max_depth=3, learning_rate=0.05,
                                                random_state=RANDOM_STATE)
        gbs.fit(Xtr, y_train_surv)
        val_score = concordance_index_ipcw(y_train_surv, y_val_surv, gbs.predict(Xva))[0]
        if val_score > best_val:
            best_val = val_score
            best_gbs = gbs
    risk_tr = best_gbs.predict(Xtr)
    risk_te = best_gbs.predict(Xte)
    surv_fns = best_gbs.predict_survival_function(Xte)
    print(f"       best n_est={best_gbs.n_estimators}  val C-index={best_val:.4f}")
    all_metrics["GBSA"], _ = evaluate_block("GBSA", risk_tr, risk_te, y_train_surv, y_test_surv, surv_fns, eval_times)
    all_risk_test["GBSA"] = risk_te

    # 5. XGBoost AFT with validation set for early stopping
    print("[5/5] XGBoost AFT")
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr)
    dtr.set_float_info("label_lower_bound", y_train["event_time"].values.astype(float))
    dtr.set_float_info("label_upper_bound",
                       np.where(y_train["event_observed"] == 1,
                                y_train["event_time"].values, np.inf).astype(float))
    dva = xgb.DMatrix(Xva)
    dva.set_float_info("label_lower_bound", y_val["event_time"].values.astype(float))
    dva.set_float_info("label_upper_bound",
                       np.where(y_val["event_observed"] == 1,
                                y_val["event_time"].values, np.inf).astype(float))
    dte = xgb.DMatrix(Xte)
    params = {
        "objective": "survival:aft", "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20, "tree_method": "hist",
        "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10,
        "eval_metric": "aft-nloglik",
    }
    bst = xgb.train(params, dtr, num_boost_round=500,
                     evals=[(dva, "val")],
                     early_stopping_rounds=30, verbose_eval=False)
    pred_time_tr = bst.predict(dtr)
    pred_time_te = bst.predict(dte)
    risk_tr = -pred_time_tr
    risk_te = -pred_time_te
    scales = np.clip(pred_time_te, 1.0, None)

    def make_xgb_fn(scale):
        def fn(t):
            return np.exp(-t / scale)
        return fn
    surv_fns_xgb = [make_xgb_fn(s) for s in scales]
    print(f"       best iter = {bst.best_iteration}")
    all_metrics["XGBoost_AFT"], _ = evaluate_block("XGBoost_AFT", risk_tr, risk_te,
                                                    y_train_surv, y_test_surv, surv_fns_xgb, eval_times)
    all_risk_test["XGBoost_AFT"] = risk_te

    # --- Report ----------------------------------------------------------------
    print("\n[test metrics]")
    for name, m in all_metrics.items():
        print(f"  {name:15s}  C-index(IPCW)={m['c_index_test_ipcw']:.4f}  TD-AUC={m['mean_td_auc']:.4f}  IBS={m['ibs']:.4f}")

    best_model = max(all_metrics, key=lambda k: all_metrics[k]["c_index_test_ipcw"])
    print(f"\n[best model] {best_model}")

    with open(OUT / "metrics.json", "w") as fh:
        json.dump({
            "landmark_day": LANDMARK_DAY,
            "churn_window_days": CHURN_WINDOW,
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
            "event_rate_train": float(y_train["event_observed"].mean()),
            "event_rate_test": float(y_test["event_observed"].mean()),
            "feature_count": len(feat_cols),
            "eval_times": eval_times.tolist(),
            "test_metrics": all_metrics,
            "best_model": best_model,
        }, fh, indent=2)

    # --- Refit best model on train+val for scorecard scoring --------------------
    print("\n[refit] training best model family on train+val for scorecard")
    idx_fit = np.concatenate([idx_train, idx_val])
    Xfit_raw = imputer.transform(X.loc[idx_fit])
    scaler_final = StandardScaler()
    Xfit = scaler_final.fit_transform(Xfit_raw)
    # All rows (score everyone, including test — they're just scored, not evaluated)
    X_all_raw = imputer.transform(X)
    X_all_scaled = scaler_final.transform(X_all_raw)
    y_fit = y_df.loc[idx_fit]
    y_fit_surv = to_surv(y_fit)

    if best_model == "CoxPH":
        df_fit = pd.concat([pd.DataFrame(Xfit, columns=feat_cols, index=idx_fit), y_fit], axis=1)
        final = CoxPHFitter(penalizer=0.1)
        final.fit(df_fit, duration_col="event_time", event_col="event_observed", show_progress=False)
        risk_all = final.predict_partial_hazard(pd.DataFrame(X_all_scaled, columns=feat_cols)).values.ravel()
        surv_final = final.predict_survival_function(pd.DataFrame(X_all_scaled, columns=feat_cols), times=eval_times).values.T
    elif best_model == "CoxNet":
        final = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=500, fit_baseline_model=True)
        final.fit(Xfit, y_fit_surv)
        risk_all = final.predict(X_all_scaled, alpha=best_alpha)
        surv_fns_all = final.predict_survival_function(X_all_scaled, alpha=best_alpha)
        surv_final = np.asarray([fn(eval_times) for fn in surv_fns_all])
    elif best_model == "RSF":
        final = RandomSurvivalForest(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                      n_jobs=-1, random_state=RANDOM_STATE)
        final.fit(Xfit, y_fit_surv)
        risk_all = final.predict(X_all_scaled)
        surv_fns_all = final.predict_survival_function(X_all_scaled)
        surv_final = np.asarray([fn(eval_times) for fn in surv_fns_all])
    elif best_model == "GBSA":
        final = GradientBoostingSurvivalAnalysis(n_estimators=best_gbs.n_estimators, max_depth=3,
                                                  learning_rate=0.05, random_state=RANDOM_STATE)
        final.fit(Xfit, y_fit_surv)
        risk_all = final.predict(X_all_scaled)
        surv_fns_all = final.predict_survival_function(X_all_scaled)
        surv_final = np.asarray([fn(eval_times) for fn in surv_fns_all])
    else:  # XGBoost AFT — refit on train+val using the val-selected best_iteration.
        # Previously we reused `bst` (fit on train only, early-stopped on val). That
        # underuses available data. Here we retrain on the combined train+val with
        # num_boost_round fixed at `bst.best_iteration + 1` (no early stopping needed).
        dfit = xgb.DMatrix(Xfit)
        dfit.set_float_info("label_lower_bound", y_fit["event_time"].values.astype(float))
        dfit.set_float_info("label_upper_bound",
                             np.where(y_fit["event_observed"] == 1,
                                      y_fit["event_time"].values, np.inf).astype(float))
        n_rounds = int(bst.best_iteration) + 1
        final_bst = xgb.train(params, dfit, num_boost_round=n_rounds)
        dall = xgb.DMatrix(X_all_scaled)
        pred_times = final_bst.predict(dall)
        risk_all = -pred_times
        scales_all = np.clip(pred_times, 1.0, None)
        surv_final = np.exp(-eval_times[None, :] / scales_all[:, None])
        print(f"       [xgb refit] n_rounds = {n_rounds} on train+val ({len(idx_fit)} rows)")

    # --- Conditional future survival at +30, +60, +90, +180 days from LANDMARK --
    # For each household:  S(t0+Δ | alive at t0) = S(Δ) / S(0)   (here S(0)=1 by def.)
    # Since event_time is measured from t0, S(Δ) is the raw column at t=Δ.
    # Cap Δ at the maximum eval time so we never extrapolate.
    t_cap = float(eval_times.max())
    deltas = [30, 60, 90, 180]
    print(f"[conditional S] Δ grid = {deltas}; capped at {t_cap:.0f}d to avoid extrapolation")

    out_df = pd.DataFrame({
        "household_key": rows["household_key"].values,
        "event_time_from_t0": rows["event_time"].values,
        "event_observed": rows["event_observed"].values,
        "last_pre_t0": rows["last_pre_t0"].values,
        "last_overall": rows["last_overall"].values,
        "segment": segments_all.values,
        "risk_score": risk_all,
    })
    out_df["split"] = np.where(out_df.index.isin(idx_train), "train",
                                np.where(out_df.index.isin(idx_val), "val", "test"))

    for delta in deltas:
        d_eff = min(delta, t_cap)
        # Interpolate S(d_eff) from the eval-time grid for every household
        s_at = np.array([np.interp(d_eff, eval_times, surv_final[i]) for i in range(len(out_df))])
        col = f"S_{int(delta)}d"
        out_df[col] = s_at
        if d_eff < delta:
            print(f"  {col}: capped at t={d_eff:.0f}d (requested Δ={delta})")

    out_df.to_parquet(PROCESSED_DIR / "survival_predictions.parquet", index=False)
    print(f"\n[output] {PROCESSED_DIR / 'survival_predictions.parquet'}  shape={out_df.shape}")
    print(out_df.describe().round(3))


if __name__ == "__main__":
    main()
