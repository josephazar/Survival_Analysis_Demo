"""Stage 04 — BTYD benchmark (Lifetimes BG/NBD + Gamma-Gamma).

Fits the probabilistic buy-till-you-die model on the synthesized
datetime transaction log and produces per-household p_alive,
expected transactions (6-month horizon), and CLV estimates.

These outputs feed the CLV benchmark stage and the final scorecard,
but are *excluded* from survival model features to prevent leakage
(the same design decision as in the Online Retail pipeline).

Outputs:
    processed/btyd_predictions.parquet
    artifacts/btyd/bgnbd_diagnostic.png
    artifacts/btyd/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data

from config import PROCESSED_DIR, ARTIFACTS_DIR, MAX_DAY, SYNTHETIC_EPOCH

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = ARTIFACTS_DIR / "btyd"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110


def build_rfm(baskets: pd.DataFrame) -> pd.DataFrame:
    """Use lifetimes' helper with synthesized InvoiceDate."""
    obs_end = pd.to_datetime(SYNTHETIC_EPOCH) + pd.Timedelta(days=MAX_DAY - 1)
    rfm = summary_data_from_transaction_data(
        baskets,
        customer_id_col="household_key",
        datetime_col="INVOICE_DATE",
        monetary_value_col="SALES_VALUE",
        observation_period_end=obs_end,
        freq="D",
    ).reset_index()
    return rfm


def build_calibration_holdout(baskets: pd.DataFrame, calibration_end_day: int = 450,
                              holdout_end_day: int = 631) -> pd.DataFrame:
    cal_end = pd.to_datetime(SYNTHETIC_EPOCH) + pd.Timedelta(days=calibration_end_day - 1)
    hold_end = pd.to_datetime(SYNTHETIC_EPOCH) + pd.Timedelta(days=holdout_end_day - 1)
    calhold = calibration_and_holdout_data(
        baskets,
        customer_id_col="household_key",
        datetime_col="INVOICE_DATE",
        calibration_period_end=cal_end,
        observation_period_end=hold_end,
        freq="D",
        monetary_value_col="SALES_VALUE",
    ).reset_index()
    return calhold


def fit_bgnbd(rfm: pd.DataFrame, penalizer: float = 0.001) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf


def fit_gamma_gamma(rfm: pd.DataFrame, penalizer: float = 0.001) -> GammaGammaFitter:
    # Only repeat customers with positive monetary values
    repeat = rfm[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf


def main():
    print("=" * 60)
    print("STAGE 04: BTYD BENCHMARK (Lifetimes)")
    print("=" * 60)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    baskets["INVOICE_DATE"] = pd.to_datetime(baskets["INVOICE_DATE"])

    rfm = build_rfm(baskets)
    print(f"[rfm] shape={rfm.shape}  repeat={(rfm['frequency'] > 0).sum()}  monetary>0={(rfm['monetary_value'] > 0).sum()}")

    # --- Fit on full RFM (for downstream features) --------------------------
    bgf = fit_bgnbd(rfm)
    ggf = fit_gamma_gamma(rfm)

    # Gamma-Gamma precondition: frequency and monetary value must be
    # (approximately) independent — check it rather than assume it.
    repeat = rfm[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)]
    freq_monetary_corr = float(repeat[["frequency", "monetary_value"]].corr().iloc[0, 1])
    print(f"[gamma-gamma check] corr(frequency, monetary) = {freq_monetary_corr:.3f} "
          f"({'OK' if abs(freq_monetary_corr) < 0.1 else 'WARNING: assumption strained'})")

    # BG/NBD sanity: on a dense loyal-shopper grocery panel the dropout
    # parameter `a` can collapse toward 0, making p_alive ~1 for nearly
    # everyone — a known BG/NBD pathology, not a data error. Flag it loudly
    # so downstream consumers don't treat p_alive as an informative churn
    # signal here (the scorecard's `already_churned` override does the work).
    a_param = float(bgf.params_["a"])
    degenerate_dropout = a_param < 0.05
    if degenerate_dropout:
        print(f"[bg/nbd diagnostic] dropout parameter a = {a_param:.4f} ~ 0: "
              "the model believes households essentially never die. p_alive is "
              "near 1 for almost everyone and should NOT be used as a churn "
              "signal on this panel (an MBG/NBD variant would be more appropriate).")

    rfm["p_alive"] = bgf.conditional_probability_alive(rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["expected_txns_180d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        180, rfm["frequency"], rfm["recency"], rfm["T"]
    )

    # CLV (180-day horizon) - only for repeat customers with positive monetary
    can_score = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
    rfm["clv_180d"] = np.nan
    if can_score.sum() > 0:
        clv = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[can_score, "frequency"],
            rfm.loc[can_score, "recency"],
            rfm.loc[can_score, "T"],
            rfm.loc[can_score, "monetary_value"],
            time=6,      # 6 months
            freq="D",
            discount_rate=0.01,
        )
        rfm.loc[can_score, "clv_180d"] = clv.values
    rfm["clv_180d"] = rfm["clv_180d"].fillna(0)

    # Expected average profit per transaction
    rfm["predicted_avg_spend"] = np.nan
    if can_score.sum() > 0:
        rfm.loc[can_score, "predicted_avg_spend"] = ggf.conditional_expected_average_profit(
            rfm.loc[can_score, "frequency"], rfm.loc[can_score, "monetary_value"]
        ).values
    rfm["predicted_avg_spend"] = rfm["predicted_avg_spend"].fillna(rfm["monetary_value"])

    # --- Calibration / holdout validation -----------------------------------
    calhold = build_calibration_holdout(baskets)
    print(f"[calhold] shape={calhold.shape}")
    bgf_cal = BetaGeoFitter(penalizer_coef=0.001)
    bgf_cal.fit(calhold["frequency_cal"], calhold["recency_cal"], calhold["T_cal"])

    holdout_duration_days = (calhold["duration_holdout"].max())
    calhold["predicted_holdout_txns"] = bgf_cal.conditional_expected_number_of_purchases_up_to_time(
        holdout_duration_days, calhold["frequency_cal"], calhold["recency_cal"], calhold["T_cal"]
    )

    mae = float(np.abs(calhold["predicted_holdout_txns"] - calhold["frequency_holdout"]).mean())
    corr = float(calhold[["predicted_holdout_txns", "frequency_holdout"]].corr().iloc[0, 1])
    rmse = float(np.sqrt(((calhold["predicted_holdout_txns"] - calhold["frequency_holdout"]) ** 2).mean()))
    print(f"[holdout] MAE={mae:.3f}  RMSE={rmse:.3f}  Pearson r={corr:.3f}  (holdout {holdout_duration_days}d)")

    # --- Diagnostics ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(rfm["p_alive"], bins=40, color="#3b82f6", edgecolor="white")
    axes[0].set_title(f"P(alive) distribution — median={rfm['p_alive'].median():.2f}")
    axes[0].set_xlabel("P(alive)")

    axes[1].scatter(calhold["frequency_holdout"], calhold["predicted_holdout_txns"],
                    s=10, alpha=0.4, color="#8b5cf6")
    lim = max(calhold["frequency_holdout"].max(), calhold["predicted_holdout_txns"].max())
    axes[1].plot([0, lim], [0, lim], color="red", linestyle="--", lw=1)
    axes[1].set_xlabel("Actual holdout txns")
    axes[1].set_ylabel("Predicted holdout txns")
    axes[1].set_title(f"Holdout — Pearson r = {corr:.3f}")
    fig.tight_layout(); fig.savefig(OUT / "bgnbd_diagnostic.png"); plt.close(fig)

    # --- Persist outputs -----------------------------------------------------
    rfm.to_parquet(PROCESSED_DIR / "btyd_predictions.parquet", index=False)

    summary = {
        "n_households": int(len(rfm)),
        "n_repeat": int((rfm["frequency"] > 0).sum()),
        "bgnbd_params": {k: float(v) for k, v in zip(["r", "alpha", "a", "b"], bgf.params_.values)},
        "gamma_gamma_params": {k: float(v) for k, v in zip(["p", "q", "v"], ggf.params_.values)},
        "p_alive_median": float(rfm["p_alive"].median()),
        "p_alive_mean": float(rfm["p_alive"].mean()),
        "freq_monetary_corr_repeat": freq_monetary_corr,
        "degenerate_dropout_warning": degenerate_dropout,
        "degenerate_dropout_note": ("a ~ 0 on this dense grocery panel: p_alive is ~1 "
                                     "for nearly all households and is not an informative "
                                     "churn signal here" if degenerate_dropout else None),
        "clv_180d_median": float(rfm["clv_180d"].median()),
        "clv_180d_mean": float(rfm["clv_180d"].mean()),
        "holdout_mae": mae,
        "holdout_rmse": rmse,
        "holdout_pearson_r": corr,
        "holdout_window_days": float(holdout_duration_days),
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[summary]")
    print(json.dumps(summary, indent=2))
    print(f"[output] {PROCESSED_DIR / 'btyd_predictions.parquet'}")


if __name__ == "__main__":
    main()
