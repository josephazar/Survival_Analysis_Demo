"""Stage 02 — Empirical churn window selection.

Mirrors `churn-window-analysis.ipynb` from the parent project but adapted
to integer DAY indices. Computes inter-basket gaps, fits a CDF, applies
Kneedle + max-curvature to locate the elbow, and validates a grid of
candidate thresholds by measuring label stability.

Key difference from Online Retail: grocery cadence is much tighter, so
the elbow should land somewhere between 14 and 30 days (vs. 43d in
Online Retail II).

Outputs (in artifacts/churn_window/):
    cdf_gaps.png               CDF with elbow markers
    curvature.png              Second-derivative curvature plot
    label_agreement.csv        Pairwise agreement across candidate windows
    summary.json               Chosen window + rationale
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator

from config import PROCESSED_DIR, ARTIFACTS_DIR, MAX_DAY

OUT = ARTIFACTS_DIR / "churn_window"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110


def compute_gaps(baskets: pd.DataFrame, collapse_same_day: bool = False) -> pd.Series:
    b = baskets.sort_values(["household_key", "DAY"])[["household_key", "BASKET_ID", "DAY"]].drop_duplicates(["household_key", "BASKET_ID"])
    if collapse_same_day:
        # Two trips on the same day are one shopping *day*; collapsing removes
        # the 0-day gaps that pile up on the left of the CDF.
        b = b.drop_duplicates(["household_key", "DAY"])
    b["prev_day"] = b.groupby("household_key")["DAY"].shift(1)
    b["gap"] = b["DAY"] - b["prev_day"]
    return b["gap"].dropna().astype(float)


def cdf(gaps: pd.Series, max_day: int = 120) -> tuple[np.ndarray, np.ndarray]:
    days = np.arange(1, max_day + 1)
    cdf_vals = np.array([(gaps <= d).mean() for d in days])
    return days, cdf_vals


def find_elbow(days: np.ndarray, cdf_vals: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Return (kneedle_elbow, max_curvature_elbow, curvature_series)."""
    kl = KneeLocator(days, cdf_vals, curve="concave", direction="increasing", S=1.0)
    kneedle = int(kl.knee) if kl.knee is not None else -1

    # Max curvature via second-derivative magnitude (normalized)
    d1 = np.gradient(cdf_vals, days)
    d2 = np.gradient(d1, days)
    curvature = np.abs(d2) / (1 + d1**2) ** 1.5
    max_curv_idx = int(np.argmax(curvature))
    max_curv_day = int(days[max_curv_idx])
    return kneedle, max_curv_day, curvature


def label_stability(baskets: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """For each candidate window, compute event_observed per household, then pairwise agreement."""
    obs = baskets.groupby("household_key", as_index=False).agg(last_day=("DAY", "max"), n_baskets=("BASKET_ID", "nunique"))
    obs = obs[obs["n_baskets"] > 1]  # survival analysis uses repeaters only
    obs["days_since_last"] = MAX_DAY - obs["last_day"]
    labels = {f"event_{w}d": (obs["days_since_last"] > w).astype(int) for w in windows}
    label_df = pd.DataFrame(labels)

    agreement = pd.DataFrame(index=windows, columns=windows, dtype=float)
    for w1 in windows:
        for w2 in windows:
            agreement.loc[w1, w2] = (label_df[f"event_{w1}d"] == label_df[f"event_{w2}d"]).mean()

    event_rate = {w: float(label_df[f"event_{w}d"].mean()) for w in windows}
    return agreement, event_rate


def main():
    print("=" * 60)
    print("STAGE 02: CHURN WINDOW ANALYSIS")
    print("=" * 60)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    gaps = compute_gaps(baskets)
    print(f"[gaps] n={len(gaps):,} median={gaps.median():.1f} mean={gaps.mean():.2f} p95={gaps.quantile(0.95):.1f}")

    days, cdf_vals = cdf(gaps, max_day=120)
    kneedle_day, max_curv_day, curvature = find_elbow(days, cdf_vals)
    print(f"[elbow] Kneedle: {kneedle_day}d   Max-curvature: {max_curv_day}d")
    # The two detectors disagree by design: the second derivative of an
    # empirical CDF amplifies noise and peaks on the initial 2-3 day cliff
    # (within-week cadence), which is not a churn boundary. Kneedle's
    # normalized distance-to-chord is the appropriate detector here.

    # Sensitivity: a sizable share of raw gaps is 0 days (two baskets on one
    # day). Collapse to shopping days and re-run the elbow to check that the
    # zeros are not dragging the elbow left.
    pct_zero = float((gaps == 0).mean())
    gaps_sd = compute_gaps(baskets, collapse_same_day=True)
    days_sd, cdf_sd = cdf(gaps_sd, max_day=120)
    kneedle_sd, _, _ = find_elbow(days_sd, cdf_sd)
    print(f"[sensitivity] {pct_zero * 100:.1f}% of raw gaps are 0d; with same-day baskets "
          f"collapsed, Kneedle = {kneedle_sd}d (raw: {kneedle_day}d)")

    # --- CDF plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, cdf_vals, color="#3b82f6", lw=2, label="Empirical CDF")
    ax.axvline(kneedle_day, color="red", linestyle="--", lw=1.5, label=f"Kneedle elbow = {kneedle_day}d")
    ax.axvline(max_curv_day, color="orange", linestyle="--", lw=1.5, label=f"Max curvature = {max_curv_day}d")
    ax.set_xlabel("Days between consecutive baskets")
    ax.set_ylabel("Cumulative probability P(gap ≤ d)")
    ax.set_title(f"CDF of {len(gaps):,} inter-basket gaps (Dunnhumby)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "cdf_gaps.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days, curvature, color="#ef4444", lw=2)
    ax.axvline(max_curv_day, color="orange", linestyle="--", label=f"Peak at {max_curv_day}d")
    ax.set_xlabel("Days")
    ax.set_ylabel("|κ(d)|")
    ax.set_title("Curvature of CDF")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "curvature.png"); plt.close(fig)

    # --- Label stability grid
    candidate_windows = [14, 21, 30, 45, 60]
    agreement, event_rate = label_stability(baskets, candidate_windows)
    agreement.to_csv(OUT / "label_agreement.csv")
    print(f"\n[event rates by window]")
    for w, rate in event_rate.items():
        print(f"  {w:3d}d -> event rate {rate * 100:.2f}% ({'churned' if rate > 0.5 else 'active majority'})")
    print(f"\n[agreement grid]")
    print(agreement.round(3))

    # Chosen window: the candidate closest to the Kneedle elbow.
    chosen = min(candidate_windows, key=lambda w: abs(w - kneedle_day))
    print(f"\n[chosen window] {chosen}d (closest candidate to Kneedle={kneedle_day}d)")

    summary = {
        "n_gaps": int(len(gaps)),
        "gap_stats": {
            "median": float(gaps.median()),
            "mean": float(gaps.mean()),
            "p90": float(gaps.quantile(0.9)),
            "p95": float(gaps.quantile(0.95)),
            "p99": float(gaps.quantile(0.99)),
        },
        "elbow_kneedle_day": kneedle_day,
        "elbow_max_curvature_day": max_curv_day,
        "elbow_method_note": ("Kneedle preferred: the empirical-CDF second derivative "
                               "amplifies noise and peaks on the 2-3d within-week cliff, "
                               "which is cadence, not a churn boundary."),
        "pct_zero_gaps_raw": pct_zero,
        "elbow_kneedle_day_same_day_collapsed": kneedle_sd,
        "candidate_windows": candidate_windows,
        "event_rate_by_window": event_rate,
        "chosen_window_days": chosen,
        "cdf_at_chosen_window": float(np.interp(chosen, days, cdf_vals)),
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[outputs] {OUT}")


if __name__ == "__main__":
    main()
