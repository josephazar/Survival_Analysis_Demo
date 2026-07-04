"""Leakage and smoke tests for the Dunnhumby pipeline.

Run after stages 00..08 have completed. Asserts:

1. Landmark feature matrix contains no post-landmark data — including a
   recomputation check: key features are rebuilt from the raw pre-t0
   transactions and must match the stored matrix exactly.
2. No feature has |corr| > 0.7 with event_observed (no direct leakage).
3. Survival probabilities S(Δ) are monotone non-increasing in Δ.
   (Monotone-by-construction for Cox-family models; kept as a tripwire
   for interpolation/serialization bugs.)
4. Scorecard has no unexpected nulls.
5. CLV benchmark test set is disjoint from train.
6. Event time definitions are internally consistent.

Usage (from dunnhumby/):
    ../venv/bin/python tests/test_leakage_and_smoke.py
or via pytest (collected as a single test wrapping all checks):
    pytest tests/test_leakage_and_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from config import (CHURN_WINDOW, LANDMARK_DAY, MAX_DAY, PROCESSED_DIR, WORKSPACE_DIR)


def assert_true(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


def check_landmark_features_are_pre_t0():
    print("\n1. Landmark features reference no post-t0 data")
    features = pd.read_parquet(PROCESSED_DIR / "household_features_landmark.parquet")
    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    passed = 0
    total = 0

    # first_day + activity_span_pre should be <= LANDMARK_DAY
    total += 1
    max_last_pre = features["last_pre_day"].max() if "last_pre_day" in features.columns else None
    passed += assert_true(
        max_last_pre is None or max_last_pre <= LANDMARK_DAY,
        f"max last_pre_day ({max_last_pre}) <= landmark ({LANDMARK_DAY})"
    )

    # Sanity: the labelled last_pre_t0 is always <= t0
    total += 1
    max_last_pre_t0 = features["last_pre_t0"].max()
    passed += assert_true(
        max_last_pre_t0 <= LANDMARK_DAY,
        f"max last_pre_t0 ({max_last_pre_t0}) <= landmark ({LANDMARK_DAY})"
    )

    # Alive-at-landmark filter: last_pre_t0 must be within CHURN_WINDOW of landmark.
    # Boundary is INCLUSIVE (last_pre_t0 == t0 - window is still alive).
    total += 1
    min_last_pre = features["last_pre_t0"].min()
    passed += assert_true(
        min_last_pre >= LANDMARK_DAY - CHURN_WINDOW,
        f"min last_pre_t0 ({min_last_pre}) >= landmark - window ({LANDMARK_DAY - CHURN_WINDOW})"
    )

    # Recomputation check: rebuild two representative features from the RAW
    # pre-t0 transactions and require exact agreement. If any feature block
    # accidentally used post-landmark rows, these would drift.
    idx = features.set_index("household_key")
    b_pre = baskets[baskets["DAY"] <= LANDMARK_DAY]

    total += 1
    recomputed_n = b_pre.groupby("household_key")["BASKET_ID"].nunique().reindex(idx.index)
    passed += assert_true(
        (idx["n_baskets_pre"] == recomputed_n).all(),
        "n_baskets_pre matches recomputation from raw DAY <= t0 baskets"
    )

    total += 1
    recomputed_s30 = (b_pre[b_pre["DAY"] > LANDMARK_DAY - 30]
                      .groupby("household_key")["SALES_VALUE"].sum()
                      .reindex(idx.index).fillna(0.0))
    passed += assert_true(
        np.allclose(idx["spend_last_30d_pre_t0"].values, recomputed_s30.values, atol=1e-6),
        "spend_last_30d_pre_t0 matches recomputation from raw DAY <= t0 baskets"
    )
    return passed, total


def check_no_leakage_correlations():
    print("\n2. No feature |corr| > 0.7 with event_observed or event_time")
    features = pd.read_parquet(PROCESSED_DIR / "household_features_landmark.parquet")
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols
                if c not in ("household_key", "event_time", "event_observed",
                              "last_pre_t0", "last_overall")]
    passed, total = 0, 0
    leaks_event = []
    leaks_time = []
    for c in num_cols:
        ce = abs(features[c].corr(features["event_observed"]))
        ct = abs(features[c].corr(features["event_time"]))
        if ce > 0.7:
            leaks_event.append((c, ce))
        if ct > 0.7:
            leaks_time.append((c, ct))
    total += 1
    passed += assert_true(len(leaks_event) == 0,
                           f"no features with |corr(event)| > 0.7; found {leaks_event}")
    total += 1
    passed += assert_true(len(leaks_time) == 0,
                           f"no features with |corr(event_time)| > 0.7; found {leaks_time}")
    return passed, total


def check_survival_probs_monotone():
    print("\n3. Scorecard S(30d) >= S(60d) >= S(90d) >= S(180d)")
    scorecard = pd.read_csv(WORKSPACE_DIR / "data" / "household_survival_scorecard.csv")
    cohort = scorecard.dropna(subset=["S_30d", "S_60d", "S_90d", "S_180d"])
    passed, total = 0, 0
    for (a, b) in [("S_30d", "S_60d"), ("S_60d", "S_90d"), ("S_90d", "S_180d")]:
        total += 1
        n_violating = int((cohort[a] < cohort[b] - 1e-6).sum())
        passed += assert_true(
            n_violating == 0,
            f"{a} >= {b}: violations={n_violating}/{len(cohort)}"
        )
    return passed, total


def check_scorecard_shape_and_nulls():
    print("\n4. Scorecard shape & acceptable null columns")
    scorecard = pd.read_csv(WORKSPACE_DIR / "data" / "household_survival_scorecard.csv")
    passed, total = 0, 0
    total += 1
    passed += assert_true(len(scorecard) == 2500,
                           f"scorecard has 2500 rows (got {len(scorecard)})")
    total += 1
    passed += assert_true(scorecard["household_key"].is_unique,
                           "household_key is unique across scorecard")
    # CLV bench columns are expected null for non-test households; p_alive is
    # intentionally NaN for frequency-0 households (BG/NBD returns exactly 1.0
    # there by construction, so stage 08 masks the artifact).
    allowed_null_cols = {"AGE_DESC", "INCOME_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
                          "KID_CATEGORY_DESC", "MARITAL_STATUS_CODE", "HOMEOWNER_DESC",
                          "segment", "churned_post_landmark", "survival_time_from_t0",
                          "risk_score", "risk_percentile", "split",
                          "S_30d", "S_60d", "S_90d", "S_180d",
                          "p_alive",
                          "clv_ml_xgb_bench", "clv_btyd_bench"}
    unexpected_nulls = [c for c in scorecard.columns
                        if scorecard[c].isna().any() and c not in allowed_null_cols]
    total += 1
    passed += assert_true(len(unexpected_nulls) == 0,
                           f"only expected columns have nulls; unexpected={unexpected_nulls}")
    return passed, total


def check_clv_split_disjoint():
    print("\n5. CLV benchmark train / val / test are pairwise disjoint")
    assignment = pd.read_parquet(PROCESSED_DIR / "clv_split_assignment.parquet")
    clv = pd.read_parquet(PROCESSED_DIR / "clv_benchmark.parquet")
    passed, total = 0, 0

    train_hh = set(assignment.loc[assignment["split"] == "train", "household_key"])
    val_hh = set(assignment.loc[assignment["split"] == "val", "household_key"])
    test_hh = set(assignment.loc[assignment["split"] == "test", "household_key"])

    total += 1
    passed += assert_true(
        len(train_hh & val_hh) == 0,
        f"train ∩ val = ∅ (overlap = {len(train_hh & val_hh)})"
    )
    total += 1
    passed += assert_true(
        len(train_hh & test_hh) == 0,
        f"train ∩ test = ∅ (overlap = {len(train_hh & test_hh)})"
    )
    total += 1
    passed += assert_true(
        len(val_hh & test_hh) == 0,
        f"val ∩ test = ∅ (overlap = {len(val_hh & test_hh)})"
    )

    # The clv_benchmark.parquet should only contain test households
    total += 1
    benchmarked = set(clv["household_key"])
    passed += assert_true(
        benchmarked == test_hh,
        f"clv_benchmark.parquet households == test fold (|bench|={len(benchmarked)}, |test|={len(test_hh)}, "
        f"diff={len(benchmarked ^ test_hh)})"
    )

    # Sanity: all three splits cover the full eligible population
    total += 1
    total_coverage = len(train_hh) + len(val_hh) + len(test_hh)
    passed += assert_true(
        total_coverage == len(train_hh | val_hh | test_hh),
        f"no household appears in more than one split (unique={len(train_hh | val_hh | test_hh)}, sum={total_coverage})"
    )

    return passed, total


def check_event_time_consistency():
    print("\n6. Survival event_time definition is self-consistent")
    features = pd.read_parquet(PROCESSED_DIR / "household_features_landmark.parquet")
    passed, total = 0, 0

    # For event=1 households: event_time == last_overall + window - t0,
    # clipped to >= 1 (the boundary household whose churn is declared exactly
    # at t0 — see compute_survival_target in 03_feature_engineering.py).
    ev = features[features["event_observed"] == 1]
    expected = (ev["last_overall"] + CHURN_WINDOW - LANDMARK_DAY).clip(lower=1)
    total += 1
    passed += assert_true(
        (ev["event_time"] == expected).all(),
        f"event=1 HHs have event_time = max(last_overall + {CHURN_WINDOW} - {LANDMARK_DAY}, 1)"
    )

    # For event=0 households: event_time == MAX_DAY - t0 (the study-end censoring time)
    ce = features[features["event_observed"] == 0]
    total += 1
    expected_c = MAX_DAY - LANDMARK_DAY
    passed += assert_true(
        (ce["event_time"] == expected_c).all(),
        f"event=0 HHs have event_time = {expected_c} (= MAX_DAY - t0)"
    )
    return passed, total


CHECKS = [check_landmark_features_are_pre_t0,
          check_no_leakage_correlations,
          check_survival_probs_monotone,
          check_scorecard_shape_and_nulls,
          check_clv_split_disjoint,
          check_event_time_consistency]


def run_all():
    results = [fn() for fn in CHECKS]
    total_passed = sum(p for p, _ in results)
    total_total = sum(t for _, t in results)
    return total_passed, total_total


def test_pipeline_invariants():
    """Pytest entry point: a genuine assertion wrapping every check, so a
    failure fails the test run instead of only printing FAIL."""
    total_passed, total_total = run_all()
    assert total_passed == total_total, (
        f"{total_total - total_passed} pipeline invariant(s) failed — "
        "see [FAIL] lines above")


def main():
    print("=" * 64)
    print("DUNNHUMBY LEAKAGE & SMOKE TESTS")
    print("=" * 64)
    total_passed, total_total = run_all()
    print(f"\n{'=' * 64}")
    print(f"  OVERALL: {total_passed}/{total_total} assertions passed")
    print("=" * 64)
    sys.exit(0 if total_passed == total_total else 1)


if __name__ == "__main__":
    main()
