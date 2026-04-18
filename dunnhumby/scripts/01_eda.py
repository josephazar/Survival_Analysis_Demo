"""Stage 01 — Exploratory data analysis.

Produces quick diagnostic plots and a summary JSON to verify data shape,
shopping frequency, seasonality, and department mix before modeling.

Outputs (in artifacts/eda/):
    summary.json
    basket_count_hist.png
    gap_distribution.png
    weekly_volume.png
    department_mix.png
    demographic_coverage.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ARTIFACTS_DIR, MAX_DAY

OUT = ARTIFACTS_DIR / "eda"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams["figure.dpi"] = 110


def main():
    print("=" * 60)
    print("STAGE 01: EDA")
    print("=" * 60)

    baskets = pd.read_parquet(PROCESSED_DIR / "baskets.parquet")
    obs = pd.read_parquet(PROCESSED_DIR / "observation.parquet")
    demo = pd.read_parquet(PROCESSED_DIR / "demographics.parquet")
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet", columns=["household_key", "DAY", "SALES_VALUE", "PRODUCT_ID"])
    products = pd.read_parquet(PROCESSED_DIR / "product.parquet")

    summary = {}

    # 1. Basket counts per household
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(obs["n_baskets"], bins=60, color="#3b82f6", edgecolor="white")
    ax.axvline(obs["n_baskets"].median(), color="red", linestyle="--", label=f"median={obs['n_baskets'].median():.0f}")
    ax.set_xlabel("Baskets per household")
    ax.set_ylabel("Households")
    ax.set_title("Distribution of basket counts")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "basket_count_hist.png"); plt.close(fig)

    summary["baskets_per_household"] = {
        "min": int(obs["n_baskets"].min()),
        "p25": float(obs["n_baskets"].quantile(0.25)),
        "median": float(obs["n_baskets"].median()),
        "p75": float(obs["n_baskets"].quantile(0.75)),
        "max": int(obs["n_baskets"].max()),
        "mean": float(obs["n_baskets"].mean()),
    }

    # 2. Inter-basket gap distribution (for churn window analysis)
    bdays = baskets.sort_values(["household_key", "DAY"])[["household_key", "BASKET_ID", "DAY"]].drop_duplicates(["household_key", "BASKET_ID"])
    bdays["prev_day"] = bdays.groupby("household_key")["DAY"].shift(1)
    bdays["gap"] = bdays["DAY"] - bdays["prev_day"]
    gaps = bdays["gap"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps.clip(upper=60), bins=60, color="#10b981", edgecolor="white")
    ax.axvline(gaps.median(), color="red", linestyle="--", label=f"median={gaps.median():.0f}d")
    ax.axvline(gaps.mean(), color="orange", linestyle="--", label=f"mean={gaps.mean():.1f}d")
    ax.set_xlabel("Days between consecutive baskets (clipped at 60)")
    ax.set_ylabel("Count")
    ax.set_title(f"Inter-basket gap distribution (n={len(gaps):,})")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "gap_distribution.png"); plt.close(fig)

    summary["inter_basket_gaps"] = {
        "n": int(len(gaps)),
        "median": float(gaps.median()),
        "mean": float(gaps.mean()),
        "p90": float(gaps.quantile(0.9)),
        "p95": float(gaps.quantile(0.95)),
        "p99": float(gaps.quantile(0.99)),
    }

    # 3. Weekly transaction volume over time
    weekly = tx.groupby(tx["DAY"] // 7 * 7)["SALES_VALUE"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly.index, weekly.values, color="#8b5cf6")
    ax.set_xlabel("Day of observation window")
    ax.set_ylabel("Total weekly sales ($)")
    ax.set_title("Weekly sales volume across the 711-day window")
    fig.tight_layout(); fig.savefig(OUT / "weekly_volume.png"); plt.close(fig)

    # 4. Department mix
    dept_spend = (
        tx.merge(products[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID")
        .groupby("DEPARTMENT")["SALES_VALUE"].sum()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(dept_spend.index[::-1], dept_spend.values[::-1], color="#f59e0b")
    ax.set_xlabel("Total sales ($)")
    ax.set_title("Top 15 departments by total sales")
    fig.tight_layout(); fig.savefig(OUT / "department_mix.png"); plt.close(fig)

    summary["top_departments_share"] = {
        d: float(v / dept_spend.sum()) for d, v in dept_spend.items()
    }

    # 5. Demographic coverage
    demo_cov = {
        "total_households": int(len(demo)),
        "with_demographics": int(demo["HAS_DEMOGRAPHICS"].sum()),
        "pct": float(demo["HAS_DEMOGRAPHICS"].mean()),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, ["AGE_DESC", "INCOME_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC"]):
        vc = demo[col].value_counts()
        ax.bar(range(len(vc)), vc.values, color="#06b6d4")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(x)[:10] for x in vc.index], rotation=45, ha="right", fontsize=8)
        ax.set_title(col)
    fig.suptitle(f"Demographic coverage ({demo_cov['pct'] * 100:.1f}% of households have data)")
    fig.tight_layout(); fig.savefig(OUT / "demographic_coverage.png"); plt.close(fig)

    summary["demographics"] = demo_cov
    summary["observation_window_days"] = MAX_DAY
    summary["total_transactions"] = int(len(tx))
    summary["total_baskets"] = int(len(baskets))
    summary["one_timer_households"] = int((obs["n_baskets"] == 1).sum())

    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary]")
    print(json.dumps(summary, indent=2))
    print(f"\n[artifacts written] {OUT}")


if __name__ == "__main__":
    main()
