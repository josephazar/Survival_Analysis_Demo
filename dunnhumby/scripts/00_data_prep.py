"""Stage 00 — Data preparation & relational merging.

Reads raw Dunnhumby CSVs from ../../data/dunnhumby/, cleans transactions,
enriches with product metadata, synthesizes datetime columns from the
integer DAY index, and persists canonical parquet tables to processed/.

Outputs (in processed/):
    transactions.parquet       Cleaned transaction log (basket-level, one row per line item)
    baskets.parquet            Basket-level aggregation (one row per household×BASKET_ID)
    product.parquet            Product catalog
    demographics.parquet       Household demographics (with 'Unknown' imputation)
    campaign_exposure.parquet  Household×campaign with start/end days
    coupon_redempt.parquet     Coupon redemptions (household, day, coupon, campaign)
    observation.parquet        Per-household observation window (first_day, last_day, T)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, PROCESSED_DIR, SYNTHETIC_EPOCH, MAX_DAY


def synth_date(day: pd.Series) -> pd.Series:
    """Map integer DAY (1..711) to a synthetic datetime starting at SYNTHETIC_EPOCH."""
    return pd.to_datetime(SYNTHETIC_EPOCH) + pd.to_timedelta(day - 1, unit="D")


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "transaction_data.csv")
    n0 = len(df)
    # Clean: only positive quantity & sales_value (drop returns / zero rows)
    df = df[(df["QUANTITY"] > 0) & (df["SALES_VALUE"] > 0)].copy()
    df["INVOICE_DATE"] = synth_date(df["DAY"])
    # Total discount received on this line (negative values in source)
    df["TOTAL_DISC"] = df["RETAIL_DISC"].fillna(0) + df["COUPON_DISC"].fillna(0) + df["COUPON_MATCH_DISC"].fillna(0)
    df["HAS_COUPON"] = (df["COUPON_DISC"].fillna(0) != 0).astype(int)
    print(f"[transactions] rows: {n0:,} -> {len(df):,} after cleaning")
    print(f"[transactions] households: {df['household_key'].nunique():,}")
    print(f"[transactions] DAY range: {df['DAY'].min()} .. {df['DAY'].max()}")
    return df


def load_products() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "product.csv")
    # Normalize brand flag: 'National' vs 'Private'
    df["IS_PRIVATE_LABEL"] = (df["BRAND"].str.strip().str.upper() == "PRIVATE").astype(int)
    # Normalize string fields
    for col in ["DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]:
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()
    print(f"[products] rows: {len(df):,}")
    return df


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "hh_demographic.csv")
    print(f"[demographics] rows with data: {len(df):,}")
    return df


def build_demographics(all_households: pd.Series) -> pd.DataFrame:
    """Build a full-coverage demographic table (with 'Unknown' for missing)."""
    demo = load_demographics()
    demo_cols = [c for c in demo.columns if c != "household_key"]
    full = pd.DataFrame({"household_key": all_households})
    full = full.merge(demo, on="household_key", how="left")
    full["HAS_DEMOGRAPHICS"] = full[demo_cols[0]].notna().astype(int)
    for c in demo_cols:
        full[c] = full[c].fillna("Unknown")
    return full


def build_baskets(tx: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line items to basket level."""
    tx_dept = tx.merge(products[["PRODUCT_ID", "DEPARTMENT", "IS_PRIVATE_LABEL"]], on="PRODUCT_ID", how="left")
    agg = (
        tx_dept.groupby(["household_key", "BASKET_ID", "DAY", "STORE_ID"], as_index=False)
        .agg(
            SALES_VALUE=("SALES_VALUE", "sum"),
            QUANTITY=("QUANTITY", "sum"),
            TOTAL_DISC=("TOTAL_DISC", "sum"),
            COUPON_DISC=("COUPON_DISC", "sum"),
            N_LINES=("PRODUCT_ID", "size"),
            N_UNIQUE_PRODUCTS=("PRODUCT_ID", "nunique"),
            N_DEPARTMENTS=("DEPARTMENT", "nunique"),
            PRIVATE_LABEL_LINES=("IS_PRIVATE_LABEL", "sum"),
            TRANS_TIME=("TRANS_TIME", "first"),
            WEEK_NO=("WEEK_NO", "first"),
        )
    )
    agg["HAS_COUPON"] = (agg["COUPON_DISC"].abs() > 0).astype(int)
    agg["DISCOUNT_DEPTH"] = agg["TOTAL_DISC"].abs() / (agg["SALES_VALUE"] + agg["TOTAL_DISC"].abs())
    agg["INVOICE_DATE"] = synth_date(agg["DAY"])
    agg["PRIVATE_LABEL_RATIO"] = agg["PRIVATE_LABEL_LINES"] / agg["N_LINES"]
    print(f"[baskets] rows: {len(agg):,} ({agg['household_key'].nunique():,} households)")
    return agg


def build_observation(baskets: pd.DataFrame) -> pd.DataFrame:
    """Per-household first/last day and observation length T."""
    obs = (
        baskets.groupby("household_key", as_index=False)
        .agg(
            first_day=("DAY", "min"),
            last_day=("DAY", "max"),
            n_baskets=("BASKET_ID", "nunique"),
        )
    )
    obs["T_days"] = MAX_DAY - obs["first_day"] + 1
    obs["days_since_last"] = MAX_DAY - obs["last_day"]
    obs["duration"] = obs["last_day"] - obs["first_day"]
    return obs


def build_campaign_exposure() -> pd.DataFrame:
    ct = pd.read_csv(DATA_DIR / "campaign_table.csv")
    cd = pd.read_csv(DATA_DIR / "campaign_desc.csv")
    merged = ct.merge(cd[["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN", how="left")
    return merged


def build_coupon_redempt() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "coupon_redempt.csv")


def main():
    print("=" * 60)
    print("STAGE 00: DATA PREPARATION")
    print("=" * 60)

    tx = load_transactions()
    products = load_products()
    baskets = build_baskets(tx, products)
    observation = build_observation(baskets)

    all_hh = pd.Series(sorted(baskets["household_key"].unique()), name="household_key")
    demo_full = build_demographics(all_hh)

    campaigns = build_campaign_exposure()
    coupon_redempt = build_coupon_redempt()

    # Write artifacts
    tx.to_parquet(PROCESSED_DIR / "transactions.parquet", index=False)
    products.to_parquet(PROCESSED_DIR / "product.parquet", index=False)
    baskets.to_parquet(PROCESSED_DIR / "baskets.parquet", index=False)
    observation.to_parquet(PROCESSED_DIR / "observation.parquet", index=False)
    demo_full.to_parquet(PROCESSED_DIR / "demographics.parquet", index=False)
    campaigns.to_parquet(PROCESSED_DIR / "campaign_exposure.parquet", index=False)
    coupon_redempt.to_parquet(PROCESSED_DIR / "coupon_redempt.parquet", index=False)

    print("\n[summary]")
    print(f"  households total: {len(all_hh):,}")
    print(f"  households with demographics: {int(demo_full['HAS_DEMOGRAPHICS'].sum()):,} ({demo_full['HAS_DEMOGRAPHICS'].mean() * 100:.1f}%)")
    print(f"  baskets total: {len(baskets):,}")
    print(f"  median baskets / household: {observation['n_baskets'].median():.0f}")
    print(f"  one-timer households (n_baskets==1): {int((observation['n_baskets'] == 1).sum())}")
    print("\n[outputs]")
    for p in sorted(PROCESSED_DIR.glob("*.parquet")):
        print(f"  {p.name:35s} {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
