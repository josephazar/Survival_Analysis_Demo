"""
Step 1: Data Cleaning & RFM Summary Preparation
Online Retail II Dataset — BTYD Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent.parent / 'data'

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'online_retail_II.csv')
print(f"Raw data shape: {df.shape}")

# ─── Cleaning Pipeline ───────────────────────────────────────────────────────
# 1. Drop rows without Customer ID (can't do customer-level analysis)
df = df.dropna(subset=['Customer ID'])
df['Customer ID'] = df['Customer ID'].astype(int)
print(f"After dropping null Customer ID: {df.shape}")

# 2. Remove cancellations (Invoice starts with 'C')
df = df[~df['Invoice'].astype(str).str.startswith('C')]
print(f"After removing cancellations: {df.shape}")

# 3. Remove zero/negative quantities and prices
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
print(f"After removing non-positive qty/price: {df.shape}")

# 4. Parse dates
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 5. Compute total spend per line item
df['TotalSpend'] = df['Quantity'] * df['Price']

# 6. Remove extreme outliers (top 0.1% by TotalSpend per line)
threshold = df['TotalSpend'].quantile(0.999)
df = df[df['TotalSpend'] <= threshold]
print(f"After outlier removal: {df.shape}")

print(f"\nDate range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Unique customers: {df['Customer ID'].nunique()}")
print(f"Unique invoices: {df['Invoice'].nunique()}")

# ─── Build RFM Summary using lifetimes utility ───────────────────────────────
from lifetimes.utils import summary_data_from_transaction_data

# We use the last date in the dataset as the observation period end
observation_period_end = df['InvoiceDate'].max()
print(f"Observation period end: {observation_period_end}")

# Build the standard RFM summary (frequency, recency, T)
rfm = summary_data_from_transaction_data(
    df,
    customer_id_col='Customer ID',
    datetime_col='InvoiceDate',
    monetary_value_col='TotalSpend',
    observation_period_end=observation_period_end,
    freq='D'  # daily granularity
)

print(f"\nRFM Summary shape: {rfm.shape}")
print(rfm.describe().round(2))

# ─── Key RFM Statistics ──────────────────────────────────────────────────────
print("\n--- RFM Distribution ---")
print(f"Customers with 0 repeat purchases: {(rfm['frequency'] == 0).sum()} ({(rfm['frequency'] == 0).mean()*100:.1f}%)")
print(f"Customers with 1+ repeat purchases: {(rfm['frequency'] > 0).sum()} ({(rfm['frequency'] > 0).mean()*100:.1f}%)")
print(f"Mean frequency (repeat purchases): {rfm['frequency'].mean():.2f}")
print(f"Mean recency (days): {rfm['recency'].mean():.2f}")
print(f"Mean T (customer age, days): {rfm['T'].mean():.2f}")
print(f"Mean monetary value: {rfm['monetary_value'].mean():.2f}")

# ─── Save cleaned data and RFM summary ───────────────────────────────────────
df.to_parquet(BASE_DIR / 'cleaned_transactions.parquet', index=False)
rfm.to_parquet(BASE_DIR / 'rfm_summary.parquet')
print("\nSaved cleaned transactions and RFM summary.")

# ─── Visualizations ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RFM Summary Distributions', fontsize=16, fontweight='bold')

# Frequency distribution
ax = axes[0, 0]
freq_data = rfm['frequency'].clip(upper=50)
ax.hist(freq_data, bins=50, color='#2196F3', edgecolor='white', alpha=0.85)
ax.set_title('Frequency (Repeat Purchases)', fontweight='bold')
ax.set_xlabel('Number of Repeat Purchases')
ax.set_ylabel('Count')
ax.axvline(rfm['frequency'].mean(), color='red', linestyle='--', label=f'Mean={rfm["frequency"].mean():.1f}')
ax.legend()

# Recency distribution
ax = axes[0, 1]
ax.hist(rfm['recency'], bins=50, color='#4CAF50', edgecolor='white', alpha=0.85)
ax.set_title('Recency (Days Since Last Purchase)', fontweight='bold')
ax.set_xlabel('Days')
ax.set_ylabel('Count')
ax.axvline(rfm['recency'].mean(), color='red', linestyle='--', label=f'Mean={rfm["recency"].mean():.1f}')
ax.legend()

# T (customer age) distribution
ax = axes[1, 0]
ax.hist(rfm['T'], bins=50, color='#FF9800', edgecolor='white', alpha=0.85)
ax.set_title('T (Customer Age in Days)', fontweight='bold')
ax.set_xlabel('Days')
ax.set_ylabel('Count')
ax.axvline(rfm['T'].mean(), color='red', linestyle='--', label=f'Mean={rfm["T"].mean():.1f}')
ax.legend()

# Monetary value distribution (for repeat customers)
ax = axes[1, 1]
monetary_repeat = rfm[rfm['frequency'] > 0]['monetary_value']
ax.hist(monetary_repeat.clip(upper=monetary_repeat.quantile(0.95)), bins=50, color='#9C27B0', edgecolor='white', alpha=0.85)
ax.set_title('Monetary Value (Repeat Customers)', fontweight='bold')
ax.set_xlabel('Avg. Spend per Transaction')
ax.set_ylabel('Count')
ax.axvline(monetary_repeat.mean(), color='red', linestyle='--', label=f'Mean={monetary_repeat.mean():.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Frequency-Recency heatmap
fig, ax = plt.subplots(figsize=(12, 8))
rfm_repeat = rfm[rfm['frequency'] > 0].copy()
rfm_repeat['freq_bin'] = pd.cut(rfm_repeat['frequency'], bins=10)
rfm_repeat['recency_bin'] = pd.cut(rfm_repeat['recency'], bins=10)
heatmap_data = rfm_repeat.groupby(['freq_bin', 'recency_bin']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='d', ax=ax)
ax.set_title('Frequency vs Recency Heatmap (Repeat Customers)', fontsize=14, fontweight='bold')
ax.set_xlabel('Recency (Days)')
ax.set_ylabel('Frequency (Repeat Purchases)')
plt.tight_layout()
plt.savefig(BASE_DIR / 'freq_recency_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualizations saved.")
print("Done with data preparation!")
