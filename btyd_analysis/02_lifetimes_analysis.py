"""
Step 2: BTYD Analysis using the Lifetimes Library
BG/NBD Model + Gamma-Gamma Model → CLV Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions,
    plot_history_alive,
)
from lifetimes.utils import calibration_and_holdout_data

# ─── Load RFM Summary ────────────────────────────────────────────────────────
rfm = pd.read_parquet(BASE_DIR / 'rfm_summary.parquet')
print(f"RFM shape: {rfm.shape}")
print(rfm.head(10))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: BG/NBD MODEL (Transaction Frequency)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BG/NBD MODEL — Lifetimes Library")
print("="*70)

bgf = BetaGeoFitter(penalizer_coef=0.001)

start_time = time.time()
bgf.fit(
    rfm['frequency'],
    rfm['recency'],
    rfm['T']
)
fit_time_bgf = time.time() - start_time

print(f"\nFit time: {fit_time_bgf:.3f} seconds")
print(f"\nBG/NBD Parameters:")
print(f"  r     = {bgf.params_['r']:.6f}")
print(f"  alpha = {bgf.params_['alpha']:.6f}")
print(f"  a     = {bgf.params_['a']:.6f}")
print(f"  b     = {bgf.params_['b']:.6f}")
print(f"\nSummary: {bgf.summary}")

# ─── Predict expected purchases in next 180 days (6 months) ──────────────────
rfm['predicted_purchases_180d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    180,  # 6 months ≈ 180 days
    rfm['frequency'],
    rfm['recency'],
    rfm['T']
)

print(f"\nPredicted purchases (next 180 days):")
print(rfm['predicted_purchases_180d'].describe().round(4))

# ─── P(alive) ────────────────────────────────────────────────────────────────
rfm['p_alive'] = bgf.conditional_probability_alive(
    rfm['frequency'],
    rfm['recency'],
    rfm['T']
)

print(f"\nP(alive) distribution:")
print(rfm['p_alive'].describe().round(4))

# ─── Model Validation: Calibration/Holdout ────────────────────────────────────
# Load raw transactions for calibration/holdout split
df = pd.read_parquet(BASE_DIR / 'cleaned_transactions.parquet')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Split: calibration = first ~70% of time, holdout = last ~30%
observation_end = df['InvoiceDate'].max()
total_days = (observation_end - df['InvoiceDate'].min()).days
cal_end = df['InvoiceDate'].min() + pd.Timedelta(days=int(total_days * 0.7))

print(f"\nCalibration period: {df['InvoiceDate'].min().date()} to {cal_end.date()}")
print(f"Holdout period: {cal_end.date()} to {observation_end.date()}")

cal_holdout = calibration_and_holdout_data(
    df,
    customer_id_col='Customer ID',
    datetime_col='InvoiceDate',
    calibration_period_end=cal_end,
    observation_period_end=observation_end,
    freq='D'
)

# Fit on calibration data
bgf_cal = BetaGeoFitter(penalizer_coef=0.001)
bgf_cal.fit(
    cal_holdout['frequency_cal'],
    cal_holdout['recency_cal'],
    cal_holdout['T_cal']
)

# Predict holdout purchases
holdout_days = (observation_end - cal_end).days
cal_holdout['predicted_holdout'] = bgf_cal.conditional_expected_number_of_purchases_up_to_time(
    holdout_days,
    cal_holdout['frequency_cal'],
    cal_holdout['recency_cal'],
    cal_holdout['T_cal']
)

# Validation metrics
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

actual = cal_holdout['frequency_holdout']
predicted = cal_holdout['predicted_holdout']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
pearson_r, _ = pearsonr(actual, predicted)
spearman_r, _ = spearmanr(actual, predicted)

print(f"\n--- BG/NBD Validation (Calibration/Holdout) ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Pearson r:  {pearson_r:.4f}")
print(f"Spearman r: {spearman_r:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: GAMMA-GAMMA MODEL (Monetary Value)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("GAMMA-GAMMA MODEL — Lifetimes Library")
print("="*70)

# Gamma-Gamma requires frequency > 0
rfm_gg = rfm[rfm['frequency'] > 0].copy()

# Check independence assumption: correlation between frequency and monetary_value
corr = rfm_gg[['frequency', 'monetary_value']].corr().iloc[0, 1]
print(f"\nCorrelation between frequency and monetary_value: {corr:.4f}")
print("(Should be close to 0 for Gamma-Gamma assumption to hold)")

ggf = GammaGammaFitter(penalizer_coef=0.001)

start_time = time.time()
ggf.fit(
    rfm_gg['frequency'],
    rfm_gg['monetary_value']
)
fit_time_gg = time.time() - start_time

print(f"\nFit time: {fit_time_gg:.3f} seconds")
print(f"\nGamma-Gamma Parameters:")
print(f"  p = {ggf.params_['p']:.6f}")
print(f"  q = {ggf.params_['q']:.6f}")
print(f"  v = {ggf.params_['v']:.6f}")
print(f"\nSummary: {ggf.summary}")

# Expected average profit per transaction
rfm_gg['expected_avg_profit'] = ggf.conditional_expected_average_profit(
    rfm_gg['frequency'],
    rfm_gg['monetary_value']
)

print(f"\nExpected avg profit per transaction:")
print(rfm_gg['expected_avg_profit'].describe().round(2))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: CUSTOMER LIFETIME VALUE (CLV) — 6 months
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CLV PREDICTION (6 months) — Lifetimes Library")
print("="*70)

# CLV for next 6 months (180 days)
# Using monthly discount rate of 1% (annual ~12.7%)
rfm_gg['clv_6m'] = ggf.customer_lifetime_value(
    bgf,
    rfm_gg['frequency'],
    rfm_gg['recency'],
    rfm_gg['T'],
    rfm_gg['monetary_value'],
    time=6,  # 6 months
    discount_rate=0.01,  # monthly discount rate
    freq='D'
)

print(f"\nCLV (6 months) distribution:")
print(rfm_gg['clv_6m'].describe().round(2))

# ─── Merge P(alive) and predicted purchases back ─────────────────────────────
rfm_gg['p_alive'] = rfm.loc[rfm_gg.index, 'p_alive']
rfm_gg['predicted_purchases_180d'] = rfm.loc[rfm_gg.index, 'predicted_purchases_180d']

# ─── Top/Bottom customers ────────────────────────────────────────────────────
print("\n--- Top 20 Customers by CLV (6 months) ---")
top20 = rfm_gg.nlargest(20, 'clv_6m')[
    ['frequency', 'recency', 'T', 'monetary_value', 'p_alive', 
     'predicted_purchases_180d', 'expected_avg_profit', 'clv_6m']
].round(2)
print(top20.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Frequency-Recency Matrix
fig = plt.figure(figsize=(10, 8))
plot_frequency_recency_matrix(bgf, T=180)
plt.title('Expected Purchases in Next 180 Days\n(Frequency-Recency Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_freq_recency_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Probability Alive Matrix
fig = plt.figure(figsize=(10, 8))
plot_probability_alive_matrix(bgf)
plt.title('Probability Customer is Alive\n(Frequency-Recency Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_p_alive_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Model fit: predicted vs actual repeat transactions
fig = plt.figure(figsize=(10, 6))
plot_period_transactions(bgf)
plt.title('BG/NBD Model Fit: Predicted vs Actual Repeat Transactions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_model_fit.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. CLV Distribution
fig, ax = plt.subplots(figsize=(10, 6))
clv_clipped = rfm_gg['clv_6m'].clip(upper=rfm_gg['clv_6m'].quantile(0.95))
ax.hist(clv_clipped, bins=50, color='#2196F3', edgecolor='white', alpha=0.85)
ax.set_title('CLV Distribution (6 Months) — Lifetimes', fontsize=14, fontweight='bold')
ax.set_xlabel('Customer Lifetime Value (£)')
ax.set_ylabel('Count')
ax.axvline(rfm_gg['clv_6m'].median(), color='red', linestyle='--', label=f'Median={rfm_gg["clv_6m"].median():.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_clv_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. P(alive) Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(rfm_gg['p_alive'], bins=50, color='#4CAF50', edgecolor='white', alpha=0.85)
ax.set_title('P(Alive) Distribution — Lifetimes', fontsize=14, fontweight='bold')
ax.set_xlabel('Probability Customer is Still Active')
ax.set_ylabel('Count')
ax.axvline(rfm_gg['p_alive'].median(), color='red', linestyle='--', label=f'Median={rfm_gg["p_alive"].median():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_p_alive_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. Calibration vs Holdout scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(cal_holdout['frequency_holdout'], cal_holdout['predicted_holdout'], 
           alpha=0.15, s=10, color='#2196F3')
max_val = max(cal_holdout['frequency_holdout'].max(), cal_holdout['predicted_holdout'].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Holdout Purchases')
ax.set_ylabel('Predicted Holdout Purchases')
ax.set_title(f'BG/NBD Calibration/Holdout Validation\nMAE={mae:.2f}, Pearson r={pearson_r:.3f}', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_cal_holdout.png', dpi=150, bbox_inches='tight')
plt.close()

# 7. Predicted purchases vs P(alive) scatter
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(rfm_gg['p_alive'], rfm_gg['predicted_purchases_180d'],
                     c=np.log1p(rfm_gg['clv_6m']), cmap='viridis', alpha=0.4, s=15)
ax.set_xlabel('P(Alive)')
ax.set_ylabel('Predicted Purchases (180 days)')
ax.set_title('P(Alive) vs Predicted Purchases, colored by log(CLV)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='log(1 + CLV)')
plt.tight_layout()
plt.savefig(BASE_DIR / 'lifetimes_palive_vs_purchases.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll Lifetimes visualizations saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

# Save key results
lifetimes_results = {
    'bgf_params': {k: float(v) for k, v in bgf.params_.items()},
    'ggf_params': {k: float(v) for k, v in ggf.params_.items()},
    'fit_time_bgf': fit_time_bgf,
    'fit_time_gg': fit_time_gg,
    'validation': {
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
    },
    'clv_stats': rfm_gg['clv_6m'].describe().to_dict(),
    'p_alive_stats': rfm_gg['p_alive'].describe().to_dict(),
    'predicted_purchases_stats': rfm_gg['predicted_purchases_180d'].describe().to_dict(),
}

with open(BASE_DIR / 'lifetimes_results.json', 'w') as f:
    json.dump(lifetimes_results, f, indent=2, default=str)

# Save customer-level results
rfm_gg.to_parquet(BASE_DIR / 'lifetimes_customer_results.parquet')

# Also save full rfm with p_alive for all customers (including freq=0)
rfm.to_parquet(BASE_DIR / 'lifetimes_rfm_full.parquet')

print("\nLifetimes results saved.")
print("Done!")
