"""
Step 4: Head-to-Head Comparison — Lifetimes vs PyMC-Marketing
Compare parameters, predictions, CLV, P(alive), timing, and agreement
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).parent

# ─── Load results ─────────────────────────────────────────────────────────────
with open(BASE_DIR / 'lifetimes_results.json') as f:
    lt_results = json.load(f)

with open(BASE_DIR / 'pymc_results.json') as f:
    pymc_results = json.load(f)

lt_customers = pd.read_parquet(BASE_DIR / 'lifetimes_customer_results.parquet')
pymc_customers = pd.read_parquet(BASE_DIR / 'pymc_customer_results.parquet')
lt_rfm_full = pd.read_parquet(BASE_DIR / 'lifetimes_rfm_full.parquet')

# Merge on customer_id for comparison
# Lifetimes has Customer ID as index
lt_customers_reset = lt_customers.reset_index()
lt_customers_reset = lt_customers_reset.rename(columns={'Customer ID': 'customer_id'})

# Merge
merged = pd.merge(
    lt_customers_reset[['customer_id', 'frequency', 'recency', 'T', 'monetary_value',
                         'predicted_purchases_180d', 'p_alive', 'expected_avg_profit', 'clv_6m']],
    pymc_customers[['customer_id', 'pymc_predicted_purchases_180d_mean', 'pymc_predicted_purchases_180d_std',
                     'pymc_p_alive_mean', 'pymc_p_alive_std',
                     'pymc_expected_spend_mean', 'pymc_expected_spend_std',
                     'pymc_clv_6m_mean', 'pymc_clv_6m_std']],
    on='customer_id',
    how='inner'
)

# Also merge full rfm for all customers (including freq=0)
lt_rfm_full_reset = lt_rfm_full.reset_index()
lt_rfm_full_reset = lt_rfm_full_reset.rename(columns={'Customer ID': 'customer_id'})

merged_all = pd.merge(
    lt_rfm_full_reset[['customer_id', 'frequency', 'recency', 'T', 'monetary_value',
                         'predicted_purchases_180d', 'p_alive']],
    pymc_customers[['customer_id', 'pymc_predicted_purchases_180d_mean', 'pymc_p_alive_mean',
                     'pymc_clv_6m_mean']],
    on='customer_id',
    how='inner'
)

print(f"Merged repeat customers: {len(merged)}")
print(f"Merged all customers: {len(merged_all)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("1. PARAMETER COMPARISON")
print("="*70)

# BG/NBD parameters
print("\n--- BG/NBD Parameters ---")
lt_bgf = lt_results['bgf_params']
pymc_bgf_mcmc = pymc_results['bgf_params_mcmc']
pymc_bgf_map = pymc_results['bgf_params_map']

# Note: PyMC-Marketing uses a hierarchical parameterization for a,b
# with phi_dropout and kappa_dropout. We need to extract a and b.
print(f"{'Parameter':<12} {'Lifetimes (MLE)':<20} {'PyMC MAP':<20} {'PyMC MCMC (mean)':<20}")
print("-"*72)

# Map PyMC params to Lifetimes params
# PyMC uses: alpha, r, a, b (+ phi_dropout, kappa_dropout for hierarchical)
for param in ['r', 'alpha', 'a', 'b']:
    lt_val = lt_bgf.get(param, 'N/A')
    pymc_map_val = pymc_bgf_map.get(param, 'N/A')
    pymc_mcmc_val = pymc_bgf_mcmc.get(param, 'N/A')
    if lt_val != 'N/A':
        print(f"{param:<12} {lt_val:<20.6f} {pymc_map_val if pymc_map_val != 'N/A' else 'N/A':<20} {pymc_mcmc_val if pymc_mcmc_val != 'N/A' else 'N/A':<20}")

print("\nAll PyMC MCMC params:")
for k, v in pymc_bgf_mcmc.items():
    print(f"  {k}: {v:.6f}")

print("\nAll PyMC MAP params:")
for k, v in pymc_bgf_map.items():
    print(f"  {k}: {v:.6f}")

# Gamma-Gamma parameters
print("\n--- Gamma-Gamma Parameters ---")
lt_gg = lt_results['ggf_params']
pymc_gg = pymc_results['ggf_params_mcmc']

print(f"{'Parameter':<12} {'Lifetimes (MLE)':<20} {'PyMC MCMC (mean)':<20}")
print("-"*52)
for param in ['p', 'q', 'v']:
    lt_val = lt_gg.get(param, 'N/A')
    pymc_val = pymc_gg.get(param, 'N/A')
    if lt_val != 'N/A' and pymc_val != 'N/A':
        print(f"{param:<12} {lt_val:<20.6f} {pymc_val:<20.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREDICTION COMPARISON (Customer-Level)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("2. PREDICTION COMPARISON (Customer-Level)")
print("="*70)

# --- Predicted Purchases ---
print("\n--- Predicted Purchases (180 days) ---")
pp_corr_pearson, _ = pearsonr(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean'])
pp_corr_spearman, _ = spearmanr(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean'])
pp_mae = mean_absolute_error(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean'])
pp_rmse = np.sqrt(mean_squared_error(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean']))

print(f"Pearson correlation:  {pp_corr_pearson:.6f}")
print(f"Spearman correlation: {pp_corr_spearman:.6f}")
print(f"MAE between libs:     {pp_mae:.4f}")
print(f"RMSE between libs:    {pp_rmse:.4f}")
print(f"Lifetimes mean:       {merged_all['predicted_purchases_180d'].mean():.4f}")
print(f"PyMC mean:            {merged_all['pymc_predicted_purchases_180d_mean'].mean():.4f}")

# --- P(alive) ---
print("\n--- P(Alive) ---")
pa_corr_pearson, _ = pearsonr(merged_all['p_alive'], merged_all['pymc_p_alive_mean'])
pa_corr_spearman, _ = spearmanr(merged_all['p_alive'], merged_all['pymc_p_alive_mean'])
pa_mae = mean_absolute_error(merged_all['p_alive'], merged_all['pymc_p_alive_mean'])

print(f"Pearson correlation:  {pa_corr_pearson:.6f}")
print(f"Spearman correlation: {pa_corr_spearman:.6f}")
print(f"MAE between libs:     {pa_mae:.4f}")
print(f"Lifetimes mean:       {merged_all['p_alive'].mean():.4f}")
print(f"PyMC mean:            {merged_all['pymc_p_alive_mean'].mean():.4f}")

# --- CLV ---
print("\n--- CLV (6 months) — Repeat Customers Only ---")
clv_corr_pearson, _ = pearsonr(merged['clv_6m'], merged['pymc_clv_6m_mean'])
clv_corr_spearman, _ = spearmanr(merged['clv_6m'], merged['pymc_clv_6m_mean'])
clv_mae = mean_absolute_error(merged['clv_6m'], merged['pymc_clv_6m_mean'])
clv_rmse = np.sqrt(mean_squared_error(merged['clv_6m'], merged['pymc_clv_6m_mean']))

print(f"Pearson correlation:  {clv_corr_pearson:.6f}")
print(f"Spearman correlation: {clv_corr_spearman:.6f}")
print(f"MAE between libs:     {clv_mae:.2f}")
print(f"RMSE between libs:    {clv_rmse:.2f}")
print(f"Lifetimes mean CLV:   {merged['clv_6m'].mean():.2f}")
print(f"PyMC mean CLV:        {merged['pymc_clv_6m_mean'].mean():.2f}")
print(f"Lifetimes median CLV: {merged['clv_6m'].median():.2f}")
print(f"PyMC median CLV:      {merged['pymc_clv_6m_mean'].median():.2f}")
print(f"Lifetimes total CLV:  {merged['clv_6m'].sum():.0f}")
print(f"PyMC total CLV:       {merged['pymc_clv_6m_mean'].sum():.0f}")

# --- Expected Spend ---
print("\n--- Expected Average Spend (Repeat Customers) ---")
es_corr_pearson, _ = pearsonr(merged['expected_avg_profit'], merged['pymc_expected_spend_mean'])
es_corr_spearman, _ = spearmanr(merged['expected_avg_profit'], merged['pymc_expected_spend_mean'])
es_mae = mean_absolute_error(merged['expected_avg_profit'], merged['pymc_expected_spend_mean'])

print(f"Pearson correlation:  {es_corr_pearson:.6f}")
print(f"Spearman correlation: {es_corr_spearman:.6f}")
print(f"MAE between libs:     {es_mae:.2f}")
print(f"Lifetimes mean:       {merged['expected_avg_profit'].mean():.2f}")
print(f"PyMC mean:            {merged['pymc_expected_spend_mean'].mean():.2f}")

# --- KS Tests ---
print("\n--- Kolmogorov-Smirnov Tests ---")
ks_pp, p_pp = ks_2samp(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean'])
ks_pa, p_pa = ks_2samp(merged_all['p_alive'], merged_all['pymc_p_alive_mean'])
ks_clv, p_clv = ks_2samp(merged['clv_6m'], merged['pymc_clv_6m_mean'])

print(f"Predicted Purchases: KS={ks_pp:.4f}, p={p_pp:.6f}")
print(f"P(Alive):            KS={ks_pa:.4f}, p={p_pa:.6f}")
print(f"CLV:                 KS={ks_clv:.4f}, p={p_clv:.6f}")

# --- Ranking Agreement ---
print("\n--- Ranking Agreement (Top 50 Customers) ---")
lt_top50 = set(merged.nlargest(50, 'clv_6m')['customer_id'])
pymc_top50 = set(merged.nlargest(50, 'pymc_clv_6m_mean')['customer_id'])
overlap = len(lt_top50 & pymc_top50)
print(f"Top 50 CLV overlap: {overlap}/50 ({overlap/50*100:.0f}%)")

lt_top100 = set(merged.nlargest(100, 'clv_6m')['customer_id'])
pymc_top100 = set(merged.nlargest(100, 'pymc_clv_6m_mean')['customer_id'])
overlap100 = len(lt_top100 & pymc_top100)
print(f"Top 100 CLV overlap: {overlap100}/100 ({overlap100/100*100:.0f}%)")

# Rank correlation
merged['lt_clv_rank'] = merged['clv_6m'].rank(ascending=False)
merged['pymc_clv_rank'] = merged['pymc_clv_6m_mean'].rank(ascending=False)
rank_corr, _ = spearmanr(merged['lt_clv_rank'], merged['pymc_clv_rank'])
print(f"Rank correlation (Spearman): {rank_corr:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TIMING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("3. TIMING COMPARISON")
print("="*70)

print(f"\n{'Operation':<35} {'Lifetimes':<15} {'PyMC MAP':<15} {'PyMC MCMC':<15} {'Ratio (MCMC/LT)':<15}")
print("-"*95)
print(f"{'BG/NBD Fit':<35} {lt_results['fit_time_bgf']:<15.3f} {pymc_results['fit_time_bgf_map']:<15.1f} {pymc_results['fit_time_bgf_mcmc']:<15.1f} {pymc_results['fit_time_bgf_mcmc']/lt_results['fit_time_bgf']:<15.0f}")
print(f"{'Gamma-Gamma Fit':<35} {lt_results['fit_time_gg']:<15.3f} {'N/A':<15} {pymc_results['fit_time_gg_mcmc']:<15.1f} {pymc_results['fit_time_gg_mcmc']/lt_results['fit_time_gg']:<15.0f}")
print(f"{'CLV Computation':<35} {'<0.1':<15} {'N/A':<15} {pymc_results['clv_computation_time']:<15.1f} {'N/A':<15}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Predicted Purchases scatter
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Lifetimes vs PyMC-Marketing: Head-to-Head Comparison', fontsize=16, fontweight='bold')

ax = axes[0]
ax.scatter(merged_all['predicted_purchases_180d'], merged_all['pymc_predicted_purchases_180d_mean'],
           alpha=0.2, s=10, color='#2196F3')
max_val = max(merged_all['predicted_purchases_180d'].max(), merged_all['pymc_predicted_purchases_180d_mean'].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
ax.set_xlabel('Lifetimes')
ax.set_ylabel('PyMC-Marketing')
ax.set_title(f'Predicted Purchases (180d)\nr={pp_corr_pearson:.4f}', fontweight='bold')
ax.legend()

ax = axes[1]
ax.scatter(merged_all['p_alive'], merged_all['pymc_p_alive_mean'],
           alpha=0.2, s=10, color='#4CAF50')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
ax.set_xlabel('Lifetimes')
ax.set_ylabel('PyMC-Marketing')
ax.set_title(f'P(Alive)\nr={pa_corr_pearson:.4f}', fontweight='bold')
ax.legend()

ax = axes[2]
ax.scatter(merged['clv_6m'], merged['pymc_clv_6m_mean'],
           alpha=0.2, s=10, color='#FF9800')
max_val = max(merged['clv_6m'].max(), merged['pymc_clv_6m_mean'].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
ax.set_xlabel('Lifetimes')
ax.set_ylabel('PyMC-Marketing')
ax.set_title(f'CLV (6 months)\nr={clv_corr_pearson:.4f}', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution Comparison: Lifetimes vs PyMC-Marketing', fontsize=16, fontweight='bold')

# Predicted purchases
ax = axes[0, 0]
bins = np.linspace(0, 10, 50)
ax.hist(merged_all['predicted_purchases_180d'].clip(upper=10), bins=bins, alpha=0.6, color='#2196F3', label='Lifetimes', density=True)
ax.hist(merged_all['pymc_predicted_purchases_180d_mean'].clip(upper=10), bins=bins, alpha=0.6, color='#FF5722', label='PyMC', density=True)
ax.set_title('Predicted Purchases (180d)', fontweight='bold')
ax.legend()

# P(alive)
ax = axes[0, 1]
bins = np.linspace(0, 1, 50)
ax.hist(merged_all['p_alive'], bins=bins, alpha=0.6, color='#2196F3', label='Lifetimes', density=True)
ax.hist(merged_all['pymc_p_alive_mean'], bins=bins, alpha=0.6, color='#FF5722', label='PyMC', density=True)
ax.set_title('P(Alive)', fontweight='bold')
ax.legend()

# CLV
ax = axes[1, 0]
q95 = max(merged['clv_6m'].quantile(0.95), merged['pymc_clv_6m_mean'].quantile(0.95))
bins = np.linspace(0, q95, 50)
ax.hist(merged['clv_6m'].clip(upper=q95), bins=bins, alpha=0.6, color='#2196F3', label='Lifetimes', density=True)
ax.hist(merged['pymc_clv_6m_mean'].clip(upper=q95), bins=bins, alpha=0.6, color='#FF5722', label='PyMC', density=True)
ax.set_title('CLV (6 months)', fontweight='bold')
ax.set_xlabel('£')
ax.legend()

# Expected spend
ax = axes[1, 1]
q95 = max(merged['expected_avg_profit'].quantile(0.95), merged['pymc_expected_spend_mean'].quantile(0.95))
bins = np.linspace(0, q95, 50)
ax.hist(merged['expected_avg_profit'].clip(upper=q95), bins=bins, alpha=0.6, color='#2196F3', label='Lifetimes', density=True)
ax.hist(merged['pymc_expected_spend_mean'].clip(upper=q95), bins=bins, alpha=0.6, color='#FF5722', label='PyMC', density=True)
ax.set_title('Expected Avg Spend', fontweight='bold')
ax.set_xlabel('£')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Difference analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Prediction Differences: PyMC − Lifetimes', fontsize=16, fontweight='bold')

ax = axes[0]
diff_pp = merged_all['pymc_predicted_purchases_180d_mean'] - merged_all['predicted_purchases_180d']
ax.hist(diff_pp.clip(-2, 2), bins=50, color='#9C27B0', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--')
ax.axvline(diff_pp.mean(), color='blue', linestyle='--', label=f'Mean={diff_pp.mean():.3f}')
ax.set_title('Predicted Purchases Diff', fontweight='bold')
ax.set_xlabel('PyMC − Lifetimes')
ax.legend()

ax = axes[1]
diff_pa = merged_all['pymc_p_alive_mean'] - merged_all['p_alive']
ax.hist(diff_pa.clip(-0.3, 0.3), bins=50, color='#009688', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--')
ax.axvline(diff_pa.mean(), color='blue', linestyle='--', label=f'Mean={diff_pa.mean():.4f}')
ax.set_title('P(Alive) Diff', fontweight='bold')
ax.set_xlabel('PyMC − Lifetimes')
ax.legend()

ax = axes[2]
diff_clv = merged['pymc_clv_6m_mean'] - merged['clv_6m']
ax.hist(diff_clv.clip(-500, 500), bins=50, color='#FF5722', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--')
ax.axvline(diff_clv.mean(), color='blue', linestyle='--', label=f'Mean={diff_clv.mean():.1f}')
ax.set_title('CLV Diff', fontweight='bold')
ax.set_xlabel('PyMC − Lifetimes (£)')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_differences.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Rank comparison
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(merged['lt_clv_rank'], merged['pymc_clv_rank'], alpha=0.15, s=10, color='#3F51B5')
ax.plot([0, len(merged)], [0, len(merged)], 'r--', linewidth=2, label='Perfect agreement')
ax.set_xlabel('Lifetimes CLV Rank')
ax.set_ylabel('PyMC CLV Rank')
ax.set_title(f'Customer CLV Ranking Agreement\nSpearman r = {rank_corr:.4f}', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_rankings.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Uncertainty visualization (PyMC advantage)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PyMC-Marketing Uncertainty Quantification (Unique Advantage)', fontsize=16, fontweight='bold')

ax = axes[0]
# Sort by CLV mean and show error bars for a sample
sample = merged.nlargest(30, 'pymc_clv_6m_mean')
ax.errorbar(range(len(sample)), sample['pymc_clv_6m_mean'], yerr=sample['pymc_clv_6m_std'],
            fmt='o', color='#FF5722', capsize=3, markersize=4, label='PyMC (mean ± std)')
ax.scatter(range(len(sample)), sample['clv_6m'], color='#2196F3', s=30, zorder=5, label='Lifetimes (point)')
ax.set_xlabel('Customer (sorted by PyMC CLV)')
ax.set_ylabel('CLV (£)')
ax.set_title('Top 30 Customers: CLV with Uncertainty', fontweight='bold')
ax.legend()

ax = axes[1]
# Coefficient of variation for CLV
cv = merged['pymc_clv_6m_std'] / (merged['pymc_clv_6m_mean'] + 1e-6)
ax.hist(cv.clip(upper=1), bins=50, color='#FF9800', edgecolor='white', alpha=0.85)
ax.set_title('CLV Coefficient of Variation (PyMC)', fontweight='bold')
ax.set_xlabel('CV = std / mean')
ax.set_ylabel('Count')
ax.axvline(cv.median(), color='red', linestyle='--', label=f'Median CV={cv.median():.3f}')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. Segment-level comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Segment-Level Agreement', fontsize=16, fontweight='bold')

# By frequency bucket
merged_all['freq_bucket'] = pd.cut(merged_all['frequency'], bins=[-1, 0, 2, 5, 10, 20, 300], 
                                    labels=['0', '1-2', '3-5', '6-10', '11-20', '20+'])
seg = merged_all.groupby('freq_bucket').agg(
    lt_pp=('predicted_purchases_180d', 'mean'),
    pymc_pp=('pymc_predicted_purchases_180d_mean', 'mean'),
    lt_pa=('p_alive', 'mean'),
    pymc_pa=('pymc_p_alive_mean', 'mean'),
).reset_index()

ax = axes[0]
x = range(len(seg))
width = 0.35
ax.bar([i - width/2 for i in x], seg['lt_pp'], width, color='#2196F3', label='Lifetimes', alpha=0.8)
ax.bar([i + width/2 for i in x], seg['pymc_pp'], width, color='#FF5722', label='PyMC', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(seg['freq_bucket'])
ax.set_xlabel('Frequency Bucket')
ax.set_ylabel('Mean Predicted Purchases (180d)')
ax.set_title('Predicted Purchases by Segment', fontweight='bold')
ax.legend()

ax = axes[1]
ax.bar([i - width/2 for i in x], seg['lt_pa'], width, color='#2196F3', label='Lifetimes', alpha=0.8)
ax.bar([i + width/2 for i in x], seg['pymc_pa'], width, color='#FF5722', label='PyMC', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(seg['freq_bucket'])
ax.set_xlabel('Frequency Bucket')
ax.set_ylabel('Mean P(Alive)')
ax.set_title('P(Alive) by Segment', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / 'comparison_segments.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll comparison visualizations saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

comparison_summary = {
    'predicted_purchases': {
        'pearson_r': pp_corr_pearson,
        'spearman_r': pp_corr_spearman,
        'mae': pp_mae,
        'rmse': pp_rmse,
        'lt_mean': merged_all['predicted_purchases_180d'].mean(),
        'pymc_mean': merged_all['pymc_predicted_purchases_180d_mean'].mean(),
    },
    'p_alive': {
        'pearson_r': pa_corr_pearson,
        'spearman_r': pa_corr_spearman,
        'mae': pa_mae,
        'lt_mean': merged_all['p_alive'].mean(),
        'pymc_mean': merged_all['pymc_p_alive_mean'].mean(),
    },
    'clv': {
        'pearson_r': clv_corr_pearson,
        'spearman_r': clv_corr_spearman,
        'mae': clv_mae,
        'rmse': clv_rmse,
        'lt_mean': merged['clv_6m'].mean(),
        'pymc_mean': merged['pymc_clv_6m_mean'].mean(),
        'lt_total': merged['clv_6m'].sum(),
        'pymc_total': merged['pymc_clv_6m_mean'].sum(),
    },
    'ranking': {
        'top50_overlap': overlap,
        'top100_overlap': overlap100,
        'rank_spearman': rank_corr,
    },
    'ks_tests': {
        'pp_ks': ks_pp, 'pp_p': p_pp,
        'pa_ks': ks_pa, 'pa_p': p_pa,
        'clv_ks': ks_clv, 'clv_p': p_clv,
    }
}

with open(BASE_DIR / 'comparison_summary.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2, default=str)

print("Comparison summary saved.")
print("Done!")
