"""
Step 3: BTYD Analysis using PyMC-Marketing Library
BG/NBD Model + Gamma-Gamma Model → CLV Prediction

Strategy: 
- BG/NBD: MAP for point estimates + 1-chain short MCMC for uncertainty
- Gamma-Gamma: MCMC (faster, only 3 parameters, no customer-level likelihood complexity)
- Record all timing for comparison
"""

import os
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,optimizer=fast_compile'

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
import arviz as az
from pathlib import Path

BASE_DIR = Path(__file__).parent

from pymc_marketing import clv
from pymc_extras.prior import Prior

# ─── Load RFM Summary ────────────────────────────────────────────────────────
rfm = pd.read_parquet(BASE_DIR / 'rfm_summary.parquet')
print(f"RFM shape: {rfm.shape}")

rfm_data = rfm.reset_index()
rfm_data = rfm_data.rename(columns={'Customer ID': 'customer_id'})
rfm_data['customer_id'] = rfm_data['customer_id'].astype(int)
rfm_data['frequency'] = rfm_data['frequency'].astype(float)
rfm_data['recency'] = rfm_data['recency'].astype(float)
rfm_data['T'] = rfm_data['T'].astype(float)
rfm_data['monetary_value'] = rfm_data['monetary_value'].astype(float)

print(rfm_data.head())
print(f"\nCustomers: {len(rfm_data)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1A: BG/NBD MODEL — MAP (Primary approach)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BG/NBD MODEL — PyMC-Marketing (MAP)")
print("="*70)

model_config = {
    "a": Prior("HalfNormal", sigma=10),
    "b": Prior("HalfNormal", sigma=10),
    "alpha": Prior("HalfNormal", sigma=10),
    "r": Prior("HalfNormal", sigma=10),
}

bgm_map = clv.BetaGeoModel(
    data=rfm_data,
    model_config=model_config,
)
bgm_map.build_model()
print(bgm_map)

start_time = time.time()
bgm_map.fit(method="map")
fit_time_bgf_map = time.time() - start_time
print(f"\nMAP Fit time: {fit_time_bgf_map:.1f} seconds")

print("\nBG/NBD Parameter Summary (MAP):")
bgf_map_summary = bgm_map.fit_summary()
print(bgf_map_summary)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1B: BG/NBD MODEL — MCMC (1 chain, 200 draws for uncertainty)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BG/NBD MODEL — PyMC-Marketing (MCMC, 1 chain x 200 draws)")
print("="*70)

bgm = clv.BetaGeoModel(
    data=rfm_data,
    model_config=model_config,
)
bgm.build_model()

start_time = time.time()
bgm.fit(
    chains=1,
    draws=200,
    tune=200,
    random_seed=42,
    progressbar=True,
    cores=1,
)
fit_time_bgf_mcmc = time.time() - start_time
print(f"\nMCMC Fit time: {fit_time_bgf_mcmc:.1f} seconds")

print("\nBG/NBD Parameter Summary (MCMC):")
bgf_mcmc_summary = bgm.fit_summary()
print(bgf_mcmc_summary)

# ─── Predict expected purchases in next 180 days ─────────────────────────────
print("\n--- Predicted Purchases (next 180 days) ---")
expected_purchases = bgm.expected_purchases(future_t=180)

ep_mean = expected_purchases.mean(("chain", "draw")).values
ep_std = expected_purchases.std(("chain", "draw")).values

rfm_data['pymc_predicted_purchases_180d_mean'] = ep_mean
rfm_data['pymc_predicted_purchases_180d_std'] = ep_std

print(f"Mean predicted purchases: {ep_mean.mean():.4f}")
print(pd.Series(ep_mean).describe().round(4))

# ─── P(alive) ────────────────────────────────────────────────────────────────
print("\n--- P(Alive) ---")
p_alive = bgm.expected_probability_alive()
p_alive_mean = p_alive.mean(("chain", "draw")).values
p_alive_std = p_alive.std(("chain", "draw")).values

rfm_data['pymc_p_alive_mean'] = p_alive_mean
rfm_data['pymc_p_alive_std'] = p_alive_std

print(f"Mean P(alive): {p_alive_mean.mean():.4f}")
print(pd.Series(p_alive_mean).describe().round(4))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: GAMMA-GAMMA MODEL — MCMC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("GAMMA-GAMMA MODEL — PyMC-Marketing (MCMC)")
print("="*70)

nonzero_data = rfm_data.query("frequency > 0").copy()
print(f"Repeat customers: {len(nonzero_data)}")

corr = nonzero_data[['frequency', 'monetary_value']].corr().iloc[0, 1]
print(f"Frequency-monetary correlation: {corr:.4f}")

gg = clv.GammaGammaModel(data=nonzero_data)
gg.build_model()
print(gg)

start_time = time.time()
gg.fit(
    chains=1,
    draws=500,
    tune=500,
    random_seed=42,
    progressbar=True,
    cores=1,
)
fit_time_gg_mcmc = time.time() - start_time
print(f"\nGamma-Gamma MCMC Fit time: {fit_time_gg_mcmc:.1f} seconds")

print("\nGamma-Gamma Parameter Summary:")
gg_summary = gg.fit_summary()
print(gg_summary)

# Expected customer spend
print("\n--- Expected Customer Spend ---")
expected_spend = gg.expected_customer_spend(data=rfm_data)
es_mean = expected_spend.mean(("chain", "draw")).values
es_std = expected_spend.std(("chain", "draw")).values

rfm_data['pymc_expected_spend_mean'] = es_mean
rfm_data['pymc_expected_spend_std'] = es_std

print(pd.Series(es_mean).describe().round(2))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: CUSTOMER LIFETIME VALUE — 6 months
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CLV PREDICTION (6 months) — PyMC-Marketing")
print("="*70)

# Thin fit results to speed up CLV computation
bgm.thin_fit_result(keep_every=2)

start_time = time.time()
clv_estimate = gg.expected_customer_lifetime_value(
    transaction_model=bgm,
    data=rfm_data,
    future_t=6,        # 6 months
    discount_rate=0.01, # monthly discount rate
    time_unit="D",      # our data is in days
)
clv_time = time.time() - start_time
print(f"CLV computation time: {clv_time:.1f} seconds")

clv_mean = clv_estimate.mean(("chain", "draw")).values
clv_std = clv_estimate.std(("chain", "draw")).values

rfm_data['pymc_clv_6m_mean'] = clv_mean
rfm_data['pymc_clv_6m_std'] = clv_std

print(f"\nCLV (6 months) distribution:")
print(pd.Series(clv_mean).describe().round(2))

# ─── Top 20 Customers ────────────────────────────────────────────────────────
print("\n--- Top 20 Customers by CLV (PyMC-Marketing) ---")
top20 = rfm_data.nlargest(20, 'pymc_clv_6m_mean')[
    ['customer_id', 'frequency', 'recency', 'T', 'monetary_value',
     'pymc_p_alive_mean', 'pymc_predicted_purchases_180d_mean',
     'pymc_expected_spend_mean', 'pymc_clv_6m_mean', 'pymc_clv_6m_std']
].round(2)
print(top20.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Parameter posteriors — BG/NBD
fig = plt.figure(figsize=(12, 8))
az.plot_posterior(bgm.fit_result)
plt.suptitle('BG/NBD Parameter Posteriors — PyMC-Marketing', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_bgf_posteriors.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Parameter posteriors — Gamma-Gamma
fig = plt.figure(figsize=(12, 4))
az.plot_posterior(gg.fit_result)
plt.suptitle('Gamma-Gamma Parameter Posteriors — PyMC-Marketing', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_gg_posteriors.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. CLV Distribution
fig, ax = plt.subplots(figsize=(10, 6))
clv_clipped = pd.Series(clv_mean).clip(upper=pd.Series(clv_mean).quantile(0.95))
ax.hist(clv_clipped, bins=50, color='#FF5722', edgecolor='white', alpha=0.85)
ax.set_title('CLV Distribution (6 Months) — PyMC-Marketing', fontsize=14, fontweight='bold')
ax.set_xlabel('Customer Lifetime Value (£)')
ax.set_ylabel('Count')
ax.axvline(pd.Series(clv_mean).median(), color='blue', linestyle='--', label=f'Median={pd.Series(clv_mean).median():.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_clv_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. P(alive) Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(p_alive_mean, bins=50, color='#009688', edgecolor='white', alpha=0.85)
ax.set_title('P(Alive) Distribution — PyMC-Marketing', fontsize=14, fontweight='bold')
ax.set_xlabel('Probability Customer is Still Active')
ax.set_ylabel('Count')
ax.axvline(np.median(p_alive_mean), color='red', linestyle='--', label=f'Median={np.median(p_alive_mean):.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_p_alive_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Uncertainty in CLV — top customers
fig, ax = plt.subplots(figsize=(10, 6))
top_idx = rfm_data.nlargest(15, 'pymc_clv_6m_mean').index
top_customers = rfm_data.loc[top_idx]
ax.barh(range(len(top_customers)), top_customers['pymc_clv_6m_mean'], 
        xerr=top_customers['pymc_clv_6m_std'], color='#3F51B5', alpha=0.8, capsize=3)
ax.set_yticks(range(len(top_customers)))
ax.set_yticklabels(top_customers['customer_id'].astype(int))
ax.set_xlabel('CLV (6 months, £)')
ax.set_ylabel('Customer ID')
ax.set_title('Top 15 Customers by CLV with Uncertainty — PyMC-Marketing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_clv_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. P(alive) with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    rfm_data['pymc_p_alive_mean'], 
    rfm_data['pymc_predicted_purchases_180d_mean'],
    c=np.log1p(rfm_data['pymc_clv_6m_mean']), 
    cmap='viridis', alpha=0.4, s=15
)
ax.set_xlabel('P(Alive) — Mean')
ax.set_ylabel('Predicted Purchases (180 days) — Mean')
ax.set_title('P(Alive) vs Predicted Purchases, colored by log(CLV) — PyMC', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='log(1 + CLV)')
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_palive_vs_purchases.png', dpi=150, bbox_inches='tight')
plt.close()

# 7. Trace plots for diagnostics
fig = plt.figure(figsize=(14, 10))
az.plot_trace(bgm.fit_result, compact=True)
plt.suptitle('BG/NBD MCMC Trace Plots — PyMC-Marketing', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'pymc_bgf_traces.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll PyMC-Marketing visualizations saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

bgf_params = {}
for var in bgm.fit_result.data_vars:
    bgf_params[var] = float(bgm.fit_result[var].mean().values)

gg_params = {}
for var in gg.fit_result.data_vars:
    gg_params[var] = float(gg.fit_result[var].mean().values)

bgf_map_params = {}
for var in bgm_map.fit_result.data_vars:
    val = bgm_map.fit_result[var].values
    bgf_map_params[var] = float(val.item() if val.ndim == 0 else val.mean())

pymc_results = {
    'bgf_params_mcmc': bgf_params,
    'bgf_params_map': bgf_map_params,
    'ggf_params_mcmc': gg_params,
    'fit_time_bgf_mcmc': fit_time_bgf_mcmc,
    'fit_time_bgf_map': fit_time_bgf_map,
    'fit_time_gg_mcmc': fit_time_gg_mcmc,
    'clv_computation_time': clv_time,
    'clv_stats': pd.Series(clv_mean).describe().to_dict(),
    'p_alive_stats': pd.Series(p_alive_mean).describe().to_dict(),
    'predicted_purchases_stats': pd.Series(ep_mean).describe().to_dict(),
    'clv_uncertainty_stats': pd.Series(clv_std).describe().to_dict(),
}

with open(BASE_DIR / 'pymc_results.json', 'w') as f:
    json.dump(pymc_results, f, indent=2, default=str)

rfm_data.to_parquet(BASE_DIR / 'pymc_customer_results.parquet')

print("\nPyMC-Marketing results saved.")
print("Done!")
