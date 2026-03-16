# Survival Analysis — Glossary & Definitions Guide

A practical reference for understanding survival analysis concepts, metrics, and how they differ from standard machine learning. Written for readers encountering survival analysis for the first time.

---

## Table of Contents

1. [What Is Survival Analysis?](#1-what-is-survival-analysis)
2. [Core Concepts](#2-core-concepts)
3. [How Survival Models Differ from Standard ML](#3-how-survival-models-differ-from-standard-ml)
4. [The Target Variable — What Are We Predicting?](#4-the-target-variable--what-are-we-predicting)
5. [Key Functions](#5-key-functions)
6. [Models Used in This Project](#6-models-used-in-this-project)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [How to Judge If a Model Is Trustworthy](#8-how-to-judge-if-a-model-is-trustworthy)
9. [Glossary — Quick Reference](#9-glossary--quick-reference)

---

## 1. What Is Survival Analysis?

Survival analysis models **time until an event occurs**. Originally developed in medicine (time until death or relapse), it applies to any duration outcome:

| Domain | Event | Duration |
|--------|-------|----------|
| Medicine | Patient death | Days since diagnosis |
| Engineering | Machine failure | Hours of operation |
| HR | Employee resignation | Months since hire |
| **Customer analytics** | **Customer churn** | **Days since first purchase** |

The key question is not *"will the customer churn?"* (binary yes/no), but *"when will the customer churn, and what is their probability of still being active at day 30, 90, or 180?"*

This shift from **if** to **when** is what makes survival analysis more informative than standard classification.

---

## 2. Core Concepts

### Survival Time (Duration)

The time elapsed from a defined starting point to the event of interest.

In this project: the number of days from a customer's **first purchase** to the date they are considered **churned** (no purchase within a 45-day window).

A customer who first bought on Jan 1 and last bought on Mar 15 (then went silent for 45+ days) has a survival time of **74 days** (Jan 1 to Mar 15).

### Event (Failure)

The occurrence we are studying. In customer analytics, the event is **churn** — the customer stops purchasing.

- **Event observed (E=1):** The customer churned during the observation period.
- **Event not observed (E=0):** The customer was still active when our data ended (see *censoring* below).

### Censoring

The single most important concept that separates survival analysis from regular ML.

**Right censoring** occurs when we know a customer was still active at a certain date, but we do not know when (or if) they will eventually churn. This happens because:

- The study period ended (data collection stopped on Dec 9, 2011)
- The customer is genuinely still active

**Example:** A customer's last purchase was Nov 25, 2011, and our data ends Dec 9, 2011 — only 14 days of silence. We cannot label them as churned (they may buy again next week), but we also cannot label them as permanent. They are **censored**.

**Why this matters:** In standard classification, you would either (a) drop these customers (losing data) or (b) guess their label (introducing bias). Survival analysis handles censored observations natively — it uses the partial information ("this customer survived *at least* this long") without requiring a definitive outcome.

### Observation Window

The total time span of available data. In this project: Dec 2009 to Dec 2011 (~2 years). Customers who joined late in this window have shorter observation periods, making censoring more prevalent for recent cohorts.

### Churn Window (Inactivity Threshold)

The number of days of inactivity after which a customer is considered churned. This project uses **45 days**, validated empirically by analyzing the distribution of inter-purchase gaps (see `churn-window-analysis.ipynb`).

If a customer's last purchase was 45+ days ago and they have not returned, they are labeled as churned.

---

## 3. How Survival Models Differ from Standard ML

### Standard Supervised Classification

| Aspect | Standard Classification |
|--------|------------------------|
| **Target** | Binary label: churned (1) or not (0) |
| **Output** | P(churn) — a single probability |
| **Limitation** | Ignores *when* churn happens; treats a customer churning at day 10 the same as day 300 |
| **Censoring** | No built-in handling; must drop or guess labels for incomplete cases |
| **Loss function** | Cross-entropy, log loss |

### Survival Analysis

| Aspect | Survival Analysis |
|--------|-------------------|
| **Target** | A pair: (duration, event indicator) |
| **Output** | S(t) — a full survival curve giving P(active) at *every* time point |
| **Advantage** | Captures *when*, not just *if*; distinguishes early vs late churners |
| **Censoring** | Handled natively in the likelihood function; no data is wasted |
| **Loss function** | Partial likelihood (Cox), negative log-likelihood with censoring adjustments |

### How Training Differs

In standard classification, the model sees `(features, label)` pairs and learns to minimize prediction error on the label.

In survival analysis, the model sees `(features, duration, event)` triples:

```
Customer A: features=[...], duration=120 days, event=1 (churned)
Customer B: features=[...], duration=85 days,  event=0 (censored — still active at day 85)
```

The training algorithm uses **partial likelihood** — it learns from the *ordering* of events. At each time point where a churn occurs, the model asks: "Among all customers who were still active at this moment, did the one who churned have the highest predicted risk?" Censored customers contribute to the "still active" risk set until their censoring time, then drop out.

This means **every customer contributes information**, even those who never churned during the study.

---

## 4. The Target Variable — What Are We Predicting?

### The (Duration, Event) Pair

Unlike standard ML where `y` is a single column, survival analysis requires two columns:

| Column | Name | Meaning |
|--------|------|---------|
| `duration` | Survival time | Days from first purchase to churn (or last known active date) |
| `event` | Event indicator | 1 = churned (event observed), 0 = censored (still active or data ended) |

**Examples from this project:**

| Customer | Duration | Event | Interpretation |
|----------|----------|-------|----------------|
| C-12345 | 180 days | 1 | Churned after 180 days |
| C-67890 | 400 days | 0 | Still active at day 400 (censored) |
| C-11111 | 45 days | 1 | Churned quickly (early churner) |
| C-99999 | 15 days | 0 | Only 15 days of data — too early to tell (censored) |

### What the Model Outputs

Instead of a single number, survival models output a **survival function S(t)** — a curve that gives the probability of remaining active at any time `t`:

```
Customer C-12345:
  S(30 days)  = 0.92   →  92% chance of being active at day 30
  S(90 days)  = 0.71   →  71% chance of being active at day 90
  S(180 days) = 0.35   →  35% chance of being active at day 180
  S(365 days) = 0.08   →   8% chance of being active at day 365
```

This curve is **personalized** — each customer gets their own survival trajectory based on their features.

---

## 5. Key Functions

### Survival Function — S(t)

The probability that a customer survives (remains active) beyond time `t`.

- S(0) = 1.0 (everyone starts active)
- S(t) decreases over time (more customers churn)
- S(t) never increases (once churned, you cannot "un-churn" in classical survival analysis)

**Interpretation:** S(90) = 0.65 means "this customer has a 65% probability of still being active at day 90."

### Hazard Function — h(t)

The instantaneous risk of the event occurring at time `t`, given survival up to that point. Think of it as the "danger rate" at each moment.

- **High hazard at early times:** Customer is at high risk of churning soon after first purchase
- **High hazard at later times:** Customer is at risk of churning after a long period of activity

The hazard function is the engine behind Cox models — they estimate how features shift a baseline hazard up or down.

### Cumulative Hazard Function — H(t)

The accumulated risk over time: H(t) = integral of h(t) from 0 to t.

Related to survival by: **S(t) = exp(-H(t))**

Used in this project for extrapolating median survival times when the observed survival curve does not drop below 0.5 within the data window.

### Kaplan-Meier Estimator

A non-parametric method for estimating the survival function from observed data. It makes no assumptions about the shape of S(t) — it simply calculates the proportion of customers surviving past each observed event time.

**Formula (conceptual):** At each time point where a churn occurs, multiply the previous survival probability by (1 - d/n), where d = number of churns and n = number still at risk.

In this project, the Kaplan-Meier curve serves two purposes:
1. **Visualization:** Comparing survival across customer segments (Gold vs Silver vs Bronze)
2. **One-timer scoring:** One-time buyers who cannot be scored by the Cox model receive the population-level KM survival estimate

---

## 6. Models Used in This Project

### Cox Proportional Hazards (Cox PH)

The most widely used survival model. Assumes each feature multiplies the baseline hazard by a constant factor (the **proportional hazards assumption**).

```
h(t | X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)
```

- `h₀(t)` = baseline hazard (shared across all customers)
- `exp(βᵢ)` = **hazard ratio** — how much feature Xᵢ multiplies the risk
  - Hazard ratio > 1: feature increases churn risk
  - Hazard ratio < 1: feature decreases churn risk
  - Hazard ratio = 1: no effect

**Strengths:** Interpretable coefficients, well-understood statistical properties, produces full survival curves.

**Limitation:** Assumes proportional hazards (the effect of each feature is constant over time).

### CoxNet (Elastic Net Cox)

Cox PH with **regularization** — a penalty that shrinks less important coefficients toward zero. Combines L1 (lasso, feature selection) and L2 (ridge, coefficient shrinkage). Useful when features are correlated or numerous.

### Random Survival Forest (RSF)

The survival analysis equivalent of a Random Forest. Builds many decision trees, each splitting on the feature/threshold that maximizes the difference in survival between the resulting groups. Averages predictions across all trees.

**Strengths:** Captures non-linear relationships and feature interactions without assumptions. Often achieves top performance.

### Gradient Boosting Survival Analysis

Sequentially builds weak survival models (typically small trees), where each new model corrects the errors of the previous ensemble. Similar in spirit to XGBoost but designed for censored outcomes.

### XGBoost Accelerated Failure Time (AFT)

An **accelerated failure time** model — instead of modeling the hazard, it models the log of survival time directly:

```
log(T) = β₀ + β₁X₁ + ... + βₚXₚ + σε
```

Features "accelerate" or "decelerate" the time to event. A coefficient > 0 means the feature extends survival time.

**Difference from Cox:** Cox models the *risk* at each moment; AFT models the *time itself*. They answer the same question from different angles.

---

## 7. Evaluation Metrics

Survival models cannot be evaluated with standard classification metrics (accuracy, AUC-ROC) because the target is not a binary label — it is a time-to-event with censoring. Special metrics are required.

### Concordance Index (C-index)

**What it measures:** Discrimination — can the model correctly rank customers by their risk?

**How it works:** Take all pairs of customers where we can determine who churned first. For each pair, check if the model assigned higher risk to the customer who actually churned sooner. The C-index is the fraction of pairs the model ranked correctly.

```
C-index = (# concordant pairs) / (# comparable pairs)
```

- **C-index = 1.0:** Perfect ranking — every higher-risk customer churned before every lower-risk customer
- **C-index = 0.5:** Random ranking — no better than flipping a coin
- **C-index < 0.5:** Worse than random (model is inverted)

**Practical interpretation:**
- 0.50–0.60: Poor discrimination
- 0.60–0.70: Moderate
- 0.70–0.80: Good
- 0.80–0.90: Very good
- 0.90+: Excellent

### IPCW C-index (Inverse Probability of Censoring Weighted)

**Why plain C-index is not enough:** The standard C-index ignores the fact that censored customers provide incomplete information. If many customers are censored, the standard C-index can be biased because we are only comparing the "easy" cases (customers who churned during observation).

**What IPCW does:** It weights each pair by the inverse probability of being censored at that time. Customers who are harder to observe (more likely to be censored) get higher weight, correcting the bias.

**Practical difference:** IPCW C-index is a more honest estimate of how well the model discriminates, especially when censoring rates are high. In this project, the IPCW C-index is the primary ranking metric.

### Time-Dependent AUC (TD-AUC)

**What it measures:** How well the model discriminates *at a specific time point*.

The standard C-index gives one number for all time. But a model might be excellent at predicting 30-day churn and poor at predicting 1-year churn (or vice versa). TD-AUC evaluates performance at each time horizon separately.

**How it works:** At each time point `t`, classify customers as:
- **Cases:** Customers who experienced the event by time `t`
- **Controls:** Customers who survived past time `t`

Then compute an AUC (area under the ROC curve) using the model's risk score at that time point.

**Mean TD-AUC** averages across multiple time points (e.g., 30, 60, 90, ..., 360 days) to give a single summary of time-varying discrimination.

**Interpretation:** Same as standard AUC:
- 1.0 = perfect separation at every time point
- 0.5 = no discrimination at any time point

### Brier Score (Time-Dependent)

**What it measures:** Calibration — are the predicted probabilities accurate?

Unlike C-index (which only cares about *ranking*), the Brier score checks whether a predicted survival probability of 0.70 actually corresponds to 70% of similar customers surviving.

**Formula (conceptual):**

```
BS(t) = average of (S(t|Xᵢ) - Iᵢ(t))²
```

Where `S(t|Xᵢ)` is the predicted survival probability and `Iᵢ(t)` is the actual outcome (1 if alive at `t`, 0 if not). Censored observations are handled via IPCW weighting.

- **BS = 0:** Perfect calibration — predicted probabilities exactly match reality
- **BS = 0.25:** Worst possible (equivalent to always predicting 0.5)
- **Lower is better**

### Integrated Brier Score (IBS)

**What it measures:** Overall calibration across all time points.

The Brier score is time-specific (one value per time point). The IBS integrates (averages) the Brier score over the entire follow-up period to give a single summary number.

```
IBS = (1 / T_max) × ∫ BS(t) dt
```

**Interpretation:**
- IBS close to 0: Excellent calibration across all time points
- IBS > 0.1: Calibration is degrading at some time horizons
- IBS > 0.2: Poor calibration — predicted probabilities are unreliable

**Why it matters:** A model can have a high C-index (good ranking) but a poor IBS (bad calibration). You want both — the model should rank customers correctly AND give accurate probability estimates.

**In this project's results (after removing leaky features — see Section 8):**

| Model | C-index (IPCW) | IBS |
|-------|----------------|-----|
| CoxNet | 0.894 | 0.049 |
| Cox PH | 0.891 | 0.054 |
| Gradient Boosting | 0.888 | 0.138 |
| Random Survival Forest | 0.887 | 0.048 |
| XGBoost AFT | 0.824 | 0.171 |

CoxNet, Cox PH, and RSF have both strong discrimination (C-index ~0.89) and good calibration (IBS < 0.06). Gradient Boosting and XGBoost AFT rank customers reasonably but their probability estimates are less accurate.

---

## 8. How to Judge If a Model Is Trustworthy

A trustworthy survival model should satisfy multiple criteria. No single metric tells the whole story.

### Checklist for Model Trust

| Criterion | What to Check | This Project |
|-----------|---------------|--------------|
| **Discrimination** | C-index (IPCW) > 0.70 | CoxNet: 0.894 |
| **Time-varying discrimination** | Mean TD-AUC > 0.80 | CoxNet: 0.970 |
| **Calibration** | IBS < 0.10 | RSF: 0.048 |
| **No data leakage** | Features use only past information; no forward-looking variables or churn proxies | Behavioral features only — BTYD predictions (p_alive, CLV) excluded |
| **Temporal validation** | Train on earlier data, test on later data | Strict temporal splits in both stages |
| **Censoring handled** | Censored observations used, not dropped or mislabeled | 90-day observation window; IPCW-weighted metrics |
| **Segment consistency** | Model works across different customer groups | Kaplan-Meier curves validated per segment |
| **Predictions are monotonic** | Higher predicted risk corresponds to shorter actual survival | Verified in scorecard analysis |

### Red Flags to Watch For

1. **C-index near 1.0 on test data** — Suspiciously perfect. Likely data leakage (features that encode the outcome).

2. **High C-index but high IBS** — Model ranks well but probabilities are miscalibrated. Risky if you are using the probabilities for decisions (e.g., "target customers with S(90) < 0.30").

3. **Large gap between train and test performance** — Overfitting. The model memorized training patterns that do not generalize.

4. **Features that should not exist** — Recency and frequency should not include the churn period itself. Features should only use information available at the prediction point.

5. **Derived churn scores used as features** — Using outputs from a churn/alive model (e.g., `p_alive` from BG/NBD) as input to a survival model creates a circular dependency. The feature directly encodes the target. In this project, `p_alive`, `expected_txns_6m`, and `clv_6m` were initially included and inflated C-index by ~3–5 points. Removing them produced honest, lower scores that reflect genuine predictive signal from behavioral features alone.

6. **One-timers mixed with repeat customers** — Customers with zero repeat purchases have duration=0, creating artificial mass points that distort model estimates. A two-stage approach (this project) is the correct solution.

### What "Good" Looks Like

For customer churn survival analysis with real-world data:

- **C-index 0.70–0.80:** Solid model. Customer ranking is meaningful.
- **C-index 0.80–0.90:** Strong model. Personalized interventions based on risk scores will be effective. (This project achieves ~0.89 with honest, leak-free features.)
- **C-index 0.90+:** Excellent — but verify there is no leakage. Scores above 0.93 on customer churn data warrant scrutiny.

- **IBS < 0.05:** Calibration is excellent. Predicted probabilities can be trusted at face value. (Cox PH and RSF achieve this in this project.)
- **IBS 0.05–0.10:** Good calibration. Probabilities are directionally correct.
- **IBS > 0.10:** Use risk rankings (C-index) but be cautious about interpreting raw probability values.

---

## 9. Glossary — Quick Reference

| Term | Definition |
|------|-----------|
| **AFT (Accelerated Failure Time)** | A survival model that directly models log(survival time) as a linear function of features. Features "speed up" or "slow down" the time to event. |
| **Baseline Hazard h₀(t)** | The hazard function when all feature values are zero. In Cox models, individual hazards are this baseline multiplied by exp(βX). |
| **BG/NBD Model** | Beta-Geometric/Negative Binomial Distribution — a probabilistic model for predicting customer purchase frequency and alive probability. |
| **Brier Score** | A calibration metric measuring the squared difference between predicted survival probabilities and actual outcomes at a specific time. Lower is better. |
| **BTYD (Buy Till You Die)** | A family of probabilistic models (BG/NBD, Pareto/NBD) for non-contractual customer behavior. Models when customers are "alive" vs "dead." |
| **C-index (Concordance Index)** | The proportion of customer pairs where the model correctly identifies who churns first. Measures ranking accuracy. Range: 0.5 (random) to 1.0 (perfect). |
| **Censoring** | When the true survival time is unknown because the study ended or the customer was lost to follow-up. The customer was observed to survive *at least* this long. |
| **Churn Window** | The number of inactive days after which a customer is declared churned. In this project: 45 days. |
| **CLV (Customer Lifetime Value)** | The total predicted revenue a customer will generate over their relationship with the business. |
| **Cox PH (Cox Proportional Hazards)** | A semi-parametric survival model that assumes features multiply the baseline hazard by constant factors (hazard ratios). |
| **CoxNet** | Cox PH with elastic net regularization (L1 + L2 penalty) to handle correlated features and perform feature selection. |
| **Duration** | Synonym for survival time — the time from entry (first purchase) to event (churn) or censoring. |
| **Event Indicator** | Binary variable: 1 if the event (churn) was observed, 0 if censored. |
| **Feature Leakage** | When a model input encodes information about the target variable, inflating apparent performance. Example: using `p_alive` (a churn probability estimate) as a feature to predict churn is circular — the model appears accurate but has not learned anything actionable. |
| **Gamma-Gamma Model** | A probabilistic model for predicting the monetary value of future transactions, conditional on the customer being alive. |
| **Gradient Boosting Survival** | An ensemble method that sequentially builds weak models to correct previous errors, adapted for censored survival outcomes. |
| **Hazard Function h(t)** | The instantaneous rate of the event occurring at time t, conditional on surviving to time t. Higher hazard = higher immediate risk. |
| **Hazard Ratio (HR)** | exp(β) from a Cox model. HR > 1 means the feature increases churn risk; HR < 1 means it decreases risk. |
| **IBS (Integrated Brier Score)** | The Brier score averaged over all time points. A single number summarizing calibration quality. Lower is better; < 0.05 is excellent. |
| **IPCW (Inverse Probability of Censoring Weighting)** | A correction technique that up-weights observations that are less likely to be observed (more likely censored), reducing bias in evaluation metrics. |
| **Kaplan-Meier (KM) Estimator** | A non-parametric estimator of the survival function. Makes no distributional assumptions. Plotted as a step function. |
| **Median Survival Time** | The time at which S(t) = 0.5 — the point where half the population has experienced the event. |
| **One-Timer** | A customer with exactly one purchase who never returns. Handled separately via Stage 1 classification in this project. |
| **Partial Likelihood** | The likelihood function used by Cox models. It considers only the *order* of events, not exact event times, and naturally accounts for censored observations. |
| **PCA (Principal Component Analysis)** | Dimensionality reduction technique used in this project to reduce correlated behavioral features before clustering. |
| **Proportional Hazards Assumption** | The assumption that hazard ratios are constant over time. If violated, time-varying coefficients or stratification may be needed. |
| **Random Survival Forest (RSF)** | An ensemble of survival trees. Each tree splits data to maximize survival difference between groups. No proportional hazards assumption required. |
| **RFM (Recency, Frequency, Monetary)** | A customer segmentation framework based on how recently, how often, and how much a customer has purchased. |
| **Right Censoring** | The most common type of censoring: the event has not yet occurred by the end of the study. The true survival time is somewhere beyond the observed time. |
| **Risk Score** | A model's predicted relative risk for a customer. Higher scores indicate higher churn risk. Used for ranking, not as a probability. |
| **Risk Set** | At any time t, the set of all customers who have not yet experienced the event or been censored. This is the denominator in survival calculations. |
| **S(t) — Survival Function** | P(T > t) — the probability of surviving beyond time t. Starts at 1.0 and decreases over time. |
| **Survival Curve** | A plot of S(t) over time. A steep initial drop indicates high early churn; a long flat tail indicates a loyal subgroup. |
| **TD-AUC (Time-Dependent AUC)** | AUC computed at a specific time point, measuring how well the model separates those who experienced the event by that time from those who did not. |
| **Temporal Split** | A train/test split based on time (train on earlier data, test on later data) rather than random splitting. Prevents temporal leakage. |
| **Two-Stage Model** | Stage 1 classifies one-timers vs repeaters; Stage 2 applies survival analysis only to predicted (or known) repeat customers. |

---

*This guide accompanies the Customer Churn Survival Analysis project. For implementation details, see `customer-survival-analysis.ipynb` and the project `README.md`.*
