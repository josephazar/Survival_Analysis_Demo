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
9. [Plain-English Cheat Sheet for Notebook Review Terms](#9-plain-english-cheat-sheet-for-notebook-review-terms)
10. [Glossary — Quick Reference](#10-glossary--quick-reference)

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
| `duration` | Survival time | Time from a reference point until the event (or until censoring) |
| `event` | Event indicator | 1 = churned (event observed), 0 = censored (still active or data ended) |

### Watch-out — event and censoring times for inactivity-based churn

If the event is defined as **"inactive for more than `W` days"**, the natural intuition ("duration = last_purchase − first_purchase") is **wrong** on both sides:

| Customer type | Wrong definition | Correct definition |
|---|---|---|
| Churner | `duration = last_purchase − first_purchase` | `duration = last_purchase + W − first_purchase` (event is declared when inactivity *crosses* W) |
| Censored (still active at end) | `duration = last_purchase − first_purchase` | `duration = study_end − first_purchase` (censored at the last time we could have observed the event) |

The wrong version under-credits both groups — churners "survived" longer than you admit (until the window was crossed), active customers "survived" to the study end, not just to their most recent basket. Everything downstream — Kaplan-Meier, Cox, AFT — is shifted by up to `W` days.

See [LESSONS_LEARNED.md §2](LESSONS_LEARNED.md#2-event-time-and-censoring--the-most-common-conceptual-error) for a deeper discussion and the specific bug that prompted this callout.

### Landmark design — clean features, clean time axis

Rather than measuring duration from each customer's own first purchase, many professional pipelines pick a **fixed calendar landmark `t0`** in the past (e.g. DAY 500 of a 711-day panel, or `2011-06-01` for Online Retail II) and:

- Keep only customers **alive at `t0`** (had a purchase in `(t0 − W, t0]`, with `>=` on the lower bound).
- Build features from data **on or before `t0`** — guaranteed no leakage from the prediction window.
- Measure event time **from `t0`** — the churn event for a customer with last basket `L` is declared at `L + W`, giving `event_time = L + W − t0`; for censored customers it is `study_end − t0`.

A big win of this formulation: the survival-function outputs `S(Δ)` are automatically **conditional on "alive at `t0`"**, which is exactly what a scorecard needs ("given this customer is still active today, what's the probability they're still active in 30 days?").

**Examples from this project (Dunnhumby, landmark `t0` = DAY 500, `W` = 14d):**

| Household | `last_overall` | Event | `event_time` (days from t0) | Interpretation |
|---|---|---|---|---|
| H-0001 | DAY 600 | 1 | `600 + 14 − 500 = 114` | Churned mid-way through follow-up |
| H-0002 | DAY 705 | 0 | `711 − 500 = 211` | Still active at study end (censored) |
| H-0003 | DAY 495 | 1 | `495 + 14 − 500 = 9` | Churned almost immediately after `t0` |

### Age-conditioned scoring — when your model clock starts at signup

Sometimes the model's time origin is still **signup** or **first purchase**,
but the scorecard is for customers who are already alive today. In that case,
reading `S(30)` from the curve does **not** mean "probability active 30 days
from today." It means "probability active 30 days after signup."

For a customer who is `a` days old today, the forward survival probability for
the next `Δ` days is:

```
S_from_today(Δ | age=a, x) = S(a + Δ | x) / S(a | x)
```

And the forward churn risk is:

```
Risk_from_today(Δ | age=a, x) = 1 - S(a + Δ | x) / S(a | x)
```

The denominator is the important part: it conditions on what you already know,
namely that the customer survived to today's scoring date. In `lifelines`, this
can be implemented directly with `predict_survival_function(...,
conditional_after=[age])`, or manually with the ratio above.

A vertical "today" line on a signup-time survival curve is only a visualization.
It becomes a true planning probability only after this age-conditioning step.

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

**In this project's results — Online Retail II, landmark design (after fixing event-time misalignment and retrospective features — see Section 8):**

| Model | C-index (IPCW) | Mean TD-AUC | IBS |
|-------|----------------|-------------|-----|
| RSF | 0.758 | 0.822 | 0.132 |
| CoxPH | 0.750 | 0.814 | 0.137 |
| GBSA | 0.747 | 0.808 | 0.137 |
| CoxNet | 0.746 | 0.811 | 0.139 |

**Dunnhumby grocery panel — landmark design, 14-day churn window:**

| Model | C-index (IPCW) | Mean TD-AUC | IBS |
|-------|----------------|-------------|-----|
| CoxPH | 0.738 | 0.767 | 0.040 |
| CoxNet (val-selected winner) | 0.733 | 0.771 | 0.042 |
| XGBoost AFT | 0.728 | 0.764 | 0.046 |
| GBSA | 0.722 | 0.745 | 0.043 |
| RSF | 0.719 | 0.751 | 0.040 |

(The winner is chosen on the **validation** fold — CoxNet — and reported at
its test numbers. CoxPH edging it on test is exactly the kind of post-hoc
selection the protocol forbids.)

Both datasets converge on C-indices in the **0.72–0.76** range for honest landmark evaluations. Earlier reports of 0.89–0.99 on these datasets reflected either retrospective features (Online Retail II) or direct target leakage (Dunnhumby's `recency_ratio = duration / T_days`). The lower numbers are the right numbers.

---

## 8. How to Judge If a Model Is Trustworthy

A trustworthy survival model should satisfy multiple criteria. No single metric tells the whole story.

### Checklist for Model Trust

| Criterion | What to Check | This Project |
|-----------|---------------|--------------|
| **Discrimination** | C-index (IPCW) > 0.70 | CoxNet (Dunnhumby, val-selected): 0.733; RSF (Online Retail landmark): 0.758 |
| **Time-varying discrimination** | Mean TD-AUC > 0.75 | 0.77 – 0.82 across both projects |
| **Calibration** | IBS < 0.10 for landmark models | Dunnhumby: 0.04; Online Retail: 0.13 (higher because the follow-up window is longer) |
| **No feature leakage** | Every numeric feature has `\|corr\|` < 0.7 with the event and event-time targets | Automated assertion in `dunnhumby/tests/test_leakage_and_smoke.py` |
| **Landmark design** | Features built strictly pre-`t0`, event time measured from `t0` forward | Both pipelines |
| **Proper event time** | `event_time = last + W − first` or `(last + W) − t0`; censoring at study end | Both pipelines; asserted in smoke tests |
| **Pairwise-disjoint splits** | train ∩ val = val ∩ test = train ∩ test = ∅ | Asserted in `test_clv_split_disjoint` |
| **No test peek** | Early stopping, hyper-tuning, Youden thresholds, and model selection all use validation, not test | Stage-1, Stage-5, Stage-6, Stage-7 all use val folds |
| **Censoring handled** | Censored observations used, not dropped or mislabeled | IPCW-weighted metrics |
| **Predictions monotone** | `S(t)` non-increasing per customer | Asserted |
| **Scorecard coverage** | Every scored customer has an S(Δ) with explicit provenance — or an explicit NaN where no honest estimate exists | `s_source` column (`cox_landmark` / `km_baseline` / `none_churned`) |

### Red Flags to Watch For

1. **C-index > 0.9 on customer churn data** — Suspiciously high. On this project, 0.993 was the result of a single feature (`recency_ratio = duration / T_days`) that embedded the survival target in the numerator. Audit every feature's correlation with the label before celebrating.

2. **Features anchored to the study end** — `spend_last_90d`, `freq_trend`, `days_since_last_at_end_of_data` etc. all implicitly encode whether the customer stopped shopping. Anchor these to a **landmark** instead.

3. **Event time = `last_purchase − first_purchase`** — If the event is "inactive > W days", the event time for churners should be `last_purchase + W − first_purchase`, not `last_purchase − first_purchase`. For censored customers it should be `study_end − first_purchase`. See Section 4.

4. **High C-index but high IBS** — Model ranks well but probabilities are miscalibrated. Risky if you are using the probabilities for decisions.

5. **Large gap between train and test performance** — Overfitting. The model memorized training patterns that do not generalize.

6. **BTYD scores (`p_alive`, `clv_6m`) used as survival features** — Circular: the feature directly estimates the target. Use BTYD as a CLV *benchmark* and as a scorecard *fallback*, not as a survival covariate.

7. **One-timers mixed with repeat customers** — Customers with zero repeat purchases have duration=0, creating artificial mass points that distort model estimates. Use a two-stage architecture or restrict the survival model to repeat customers.

8. **Test set used for early stopping** — `xgb.fit(..., eval_set=[(X_test, y_test)], early_stopping_rounds=30)` quietly selects `num_rounds` based on test performance. Carve a validation slice from train instead.

9. **Unconditional `S(Δ)` presented as forward risk** — `S(30)` from a model where time origin is "first purchase" is the probability of surviving the first 30 days of a customer's lifetime, not the probability of surviving the next 30 days from today. Use either a landmark model so `S(Δ)` is automatically conditional on "alive at landmark", or age-condition a signup-time model with `S(age + Δ) / S(age)`.

10. **Scorecard columns with silent `NaN`** — If your contract says every customer gets an `S_30d`, then every customer needs an `S_30d`. Use a population KM baseline to fill in out-of-cohort customers, and track provenance.

### What "Good" Looks Like

For customer churn survival analysis with real-world data:

- **C-index 0.70–0.80:** Solid model. Customer ranking is meaningful. This project's landmark designs land here (0.72–0.76 across both datasets).
- **C-index 0.80–0.90:** Strong model. Personalized interventions based on risk scores will be effective. Plausible on contractual / subscription data where the signal is cleaner.
- **C-index 0.90+:** On inactivity-based retail churn data, treat as a red flag until you have audited every feature. This project initially saw 0.99 and it was feature leakage.

- **IBS < 0.05:** Calibration is excellent. Predicted probabilities can be trusted at face value.
- **IBS 0.05–0.10:** Good calibration. Probabilities are directionally correct.
- **IBS > 0.10:** Use risk rankings (C-index) but be cautious about interpreting raw probability values.

---

## 9. Plain-English Cheat Sheet for Notebook Review Terms

This section translates the exact phrases used in the intro notebooks and review notes. Read it as a "what does this mean, and what do I do with it?" guide.

| Phrase | Plain English | How to use it | Watch out |
|---|---|---|---|
| **Numerical claim** | Any number written in markdown, a title, or a conclusion: "C-index = 0.72", "median = 8 days", "tier churn = 75%". | Every number in the story should be produced by a nearby code cell or artifact. If the notebook is rerun, the text should still be true. | Stale markdown is common after reruns. Treat hard-coded numbers as promises that need checking. |
| **Notebook output** | The printed tables, plots, and metrics saved inside a Jupyter notebook after code execution. | Use outputs to make the notebook readable on GitHub without forcing the reader to rerun everything. | Outputs can be old. A publish pass should execute notebooks top-to-bottom and check for error outputs. |
| **Markdown claim** | Explanatory text in a notebook, not code. | Use markdown to interpret results in business language: what changed, why it matters, and what decision follows. | Markdown can overclaim. If it says "the model is calibrated", there should be calibration evidence below it. |
| **Spot-check** | A quick targeted verification that important numbers in the prose match the code outputs. | Use it after edits to catch obvious drift: p-values, C-index, medians, tier counts, and risk estimates. | Spot-checking is not a full test suite. It complements, but does not replace, executing notebooks and running tests. |
| **KM shortcut example** | A teaching example that compares Kaplan-Meier against naive shortcuts like "drop censored rows" or "treat censored as non-events". | Use it to show why survival analysis is needed: censored customers still contain partial information. | Shortcuts often look reasonable but bias durations or event rates. |
| **Kaplan-Meier (KM)** | A simple step-by-step estimate of "what fraction are still event-free over time?" | Use KM for baseline survival curves, segment comparisons, and censoring-aware observed rates by risk tier. | KM is descriptive. It does not control for covariates unless you split into groups manually. |
| **Activation median / median survival time** | The time when the survival curve crosses 50%. For activation, it is the day when half the users have activated. | Use it as a simple business summary: "half of users activate by day 8." | If the curve never drops below 50%, the median is not observed. Do not invent it. |
| **Log-rank test** | A statistical test asking whether two or more survival curves are different. | Use it after plotting KM curves by segment to check whether the gap is larger than random noise. | It is not an effect size and not causal. Pair it with medians, survival probabilities, or business impact. |
| **p-value** | Under a "no real difference" assumption, the chance of seeing a result this extreme or more extreme. | Use it as a signal to look closer, especially in log-rank and PH diagnostics. | It is not "the probability the model is right." Tiny p-values can come from huge samples and trivial effects. |
| **Cox model / Cox PH** | A survival model that estimates how features multiply the current event risk. | Use it when you need interpretable drivers and a survival curve per customer. | It assumes proportional hazards. Check that assumption before quoting hazard ratios too confidently. |
| **C-index** | A ranking score: did the model assign higher risk to the customer who churned sooner? | Use it to judge whether risk ordering is useful for prioritization. Around 0.70-0.80 is solid for honest customer churn models. | It is not calibration. A model can rank well but give bad probabilities. A value above 0.90 is suspicious on inactivity-based churn unless carefully audited. |
| **PH test / proportional hazards test** | A diagnostic for whether a Cox feature's effect stays roughly constant over time. | Use it before treating hazard ratios as stable business drivers. A small p-value means "inspect this feature." | It is a screen, not a verdict. Multiple features mean multiple tests, so one small p-value can happen by chance. |
| **Leakage** | A feature accidentally knows the answer because it uses future information or encodes the target. | Audit every feature by asking: "Would I know this at prediction time?" If not, remove or rebuild it around a landmark. | Leakage can look like a genuine model improvement. Too-good C-index jumps are a major warning sign. |
| **C-index jump from leakage** | The model score improves sharply after adding a forbidden future-looking feature. | Use deliberate leakage examples as teaching tests: they show what a suspicious metric looks like. | A small jump can still be dangerous. Mild leakage can ship because it looks plausible. |
| **End-to-end project** | A complete modeling story: define cohort, split data, fit/tune model, evaluate, score customers, explain the decision. | Use it as the template for real work, not just a demo. The final output should be an action table or scorecard. | Do not optimize every step on the test set. Use train for fitting, validation for choices, test once for reporting. |
| **Tier table / risk tiers** | A scorecard summary that groups customers into action buckets such as Monitor, Watchlist, Immediate Intervention. | Use tiers to turn survival predictions into operational decisions and capacity planning. | Freeze cutoffs on train. Do not recompute percentiles separately on each scoring batch or the meaning of a tier will drift. |
| **Calibration table** | A table comparing predicted event risk with observed event rates at a horizon. | Use it to answer: "When the model says 30% risk, does about 30% happen?" | With censoring, use Kaplan-Meier or IPCW methods. Complete-case rates can be biased. |

### Minimal Workflow for Using These Terms

1. Start with **KM** to understand the raw time pattern and censoring.
2. Use **median survival / activation** only if the curve actually crosses 50%.
3. Use a **log-rank test** to compare raw segment curves, but report effect size too.
4. Fit **Cox PH** when you want interpretable feature effects and survival curves.
5. Check **PH tests** before quoting hazard ratios as stable drivers.
6. Evaluate ranking with **C-index**, then evaluate probability quality with calibration/Brier-style checks.
7. Hunt for **leakage** whenever metrics jump too high or features reference the study end.
8. Convert predictions into a **tier table** only after thresholds are chosen without touching test data.

## 10. Glossary — Quick Reference

| Term | Simple definition |
|------|-----------|
| **AFT (Accelerated Failure Time)** | A survival model that directly models log(survival time) as a linear function of features. Features "speed up" or "slow down" the time to event. |
| **Age-Conditioned Survival Scoring** | Scoring future survival for a customer who is already alive today. If the model clock starts at signup and the customer is age `a`, use `S(a + Δ) / S(a)`, not `S(Δ)`. |
| **Activation Median** | In an activation notebook, the time by which 50% of users have activated. Same idea as median survival time, but the event is activation instead of churn. |
| **Baseline Hazard h₀(t)** | The hazard function when all feature values are zero. In Cox models, individual hazards are this baseline multiplied by exp(βX). |
| **BG/NBD Model** | Beta-Geometric/Negative Binomial Distribution — a probabilistic model for predicting customer purchase frequency and alive probability. |
| **Brier Score** | A calibration metric measuring the squared difference between predicted survival probabilities and actual outcomes at a specific time. Lower is better. |
| **BTYD (Buy Till You Die)** | A family of probabilistic models (BG/NBD, Pareto/NBD) for non-contractual customer behavior. Models when customers are "alive" vs "dead." |
| **Calibration Table** | A table comparing predicted risk with observed risk at a fixed horizon. In survival analysis, observed risk should handle censoring, often with Kaplan-Meier. |
| **C-index (Concordance Index)** | The proportion of customer pairs where the model correctly identifies who churns first. Measures ranking accuracy. Range: 0.5 (random) to 1.0 (perfect). |
| **Censoring** | When the true survival time is unknown because the study ended or the customer was lost to follow-up. The customer was observed to survive *at least* this long. |
| **Churn Window** | The number of inactive days after which a customer is declared churned. In this project: 45 days. |
| **CLV (Customer Lifetime Value)** | The total predicted revenue a customer will generate over their relationship with the business. |
| **Conditional Survival** | The probability of surviving an additional period given survival up to a previous time. Formula: `P(T > s + t | T > s) = S(s + t) / S(s)`. |
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
| **KM Shortcut Example** | A teaching comparison showing why naive shortcuts, such as dropping censored rows, give biased answers compared with Kaplan-Meier. |
| **Log-rank Test** | A test for whether two or more survival curves differ. It compares observed vs expected events over time. It is evidence of a difference, not a business effect size or causal proof. |
| **Markdown Claim** | A number or interpretation written in notebook text. It should be backed by nearby code output and updated after reruns. |
| **Median Survival Time** | The time at which S(t) = 0.5 — the point where half the population has experienced the event. |
| **Notebook Output** | The saved printed tables, plots, and metrics from executed notebook cells. Outputs make notebooks readable on GitHub but must be refreshed before publishing. |
| **One-Timer** | A customer with exactly one purchase who never returns. Handled separately via Stage 1 classification in this project. |
| **Partial Likelihood** | The likelihood function used by Cox models. It considers only the *order* of events, not exact event times, and naturally accounts for censored observations. |
| **PCA (Principal Component Analysis)** | Dimensionality reduction technique used in this project to reduce correlated behavioral features before clustering. |
| **PH Test** | A proportional-hazards diagnostic. It checks whether a Cox feature's hazard ratio appears to change over time. A small p-value means inspect the feature. |
| **p-value** | A measure of how surprising the data would be if the null hypothesis were true. It is not the probability that the model or hypothesis is true. |
| **Proportional Hazards Assumption** | The assumption that hazard ratios are constant over time. If violated, time-varying coefficients or stratification may be needed. |
| **Random Survival Forest (RSF)** | An ensemble of survival trees. Each tree splits data to maximize survival difference between groups. No proportional hazards assumption required. |
| **RFM (Recency, Frequency, Monetary)** | A customer segmentation framework based on how recently, how often, and how much a customer has purchased. |
| **Right Censoring** | The most common type of censoring: the event has not yet occurred by the end of the study. The true survival time is somewhere beyond the observed time. |
| **Risk Score** | A model's predicted relative risk for a customer. Higher scores indicate higher churn risk. Used for ranking, not as a probability. |
| **Risk Set** | At any time t, the set of all customers who have not yet experienced the event or been censored. This is the denominator in survival calculations. |
| **Risk Tier / Tier Table** | An operational grouping of scored customers, such as Monitor or Immediate Intervention. Cutoffs should be chosen on train/validation data and then frozen for test or production scoring. |
| **S(t) — Survival Function** | P(T > t) — the probability of surviving beyond time t. Starts at 1.0 and decreases over time. |
| **Spot-check** | A targeted manual check that key markdown numbers match code outputs. Useful before publishing, but not a replacement for full notebook execution and tests. |
| **Survival Curve** | A plot of S(t) over time. A steep initial drop indicates high early churn; a long flat tail indicates a loyal subgroup. |
| **TD-AUC (Time-Dependent AUC)** | AUC computed at a specific time point, measuring how well the model separates those who experienced the event by that time from those who did not. |
| **Temporal Split** | A train/test split based on time (train on earlier data, test on later data) rather than random splitting. Prevents temporal leakage. |
| **Two-Stage Model** | Stage 1 classifies one-timers vs repeaters; Stage 2 applies survival analysis only to predicted (or known) repeat customers. |

---

*This guide accompanies the Customer Churn Survival Analysis project. For implementation details, see `customer-survival-analysis.ipynb` and the project `README.md`.*
