# Lessons learned — customer survival analysis

A practical checklist distilled from two rounds of code review on the
Online Retail II and Dunnhumby "Complete Journey" survival pipelines.
Every item below corresponds to a mistake that was actually made in this
repo and caught in review. Read before writing your first survival
feature; come back to it before you report results.

---

## 1. Feature leakage — the sneakiest form kills you

The textbook kind of leakage ("I accidentally used the label as a feature")
is rare. The kind that ruined a 0.99 C-index on Dunnhumby was subtler —
**features that share their denominator with the survival target**.

### Specific traps we hit

- `recency_ratio = duration / T_days` → `duration` **is** the survival
  target. You are regressing `y` on `y / constant`. C-index: 0.99.
- `spend_last_90d` when the 90-day window ends at the study end → a
  household with `spend_last_90d == 0` has by definition been inactive
  for the last 90 days, which (with a 14-day churn window) is certain
  churn.
- `spend_trend = recent_avg / older_avg` when both windows end at the
  study end → same problem, dressed up.

### Mitigation

Audit rule: **for every feature, ask "would I have known this value at
the prediction time?"**. If the feature references the study end, the
answer is no. Use a **landmark** — a fixed time `t0` in the past — and
build features strictly from data on or before `t0`.

### Automated check

Compute `|corr(feature, event_observed)|` and `|corr(feature, event_time)|`
for every numeric feature. If anything tops 0.7 on an inactivity-based
churn target, you have a leak. See
[`dunnhumby/tests/test_leakage_and_smoke.py`](dunnhumby/tests/test_leakage_and_smoke.py)
for the assertion.

---

## 2. Event time and censoring — the most common conceptual error

For **inactivity-based** churn (the most common kind), a household is
declared churned when they have been silent for longer than the window.
That subtlety has two consequences most implementations miss.

| Observation | Wrong | Right |
|---|---|---|
| Churner event time | `last_purchase − first_purchase` | `last_purchase + window − first_purchase` |
| Censored (still active) time | `last_purchase − first_purchase` | `study_end − first_purchase` |

The wrong version under-credits both groups: churners "survived" longer
than you are giving them credit for (until the window was crossed), and
active customers "survived" to the study end, not just to their most
recent basket.

**What this breaks:** the time axis of every Kaplan-Meier curve, the
proportional-hazards denominator in Cox, the AFT target in XGBoost.
Everything downstream is wrong by up to `window` days.

### Even better: the landmark formulation

If you pick a landmark `t0` and measure event time from `t0` forward, the
formulas become:

- Churner: `event_time = last_purchase + window − t0`
- Censored: `event_time = study_end − t0`

Survival-function outputs `S(Δ)` are then automatically conditional on
"alive at `t0`" — which is exactly what you want for a scorecard ("given
this household was active on DAY 500, what's the probability they're
still active on DAY 530?").

---

## 3. The off-by-one on "alive at the landmark"

Given churn is **strictly greater than** `window` days of inactivity, a
household whose last pre-landmark basket is exactly at `t0 − window` is
**still alive**. Their inactivity at `t0` is `window`, which is not
strictly greater than `window`.

So the alive filter is `last_pre_t0 >= t0 - window`, **not** `>`. On the
Dunnhumby data this flipped 27 of 1,794 households in and out of cohort —
a real effect, a real bug.

```python
# Wrong
alive = last_pre[last_pre > t0 - window].index

# Right
alive = last_pre[last_pre >= t0 - window].index
```

---

## 4. Data discipline — split first, fit later, touch test once

Three distinct mistakes all share the same root cause: **population-level
information bleeds from test into training**.

| Mistake | Found in | Symptom |
|---|---|---|
| Fit BTYD (BG/NBD + Gamma-Gamma) on all households before the train/test split | `07_clv_prediction_benchmark.py` v1 | Population parameters `(r, α, a, b)` were tuned on test-set RFM too |
| XGBoost `eval_set=[(X_test, y_test)]` with `early_stopping_rounds` | Stage-1 notebook, Dunnhumby v1 | Number of rounds implicitly chosen on test, giving 0.01–0.02 AUC lift |
| Use test-set for best-alpha / best-n_estimators selection | CoxNet / GBSA first cut | Same as above |

### Rule of thumb

```
split first    → train / val / test
fit always on  → train (for final models: train ∪ val)
tune on        → val
report once on → test
```

For BTYD specifically, that means `lifetimes.BetaGeoFitter.fit()` sees
only the training households' calibration RFM. Scoring is applied to
test households' own calibration RFM with the train-fit parameters.

### Smoke test

Don't trust a test name. Write an assertion that actually checks the
invariant you care about:

```python
train_hh = set(assignment.loc[assignment["split"] == "train", "household_key"])
val_hh   = set(assignment.loc[assignment["split"] == "val",   "household_key"])
test_hh  = set(assignment.loc[assignment["split"] == "test",  "household_key"])

assert len(train_hh & val_hh)  == 0
assert len(train_hh & test_hh) == 0
assert len(val_hh   & test_hh) == 0
```

A test named `test_clv_split_disjoint` that only asserts "test set size
is ~25%" would pass if train and test fully overlapped. That exact
mistake slipped through round 1.

---

## 5. Scorecards: conditional vs unconditional survival

`S(30d)` means different things depending on anchor:

- **Unconditional from first purchase:** `P(still active 30 days after they
  joined)`. Useless for a live scorecard because most customers are
  already far past day 30.
- **Conditional from today (= landmark):** `P(still active 30 days from
  today | alive today)`. This is the number an ops team cares about.

For a landmark model where `event_time` is measured from `t0`, the
conditional probability is just `S(Δ)` read off the model's survival
function at `t = Δ`. **Do not compute** `S(current_duration + Δ)` and
interpolate — that will silently extrapolate past the modelled time
range. We hit this: 57% of households had identical `S(90d)` and
`S(180d)` values because `np.interp` was clamping both to the last
training time point.

### Cap `Δ` at the max modelled time

```python
eval_times = np.linspace(1, max_train_event_time, 50)
capped_deltas = [min(delta, eval_times.max()) for delta in (30, 60, 90)]
```

If you have to extrapolate, at least label the column so nobody ships a
slide deck with it.

---

## 6. Fill gaps explicitly

If your scorecard contract says "every customer gets an `S_30d` value",
then every customer needs a value — not just the landmark cohort.

The right fallback hierarchy for customers outside the landmark cohort
(long-inactive, or joined too recently):

1. **Kaplan-Meier baseline** on the training cohort — a single
   population-average curve. Cheap, honest, monotone.
2. **BTYD `p_alive`** — useful as an independent cross-check, but not
   directly comparable to a Cox S(t) because the time axes differ.
3. **NaN** — only if you genuinely don't know.

Whatever you pick, **track provenance** in a column so consumers can
filter:

```python
scorecard["s_source"] = np.where(in_cohort, "cox_landmark", "km_baseline")
```

---

## 7. Refit the chosen model on all training data before deploying

It is standard to use a train/val split for hyperparameter selection and
early stopping. But the model you then export — the one that scores
everyone for the scorecard — should be refit on `train ∪ val` using the
validated hyperparameters. Skipping this wastes 15–25% of training data.

**Watch the latent-path version of this bug.** On this repo the code was
correct for CoxPH / CoxNet / RSF / GBSA but silently wrong for XGBoost
AFT — if XGBoost ever won, `bst` (fit on train only) would be reused.
It didn't win, so the bug never fired. But *"this doesn't matter today"*
is different from *"this is correct"*.

---

## 8. Apples-to-apples before/after comparisons

If you are reporting "the fix improved AUC from 0.620 to 0.628", change
**exactly one thing**. Otherwise the delta is unattributable.

In our case, a "fixed stage-1 script" quietly dropped the product-keyword
features, the geography dummies, the description diversity and spend
concentration, and used a three-way date split instead of the notebook's
single cutoff. The reported AUC delta mixed the early-stopping fix with a
different feature set. The honest isolated fix produced a **smaller**
delta (0.6204 → 0.6097) — in the other direction than originally
reported.

When you publish a before/after:

1. Keep the feature set and split identical to the original.
2. Change the single methodology choice you are isolating.
3. If you also want to show feature-set improvements, show them as a
   *separate* comparison.

---

## 9. Choose the churn window empirically, not by convention

Online retail literature uses 30 / 45 / 60-day windows. That is fine if
your median inter-purchase gap is ~50 days (as in the Online Retail II
dataset). For a grocery panel where the median gap is **2 days** and p95
is **20 days**, a 45-day window is three months of "still active" by
grocery standards — you will miss most churn events.

Calibrate from the data:

1. Compute the CDF of observed inter-purchase gaps.
2. Apply Kneedle elbow detection + max-curvature as a sanity check.
3. Validate label stability across a grid of candidate windows (≥95%
   agreement between adjacent choices is a good sign).

Dunnhumby: Kneedle → 16 days → chose **14**. Online Retail II: Kneedle →
43 days → chose **45**. Both are defensible; both are data-driven; both
had to be derived, not assumed.

---

## 10. One-timers don't belong in survival models

Households with a single purchase have `duration = 0`, which puts a mass
point at the origin of every hazard curve and pulls the baseline hazard
toward infinity. Standard practice:

- **Exclude them from the survival model.** Model them separately as a
  binary conversion classifier (or an early-dropout classifier if your
  data is mostly repeaters, as with Dunnhumby where only 3 of 2,500
  households are one-timers — essentially no positive class for
  classical "will they come back?" framing).
- **Handle them in the scorecard via a population KM curve** or a
  "one-timer" tier, not the fitted Cox model.

This is cell-30 of the Online Retail notebook's cleanest design choice
and it carries over.

---

## 11. Exclude forward-looking BTYD outputs from survival features

`p_alive`, `expected_txns_6m`, and `clv_6m` from the BG/NBD + Gamma-Gamma
models directly encode "are they still alive?". Adding them as features
to a survival model predicting churn creates circular reasoning —
you are using an estimate of the target as a predictor of the target.

They are still useful:

- As **benchmarks** for CLV prediction.
- As **fallback scores** for out-of-cohort customers in the scorecard.
- For **exploratory segmentation**.

They are not useful as survival-model features. Keep them in a separate
parquet and join them only at scorecard time.

---

## 12. Documentation drift hides bugs

A short selection of things that were written down in this repo but not
true:

- Docstring said "Population KM baseline is used to fill in customers
  outside the landmark cohort." Implementation merged landmark outputs
  with a left join and left 4,609 customers with `NaN`.
- Docstring said "refit the best model on train+val before scoring
  everyone". For four of five branches, true. For the fifth (XGBoost
  AFT), false — it reused the train-only model.
- Docstring said "split is `first_day <= 365`". Code used a 60th
  percentile split.
- README said "verified / no leakage". Actually had a 0.99 correlation
  feature.

The prophylaxis: write docstrings **after** the code stabilises, not
before. If you catch yourself writing "refactored to use X" in a
commit message, grep the code to make sure X is actually used.

---

## 13. Test-suite discipline

A few anti-patterns we hit:

- **Size checks masquerading as disjointness checks.** `len(test_df) ≈
  25% of total` passes even when `train_hh == test_hh`.
- **Tests that don't regression-protect the main claim.** If the README
  says "we prevent CLV benchmark leakage", the test should assert
  disjointness, not file size.
- **Tests that silently skip.** If your fixture file doesn't exist, don't
  just skip — fail loudly.

### Minimum test set for a survival pipeline

1. No feature has `|corr|` > 0.7 against the event or event time.
2. `S(t)` is non-increasing per customer (monotonicity).
3. Event/censoring time formulas are self-consistent.
4. Splits are **pairwise disjoint**, not just sized.
5. Scorecard has the promised coverage.

See [`dunnhumby/tests/test_leakage_and_smoke.py`](dunnhumby/tests/test_leakage_and_smoke.py) — 18 assertions, pass or the build fails.

---

## 14. Read the data before you read the tutorials

Grocery and e-commerce differ in every variable that matters for
survival: inter-purchase cadence (2 days vs 50), basket size, household
vs individual identity, seasonality, one-timer rate, promo response.
A textbook 45-day churn window transplanted from online retail to
grocery is not a defensible starting point — it's a bug.

Dataset-specific priors to check before you start:

- Distribution of inter-purchase gaps (pick the churn window).
- Distribution of first-basket dates (decide eligibility for
  temporal splits).
- Ratio of one-timers (decide if a conversion model makes sense).
- Demographic coverage (decide on `Unknown` imputation vs sub-model).
- Seasonality (decide if the panel needs time-weighted fits).

---

## Appendix — checklist before you publish results

- [ ] Event time = `last + window − t0`; censoring = `study_end − t0`.
- [ ] Features built from data **strictly before** the landmark.
- [ ] Max `|corr|` of any feature with target < 0.7.
- [ ] Train / val / test split done BEFORE any model sees data.
- [ ] BTYD / population models fit on training households only.
- [ ] Early stopping uses validation, never test.
- [ ] Hyperparameters (alpha, n_estimators) tuned on validation, never test.
- [ ] Final scorecard model refit on train ∪ val.
- [ ] `S(Δ)` values are conditional forward-from-landmark, not
      unconditional-from-first-purchase.
- [ ] No extrapolation past `max(event_time_train)` — cap Δ.
- [ ] Scorecard coverage is 100% with explicit provenance.
- [ ] Docstrings match code. `git diff` them before shipping.
- [ ] Automated leakage + smoke tests green.
- [ ] Before/after AUC deltas isolate a single change.
