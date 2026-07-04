# Field guide — the traps in BTYD, Bayesian CLV, and Markov LTV modeling

Companion to the repo-level [`../LESSONS_LEARNED.md`](../LESSONS_LEARNED.md)
(survival-analysis traps). This file covers the probabilistic-CLV side:
BG/NBD + Gamma-Gamma, their Bayesian implementations, and Markov-chain LTV.
Every item below is a real failure mode that these models can and do produce
— each one is guarded against, or explicitly demonstrated, in this folder's
notebooks and scripts.

---

## 1. BG/NBD has degenerate regimes — check the parameters, not just the fit

A converged fit is not a healthy fit. Two pathologies to check for **every
time**, one line each:

- **`a + b ≤ 1`** (the dropout Beta parameters). In `lifetimes`, the
  conditional expected-purchases formula evaluates a Gaussian hypergeometric
  term whose argument goes invalid in this regime — the result is **NaN
  predictions for frequency-0 customers**, silently. If your holdout
  evaluation then does a quiet `dropna()`, you have removed exactly the
  hardest, lowest-activity customers from your metrics and inflated them.
- **`a → 0`** (dropout probability collapsing to zero). The model has
  concluded that nobody ever dies: `p_alive ≈ 1` for everyone, and any
  ranking built on it is vacuous. This happens for structural reasons —
  dense loyal panels (see the Dunnhumby stage-04 diagnostic), or
  survivorship in the training data (lesson 2).

```python
r, alpha, a, b = bgf.params_
assert a + b > 1, "degenerate regime: conditional expectations will NaN at frequency=0"
```

For Gamma-Gamma the analogous check is **`q > 1`**: the implied population
mean spend is `p·v / (q−1)`. If `q ≤ 1` that mean is undefined, and the
shrinkage weight flips sign — instead of pulling noisy customer averages
toward a population mean, the model pushes **every** prediction away from
the observed data. The tell: 100% of customers with predicted spend above
their observed average. Print the implied mean after every fit.

---

## 2. Survivorship in the training data forges the dropout parameter

Fit BG/NBD only on customers who are currently active (or who survived to
some landmark) and you have shown the model a world where **no purchase
stream ever ends**. The dropout parameter goes to zero *by construction*,
`p_alive` is identically 1.0, and none of it is evidence about anything.
Fit on the full transaction history — including the customers who already
lapsed or cancelled — or the exercise is circular.

Related display trap: a constant score fed to `roc_auc_score` prints 0.500
by tie convention. That is not a "measured chance-level AUC"; it is
"ranking undefined." Label it as such or don't put it in a comparison bar
chart.

---

## 3. `p_alive = 1.0` at frequency 0 is an artifact, not an estimate

A customer with no repeat purchase has never given the BG process a dropout
opportunity, so `conditional_probability_alive` returns exactly 1.0 — a
formula convention. Consequences:

- the `p_alive` histogram has a spike at 1.0 (explain it, or readers will
  read it as "lots of super-healthy customers");
- never build a risk tier on raw `p_alive` without masking frequency-0
  customers (this repo routes them to a "New/Unproven" tier);
- agreement statistics between two BTYD implementations are inflated if you
  pool frequency-0 customers, because both return 1.0 identically.

---

## 4. If you simulate data, simulate the model's world — or disclose that you didn't

Two honest options for a pedagogical or benchmark simulation:

1. **Generate from the model's own assumptions** (Poisson purchasing with
   gamma heterogeneity, dropout coin after each repeat purchase, Gamma-Gamma
   spend). Then fitted parameters should recover the truth, and you can
   *show* that — parameter recovery, `p_alive` vs the stored alive flag
   (AUC), predicted vs latent spend. That's a real validation section for
   free.
2. **Deliberately generate from a different world** (e.g., calendar-time
   dropout — the Pareto/NBD story) as a robustness exercise. Fine — but say
   so, because BG/NBD's `p_alive` will be systematically overconfident
   there, and a reader who checks will find it.

The indefensible middle ground: simulating a mismatched world by accident,
storing the ground truth, and never comparing the model against it.

---

## 5. Know your library's purchase-occasion convention

`lifetimes.summary_data_from_transaction_data(..., freq="D")` merges
same-day orders into **one purchase occasion** and sums their value before
averaging. So:

- `frequency` ≠ (number of orders − 1) whenever same-day orders exist;
- `monetary_value` is the mean of *day-level* spend over repeat purchase
  days, first purchase excluded.

If you also build RFM "by hand" (as the PyMC notebook does), collapse to
purchase days first, or the two implementations silently model different
quantities from the same table.

---

## 6. Validation that quietly drops customers is not validation

Rules for a calibration/holdout backtest:

- Refit on **calibration data only**; score the true holdout window length.
- Evaluate **all** customers. If any prediction is NaN, that is a model
  failure to surface (lesson 1), not a row to drop. `assert
  predictions.notna().all()` is one line.
- Report an **aggregate bias line** (predicted vs actual total purchases)
  next to MAE/RMSE/rank correlation — decile tables hide net over/under
  prediction.
- If the number of evaluated customers differs from the cohort size,
  explain the difference in the text. "Evaluated: 1,212 of 1,400" with no
  explanation is where a reviewer starts digging.

---

## 7. A CLV × p_alive quadrant has confounded axes

The classic action matrix — value on one axis, "fading" (`p_alive`) on the
other — is empty in its most-advertised cell if you use model CLV as the
value axis: **CLV already embeds `p_alive`**, so "high CLV, low p_alive"
customers barely exist by construction. Use an *observable* value axis
(e.g., annual spend run-rate) against the model's risk axis, and the
win-back quadrant becomes a real, targetable segment. If a promised segment
comes back empty, say why — don't let a `groupby` drop it silently.

---

## 8. Bayesian means priors you can defend and diagnostics you actually ran

- **"Weak prior" is a claim about scale, not a vibe.** A `HalfNormal(10)`
  prior on a timescale parameter that lives near 50 days is *informative*
  (the posterior mode sits 4+ prior SDs into the tail and gets visibly
  shrunk). Scale priors to the parameter's natural range, and check: if the
  posterior mode lands in the extreme tail of your prior, your prior was
  not weak.
- **Know the defaults.** `pymc-marketing`'s Gamma-Gamma default priors are
  improper flat (`HalfFlat`) — under MAP that is plain MLE with extra
  steps. State priors explicitly for every model in the analysis, not just
  the first one.
- **MAP is the posterior mode** (prior × likelihood) — not "the most likely
  value under the prior" — and `find_MAP` optimizes in transformed space,
  so it is an approximation on top.
- **If uncertainty is the selling point, show uncertainty.** A MAP-only
  analysis advertising "Bayesian, uncertainty-aware" delivers neither. The
  minimum bar: ≥ 4 chains, R-hat < 1.01, bulk-ESS > 400 (checked, printed),
  credible intervals on parameters, and posterior spread on at least one
  customer-level quantity.
- **Unit traps in the API**: `expected_purchases(future_t=...)` is in the
  data's time units (days here); `expected_customer_lifetime_value(
  future_t=..., discount_rate=...)` measures `future_t` in **months** with
  a **monthly** discount rate regardless of `time_unit`. Name the units at
  every call site.

---

## 9. Markov LTV: the state definitions must match the transition structure

If states are defined as *months-of-silence tiers* ("Warm = no purchase
this month"), then (a) silent states cannot earn revenue, and (b) the only
legal moves are deterministic aging (Warm → Cooling → Dormant) plus
"purchase → Active". A matrix with `Warm→Warm = 0.27` and revenue in every
state contradicts those definitions on sight. If you want free-form
transitions and per-state revenue — the Pfeifer–Carraway segment-migration
model — define states as **monthly engagement tiers**, and let every
non-churned tier transact. Pick one semantics and hold it everywhere:
narrative, generator, matrix, reward vector.

---

## 10. In non-contractual data, "Churned" as an absorbing state is a convention

BTYD exists because non-contractual churn is *latent*. A Markov model with
an observable absorbing "Churned" state is therefore making an operational
choice (recency cutoff, account closure) that real customers routinely
violate by coming back. Say so, and treat the absorption assumption as
something to sensitivity-check — not a fact of nature.

---

## 11. Row sums equal to 1 is not validation

That's the normalization you just performed. Real checks for an estimated
transition matrix:

- **Row support**: show the transition *counts*, not just probabilities —
  a row estimated from 30 observations and a row from 2,000 deserve
  different trust.
- **Recovery** (if simulated): `max |P̂ − P_true|`.
- **Temporal backtest**: estimate P on the first K months, project state
  counts forward, compare with the actual later months.
- **Uncertainty**: transition rows are estimates; bootstrap customers (or
  put Dirichlet posteriors on rows) and propagate to LTV. Reporting
  state-level LTV to the cent from a point estimate is false precision.
- **Zero-visit states**: `counts.div(row_sums).fillna(0)` silently turns a
  never-visited state into an all-zero (sub-stochastic) row. Assert
  coverage instead.

And two assumptions to name out loud: rewards independent of the next
transition given the state (false when big spenders retain better), and
time-homogeneity (real transition rates drift).

---

## 12. Scenario analyses need costs, sensitivity, and causal humility

A "win-back campaign" scenario that shifts a transition probability and
reports the gross LTV delta answers the CFO's question halfway. Net the
campaign cost against the affected state's reward, sweep the assumed shift
size (it is an assumption), and keep the disclaimer that a transition-matrix
what-if is not a causal estimate — an experiment is.

---

## 13. Model comparisons on simulated data are stacked by construction

If the data-generating process is a proportional-hazards model in exactly
the covariates you hand to Cox, then Cox is correctly specified *by
design* — the comparison's qualitative lesson can be real, but the
magnitudes are simulation artifacts, and PH diagnostics pass trivially.
Disclose it. Corollaries:

- AUC differences need uncertainty before you crown a winner: with ~65
  positives in a test fold, AUC standard errors are ±0.05 — bootstrap the
  comparison.
- Wald p-values from a **penalized** Cox fit are not valid inference
  (same lesson as the survival guide, item 16 — it applies to display
  tables in notebooks too).
- Contractual settings come in **discrete** (annual renewals — sBG,
  per-renewal hazards) and **continuous** (cancel-anytime) time. A
  "contractual vs non-contractual" discussion that silently covers only
  one quadrant of the Fader–Hardie taxonomy will be called on it.

---

## 14. Notebook hygiene that protects all of the above

- **Don't blanket-suppress warnings.** The RuntimeWarnings you silence are
  the exact signal that a fit entered a degenerate regime (lesson 1).
  Filter narrowly (FutureWarning/UserWarning) if you must.
- **Seed inside the generator.** A simulation that consumes a module-level
  RNG reproduces only under run-once discipline; `default_rng(seed)` inside
  the function makes every cell independently re-runnable.
- **Every number in the markdown must match an output**, every promised
  section must exist, and every promised segment must be non-empty — or
  the text must say why. Reviewers grep for exactly these gaps.

---

## Checklist before you publish a BTYD / LTV analysis

- [ ] BG/NBD: `a + b > 1`, `a` not ≈ 0, params printed and sane.
- [ ] Gamma-Gamma: `q > 1`, implied population mean finite and plausible,
      frequency–monetary independence checked *and interpreted*.
- [ ] Trained on the full history — no survivor-only fitting.
- [ ] `p_alive` frequency-0 artifact explained; no tier built on it.
- [ ] Holdout backtest scores every customer; zero silent drops; aggregate
      bias reported.
- [ ] Purchase-occasion convention (same-day merging) consistent across
      every RFM construction in the project.
- [ ] Action quadrants use an observable value axis; no empty flagship
      segment.
- [ ] Bayesian: priors stated for every model and defensible at the
      parameter's scale; ≥ 4 chains; R-hat/ESS printed and passing;
      credible intervals actually shown.
- [ ] Markov: state semantics match the matrix; counts shown; recovery or
      backtest shown; LTV with uncertainty; absorbing churn acknowledged as
      operational.
- [ ] Scenarios net of cost, with sensitivity, framed non-causally.
- [ ] No blanket warning suppression; seeded generators; markdown numbers
      match outputs.
