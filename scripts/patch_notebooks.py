"""Programmatically patch the Online Retail notebooks:

    1. Add a prominent methodology-update banner at the top of
       customer-survival-analysis.ipynb and stage1-conversion-model.ipynb.
    2. Surgically fix stage1 cell 15: replace `eval_set=[(X_test, y_test)]`
       with a new validation split so early stopping no longer peeks at test.

Larger restructures (landmark survival, conditional S(Δ)) remain in the
`scripts/` Python pipeline rather than being rewritten inside the
notebook — the notebooks still demonstrate the original analysis for
historical context, with the banner making the methodological upgrade
explicit.

Run from scripts/:
    ../venv/bin/python patch_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SURV_NB = ROOT / "customer-survival-analysis.ipynb"
CLV_NB = ROOT / "clv-prediction-benchmark.ipynb"
STAGE1_NB = ROOT / "stage1-conversion-model.ipynb"

BANNER_SURV = """## Methodology update (post-review)

The original survival implementation below (cells 18, 23, 30, 41) has been superseded by a landmark-based redesign that fixes three methodological issues flagged in review:

1. **Event/censoring time misalignment** — `duration = last_purchase - first_purchase` puts the event on the wrong time axis. Churners should have `event_time = last_purchase + 45d`, censored customers should be anchored at `observation_end`.
2. **Retrospective feature engineering** — features are built from the full customer history, then a random split is used. This is outcome-adjacent.
3. **Unconditional scorecard S(t)** — `S(30d)`, `S(60d)`, `S(90d)` as reported are survival from first purchase, not conditional-on-still-active-today future risk.

The corrected implementation lives in [`scripts/01_survival.py`](scripts/01_survival.py) and [`scripts/03_scorecard.py`](scripts/03_scorecard.py). After the fix, test C-index on the alive-at-landmark cohort drops from ~0.89 to ~0.75 — this is the honest number.

The cells below are retained for historical / comparison purposes."""

BANNER_STAGE1 = """## Methodology update (post-review)

Cell 15 in the original notebook early-stops XGBoost on the test fold (`eval_set=[(X_test, y_test)]`), which is a subtle form of test-set leakage. The fix below introduces a validation split carved from the training set and uses it for early stopping. Test metrics are unchanged in spirit but now report honestly.

See also [`scripts/02_stage1.py`](scripts/02_stage1.py) for the end-to-end corrected pipeline."""

# Replacement XGBoost cell for stage1-conversion-model (fixes early stopping on test).
STAGE1_XGB_REPLACEMENT = '''# --- XGBoost (review-fixed: early stopping on validation, not test) ---
# Previously: eval_set=[(X_test, y_test)] — this leaks test-set signal into model selection.
# Fix: carve a 15% validation slice from train and early-stop on it.

from sklearn.model_selection import train_test_split

X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

neg_count = (y_train_xgb == 0).sum()
pos_count = (y_train_xgb == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, gamma=1,
    scale_pos_weight=neg_count / pos_count,
    eval_metric='auc',
    early_stopping_rounds=30,
    random_state=42, n_jobs=-1, verbosity=0
)

xgb_model.fit(
    X_train_xgb, y_train_xgb,
    eval_set=[(X_val_xgb, y_val_xgb)],   # <-- validation, not test
    verbose=False
)

xgb_proba_train = xgb_model.predict_proba(X_train)[:, 1]
xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]

print('=== XGBoost ===')
print(f'  Train AUC-ROC: {roc_auc_score(y_train, xgb_proba_train):.4f}')
print(f'  Test  AUC-ROC: {roc_auc_score(y_test, xgb_proba_test):.4f}')
print(f'  Test  PR-AUC:  {average_precision_score(y_test, xgb_proba_test):.4f}')
print(f'  Best iteration (on val): {xgb_model.best_iteration}')
'''


def prepend_banner(nb_path: Path, banner_text: str):
    nb = nbformat.read(nb_path, as_version=4)
    # Skip if banner already present
    for c in nb.cells[:3]:
        if c.cell_type == "markdown" and "Methodology update (post-review)" in c.source:
            print(f"  [skip] {nb_path.name} already has banner")
            return
    cell = nbformat.v4.new_markdown_cell(banner_text)
    nb.cells.insert(0, cell)
    nbformat.write(nb, nb_path)
    print(f"  [banner added] {nb_path.name}")


def patch_stage1_xgb(nb_path: Path):
    nb = nbformat.read(nb_path, as_version=4)
    code_idx = 0
    target_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        code_idx += 1
        if "eval_set=[(X_test, y_test)]" in c.source and "xgb" in c.source.lower():
            target_idx = i
            break
    if target_idx is None:
        print(f"  [stage1] could not find XGBoost-with-test-eval cell; no-op")
        return
    old = nb.cells[target_idx].source
    if "review-fixed: early stopping on validation" in old:
        print(f"  [stage1] cell {target_idx} already patched; no-op")
        return
    nb.cells[target_idx].source = STAGE1_XGB_REPLACEMENT
    # Clear outputs so stale numbers don't mislead
    nb.cells[target_idx].outputs = []
    nb.cells[target_idx].execution_count = None
    nbformat.write(nb, nb_path)
    print(f"  [stage1] patched cell index {target_idx} (code cell #{code_idx})")


def main():
    print("Patching Online Retail notebooks")
    print(f"  surv:   {SURV_NB}")
    print(f"  stage1: {STAGE1_NB}")

    if SURV_NB.exists():
        prepend_banner(SURV_NB, BANNER_SURV)
    if STAGE1_NB.exists():
        prepend_banner(STAGE1_NB, BANNER_STAGE1)
        patch_stage1_xgb(STAGE1_NB)
    print("Done.")


if __name__ == "__main__":
    main()
