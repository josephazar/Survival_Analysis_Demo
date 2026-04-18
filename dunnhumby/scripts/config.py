"""Shared configuration for the Dunnhumby survival analysis pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent.parent

DATA_DIR = REPO_ROOT / "data" / "dunnhumby"
PROCESSED_DIR = ROOT / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

SYNTHETIC_EPOCH = "2017-01-01"
MAX_DAY = 711
# Chosen via churn-window analysis (Kneedle elbow = 16d -> nearest candidate = 14d).
# See artifacts/churn_window/summary.json for rationale.
CHURN_WINDOW = 14
RANDOM_STATE = 42

# Landmark for the survival analysis. Features are built from DAY <= LANDMARK_DAY only;
# the survival target is measured from LANDMARK_DAY forward. Chosen to leave
# MAX_DAY - LANDMARK_DAY = 211 days of follow-up, which is comfortably > CHURN_WINDOW.
LANDMARK_DAY = 500
