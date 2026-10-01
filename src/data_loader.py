"""Load the raw PaySim transaction dataset.

The dataset is downloaded manually (see README) and placed in ``data/raw/``.
Its location can be overridden with the ``PAYSIM_PATH`` environment variable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PAYSIM_FILENAME = "PS_20174392719_1491204439457_log.csv"

# PaySim's original headers are inconsistent (e.g. "oldbalanceOrg" vs
# "newbalanceOrig"). We rename them once, at load time, so the rest of the
# codebase uses one predictable naming convention.
COLUMN_RENAMES = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

# Explicit dtypes keep memory predictable on ~6M rows. Monetary columns stay
# float64: float32 only holds ~7 significant digits, which would corrupt
# balances in the tens of millions.
RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

EXPECTED_COLUMNS = list(COLUMN_RENAMES.values())
TARGET_COLUMN = "is_fraud"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the PaySim CSV cannot be located."""


def resolve_paysim_path(path: Optional[str | Path] = None) -> Path:
    """Return the path to the PaySim CSV.

    Resolution order: explicit ``path`` argument, then the ``PAYSIM_PATH``
    environment variable, then ``data/raw/<PAYSIM_FILENAME>``.
    """
    candidate = Path(path or os.getenv("PAYSIM_PATH") or RAW_DATA_DIR / PAYSIM_FILENAME)
    if not candidate.is_file():
        raise DatasetNotFoundError(
            f"PaySim dataset not found at '{candidate}'. Download "
            f"'{PAYSIM_FILENAME}' from Kaggle (ealaxi/paysim1) and place it in "
            f"'{RAW_DATA_DIR}', or set the PAYSIM_PATH environment variable."
        )
    return candidate


def load_paysim(
    path: Optional[str | Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the raw PaySim CSV with explicit dtypes and snake_case columns.

    Args:
        path: Optional explicit path to the CSV.
        nrows: Optionally read only the first ``nrows`` rows (useful for quick
            experiments; note PaySim is ordered by ``step``, so a head sample
            is *not* representative of the whole time range).

    Returns:
        The raw transactions with renamed columns. No rows are dropped or
        modified here: cleaning decisions are made explicitly downstream.
    """
    csv_path = resolve_paysim_path(path)
    logger.info("Loading PaySim data from %s", csv_path)

    df = pd.read_csv(csv_path, dtype=RAW_DTYPES, nrows=nrows)

    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"PaySim file is missing expected columns: {sorted(missing)}")

    df = df.rename(columns=COLUMN_RENAMES)
    logger.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])
    return df
