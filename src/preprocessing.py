"""Clean the raw PaySim data and save it for analysis and modelling.

Cleaning is deliberately minimal. Each rule below exists because of a specific
finding in ``notebooks/01_fraud_eda.ipynb``; everything else is kept as-is,
including PaySim's unusual balance behaviour, which is a property of the
simulator rather than corruption.

Run as a script to build ``data/processed/``:

    python -m src.preprocessing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data_loader import PROCESSED_DATA_DIR, load_paysim
from src.data_validation import check_schema

logger = logging.getLogger(__name__)

CLEAN_PARQUET_PATH = PROCESSED_DATA_DIR / "transactions_clean.parquet"
EXCLUDED_ROWS_PATH = PROCESSED_DATA_DIR / "excluded_rows.csv"


@dataclass
class CleaningReport:
    """Audit trail of what cleaning changed."""

    input_rows: int
    output_rows: int = 0
    excluded: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Input rows:  {self.input_rows:,}", f"Output rows: {self.output_rows:,}"]
        lines += [f"  excluded ({reason}): {n:,}" for reason, n in self.excluded.items()]
        return "\n".join(lines)


def format_transaction_id(transaction_id: int) -> str:
    """Human-readable ID used in the API and dashboard, e.g. 7 -> 'TX-0000007'."""
    return f"TX-{transaction_id:07d}"


def add_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable integer ID: the row's 1-based position in the raw file.

    PaySim has no transaction ID. We need one to trace a row from SQL results,
    through model predictions, to SHAP explanations. It is stored as an integer
    (cheap, and an efficient SQLite primary key) and formatted only for display.

    Must be called on the *raw* data, before any rows are removed, so the IDs
    always point back to the original file.
    """
    out = df.copy()
    out.insert(0, "transaction_id", pd.RangeIndex(1, len(df) + 1, dtype="int64"))
    return out


def find_zero_amount_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of transactions where no money moved.

    In PaySim these are 16 CASH_OUTs, all labelled fraud, all from zero-balance
    accounts. A "fraud" that moves nothing has no transaction behaviour to
    learn from, so we exclude these rows rather than let them distort amount-
    and balance-based features.
    """
    return df["amount"] == 0


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Apply the cleaning rules.

    Args:
        df: Raw transactions with ``transaction_id`` already added.

    Returns:
        (clean rows, excluded rows with a ``reason`` column, report)
    """
    schema = check_schema(df.drop(columns="transaction_id", errors="ignore"))
    if not schema["is_valid"]:
        raise ValueError(f"Cannot clean data with missing columns: {schema['missing_columns']}")
    if "transaction_id" not in df.columns:
        raise ValueError("Call add_transaction_id() on the raw data before cleaning.")

    report = CleaningReport(input_rows=len(df))

    rules = {"zero_amount": find_zero_amount_rows(df)}

    exclude_mask = pd.Series(False, index=df.index)
    excluded_parts = []
    for reason, mask in rules.items():
        new_rows = mask & ~exclude_mask  # attribute each row to its first matching rule
        report.excluded[reason] = int(new_rows.sum())
        excluded_parts.append(df[new_rows].assign(reason=reason))
        exclude_mask |= mask

    clean = df[~exclude_mask].reset_index(drop=True)
    excluded = pd.concat(excluded_parts, ignore_index=True)
    report.output_rows = len(clean)

    logger.info("Cleaning complete\n%s", report.summary())
    return clean, excluded, report


def build_clean_dataset(
    output_path: Path = CLEAN_PARQUET_PATH,
    excluded_path: Path = EXCLUDED_ROWS_PATH,
) -> CleaningReport:
    """Load raw data, clean it, and write the clean Parquet + excluded-rows audit file."""
    raw = add_transaction_id(load_paysim())
    clean, excluded, report = clean_transactions(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    logger.info("Wrote %s and %s", output_path, excluded_path)
    return report


def load_clean_transactions(path: Path = CLEAN_PARQUET_PATH) -> pd.DataFrame:
    """Load the cleaned dataset produced by :func:`build_clean_dataset`."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.preprocessing` first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    build_clean_dataset()
