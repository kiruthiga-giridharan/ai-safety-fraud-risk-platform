"""Feature engineering for the fraud model.

Framing: **post-transaction monitoring**. A transaction is scored after it has
been processed, so post-transaction balances (``new_balance_*``) are known at
scoring time. A stricter real-time (pre-authorisation) setting is approximated
by the ``no_origin_balance`` feature set.

Every decision here is traced to a finding in notebooks 01 / 01b. Run as a
script to build the train / validation / test splits:

    python -m src.feature_engineering
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import PROCESSED_DATA_DIR, TARGET_COLUMN
from src.data_validation import BALANCE_TOLERANCE

logger = logging.getLogger(__name__)

# Fraud only occurs in these types (notebook 01, Q2). Other types are out of
# model scope: training on them would add millions of trivially-easy negatives
# and inflate metrics without teaching the model anything.
MODEL_TYPES = ("TRANSFER", "CASH_OUT")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

SPLIT_PATHS = {
    split: PROCESSED_DATA_DIR / f"fraud_{split}.parquet"
    for split in ("train", "validation", "test")
}

# Columns that must never be model inputs, and why.
EXCLUDED_COLUMNS = {
    "step": "Simulation artefact: legitimate traffic disappears in parts of the timeline "
            "(320 of 743 steps are 100% fraud), so the model would learn when the simulator "
            "ran, not fraud behaviour.",
    "name_orig": "Account identifier: 6.35M unique values; the model would memorise identities.",
    "name_dest": "Account identifier: memorisation risk; no mule-account concentration to exploit.",
    "is_flagged_fraud": "Output of another fraud system (16 flags, all fraud), not an input signal.",
    "transaction_id": "Row identifier; follows file order, which encodes the TRANSFER->CASH_OUT "
                      "chain (cash-out is the next row in 4,075 fraud pairs).",
    "type": "Replaced by the binary is_transfer (only two types are in scope).",
    "amount": "Replaced by log_amount (same information, less skew).",
}

# Note: chain features and whole-dataset account counts are also excluded.
# They are not columns here, so they are documented in notebook 02.

BASE_FEATURES = ["log_amount", "is_transfer"]

ORIGIN_BALANCE_FEATURES = [
    "log_old_balance_orig",
    "log_new_balance_orig",
    "orig_balance_error",
    "log_amount_to_orig_balance",
    "orig_zero_before",
    "amount_exceeds_orig_balance",
    "orig_fully_drained",
]

DEST_BALANCE_FEATURES = [
    "log_old_balance_dest",
    "log_new_balance_dest",
    "dest_balance_error",
    "dest_zero_before_and_after",
]

FEATURE_SETS = {
    # Everything available in post-transaction monitoring.
    "full": BASE_FEATURES + ORIGIN_BALANCE_FEATURES + DEST_BALANCE_FEATURES,
    # Drops the origin-balance features that encode PaySim's "fraud empties the
    # account" artefact (one rule on these gives 97.82% recall at 100% precision).
    # Comparing the two sets shows what a model learns beyond that artefact.
    "no_origin_balance": BASE_FEATURES + DEST_BALANCE_FEATURES,
}


def signed_log1p(x: pd.Series) -> pd.Series:
    """sign(x) * log(1 + |x|): compresses huge magnitudes but keeps the sign and maps 0 -> 0."""
    return np.sign(x) * np.log1p(np.abs(x))


def filter_model_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only transaction types that the model is trained to score."""
    return df[df["type"].isin(MODEL_TYPES)].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every engineered feature from the cleaned transaction columns.

    All features use only the transaction's own fields, never other rows, so
    the same function works for one transaction in the API and millions here.

    Returns a DataFrame with one column per feature, aligned with ``df``.
    """
    amount = df["amount"]
    old_orig, new_orig = df["old_balance_orig"], df["new_balance_orig"]
    old_dest, new_dest = df["old_balance_dest"], df["new_balance_dest"]

    features = pd.DataFrame(index=df.index)

    # --- base -------------------------------------------------------------
    # Amounts span 0.01 to 92M; the log keeps large values from dominating.
    features["log_amount"] = np.log1p(amount)
    # TRANSFER fraud rate is ~4x CASH_OUT's (0.77% vs 0.18%).
    features["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")

    # --- origin account ----------------------------------------------------
    features["log_old_balance_orig"] = np.log1p(old_orig)
    features["log_new_balance_orig"] = np.log1p(new_orig)
    # Both in-scope types debit the origin, so a consistent ledger has
    # old - amount - new = 0. Positive = less left than expected (e.g. the
    # balance was floored at 0); negative = more left than expected.
    features["orig_balance_error"] = signed_log1p(old_orig - amount - new_orig)
    # log((amount + 1) / (old + 1)): ~0 when the whole balance is moved,
    # > 0 when the amount exceeds the balance, < 0 for a partial spend.
    # Defined even when the balance is 0, unlike a plain ratio.
    features["log_amount_to_orig_balance"] = np.log1p(amount) - np.log1p(old_orig)
    # 47.37% of legitimate vs 0.50% of fraud start from a zero balance.
    features["orig_zero_before"] = (old_orig == 0).astype("int8")
    # Legitimate customers overspend (90.1%); fraud almost never does (0.35%).
    features["amount_exceeds_orig_balance"] = (amount > old_orig + BALANCE_TOLERANCE).astype("int8")
    # The whole balance is moved: 97.82% of fraud, 0.0001% of legitimate.
    features["orig_fully_drained"] = (
        (old_orig > 0) & ((amount - old_orig).abs() <= BALANCE_TOLERANCE)
    ).astype("int8")

    # --- destination account -----------------------------------------------
    features["log_old_balance_dest"] = np.log1p(old_dest)
    features["log_new_balance_dest"] = np.log1p(new_dest)
    # Destination is credited, so consistent means old + amount - new = 0.
    features["dest_balance_error"] = signed_log1p(old_dest + amount - new_dest)
    # Destination shows 0 -> 0 despite receiving money: 49.63% of fraud vs 0.06% of legitimate.
    features["dest_zero_before_and_after"] = ((old_dest == 0) & (new_dest == 0)).astype("int8")

    expected = FEATURE_SETS["full"]
    assert list(features.columns) == expected, "feature order drifted from FEATURE_SETS"
    return features


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Scope-filter, build features, and attach transaction_id + target."""
    scoped = filter_model_scope(df)
    table = build_features(scoped)
    table.insert(0, "transaction_id", scoped["transaction_id"].to_numpy())
    table[TARGET_COLUMN] = scoped[TARGET_COLUMN].to_numpy()
    return table


def split_train_validation_test(
    table: pd.DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Stratified random split into train / validation / test.

    - Stratified: with 0.3% fraud, an unstratified split could leave a split
      noticeably short of fraud cases.
    - Random rather than chronological: legitimate traffic vanishes after day 17,
      so a time-based test set would be mostly fraud and unrepresentative.
    - Validation exists so thresholds and risk-level boundaries (Phase 5) are
      chosen without ever looking at the test set.
    """
    train_val, test = train_test_split(
        table, test_size=test_size, stratify=table[TARGET_COLUMN], random_state=random_state
    )
    relative_val = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[TARGET_COLUMN], random_state=random_state
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, fraud count and fraud rate for each split."""
    return pd.DataFrame(
        {
            name: {
                "rows": len(part),
                "fraud": int(part[TARGET_COLUMN].sum()),
                "fraud_rate_pct": round(part[TARGET_COLUMN].mean() * 100, 4),
            }
            for name, part in splits.items()
        }
    ).T.astype({"rows": "int64", "fraud": "int64"})


def build_and_save_splits(paths: dict[str, Path] = SPLIT_PATHS) -> pd.DataFrame:
    """Build the model table from the cleaned data, split it, and save each split."""
    from src.preprocessing import load_clean_transactions

    table = build_model_table(load_clean_transactions())
    splits = split_train_validation_test(table)
    for name, part in splits.items():
        part.to_parquet(paths[name], index=False)
        logger.info("Wrote %s (%s rows)", paths[name], f"{len(part):,}")
    summary = split_summary(splits)
    logger.info("Split summary:\n%s", summary)
    return summary


def load_split(name: str, feature_set: str = "full") -> tuple[pd.DataFrame, pd.Series]:
    """Load one saved split and return (X, y) for the chosen feature set."""
    path = SPLIT_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.feature_engineering` first.")
    part = pd.read_parquet(path)
    return part[FEATURE_SETS[feature_set]], part[TARGET_COLUMN]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    build_and_save_splits()
