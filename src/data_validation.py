"""Reusable data-quality checks for the PaySim transactions dataset.

Every function here *reports* on the data and never mutates or drops rows.
Deciding what to clean is a separate, explicit step so that each removal can
be justified.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_loader import EXPECTED_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

BALANCE_COLUMNS = [
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
]
MONETARY_COLUMNS = ["amount", *BALANCE_COLUMNS]

# PaySim simulates 30 days at one step per hour: 744 steps.
MIN_STEP, MAX_STEP = 1, 744

# Balances are recorded to the cent; allow for floating-point noise.
BALANCE_TOLERANCE = 0.01


def check_schema(df: pd.DataFrame, expected: Sequence[str] = EXPECTED_COLUMNS) -> dict:
    """Compare the DataFrame's columns with the expected schema."""
    actual = list(df.columns)
    return {
        "missing_columns": [c for c in expected if c not in actual],
        "unexpected_columns": [c for c in actual if c not in expected],
        "is_valid": set(expected) <= set(actual),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column."""
    counts = df.isna().sum()
    return pd.DataFrame(
        {"missing_count": counts, "missing_pct": (counts / len(df) * 100).round(4)}
    ).sort_values("missing_count", ascending=False)


def duplicate_report(df: pd.DataFrame, subset: Iterable[str] | None = None) -> dict:
    """Count exact duplicate rows (optionally on a subset of columns)."""
    subset = list(subset) if subset is not None else None
    n_dup = int(df.duplicated(subset=subset).sum())
    return {
        "subset": subset or "all columns",
        "duplicate_rows": n_dup,
        "duplicate_pct": round(n_dup / len(df) * 100, 4) if len(df) else 0.0,
    }


def class_distribution(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Count and share of each target class, plus the imbalance ratio."""
    counts = df[target].value_counts().sort_index()
    out = pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(4)})
    out.index.name = target
    return out


def imbalance_ratio(df: pd.DataFrame, target: str = TARGET_COLUMN) -> float:
    """Number of majority-class rows per minority-class row."""
    counts = df[target].value_counts()
    if len(counts) < 2:
        raise ValueError(f"Target '{target}' has fewer than two classes.")
    return float(counts.max() / counts.min())


def fraud_rate_by_type(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Transaction count, fraud count and fraud rate per transaction type."""
    grouped = df.groupby("type", observed=True)[target].agg(transactions="size", fraud="sum")
    grouped["fraud_rate_pct"] = (grouped["fraud"] / grouped["transactions"] * 100).round(4)
    grouped["share_of_all_fraud_pct"] = (grouped["fraud"] / grouped["fraud"].sum() * 100).round(2)
    return grouped.sort_values("fraud_rate_pct", ascending=False)


def impossible_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count values that are impossible or suspicious for a payment ledger.

    - negative amounts or balances (impossible)
    - zero-amount transactions (suspicious: nothing moved)
    - steps outside the simulated 1..744 hour range (impossible)
    """
    checks = {f"negative_{col}": int((df[col] < 0).sum()) for col in MONETARY_COLUMNS}
    checks["zero_amount"] = int((df["amount"] == 0).sum())
    checks["step_out_of_range"] = int((~df["step"].between(MIN_STEP, MAX_STEP)).sum())
    return pd.DataFrame.from_dict(checks, orient="index", columns=["count"])


def account_prefix_report(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of account-ID prefixes (PaySim uses C=customer, M=merchant)."""
    orig = df["name_orig"].str[0].value_counts().rename("name_orig")
    dest = df["name_dest"].str[0].value_counts().rename("name_dest")
    return pd.concat([orig, dest], axis=1).fillna(0).astype(int)


def balance_consistency_report(
    df: pd.DataFrame, tolerance: float = BALANCE_TOLERANCE
) -> pd.DataFrame:
    """Per transaction type, how often balance changes are consistent with the amount.

    We check whether the *absolute* balance change equals the transaction
    amount. Using the absolute value avoids hard-coding assumptions about the
    direction of money flow for each transaction type.

    Merchant destinations (``name_dest`` starting with "M") are excluded from
    the destination check because PaySim does not record merchant balances.

    Columns returned (all percentages of rows within that type):
        orig_change_matches_amount_pct
        orig_zero_before_pct        - origin had 0 balance before transacting
        dest_change_matches_amount_pct (customer destinations only)
        dest_zero_before_and_after_pct - customer destination shows 0 -> 0
                                         despite receiving a non-zero amount
    """
    orig_change = (df["new_balance_orig"] - df["old_balance_orig"]).abs()
    dest_change = (df["new_balance_dest"] - df["old_balance_dest"]).abs()
    is_customer_dest = df["name_dest"].str.startswith("C").fillna(False)

    flags = pd.DataFrame(
        {
            "type": df["type"],
            "orig_match": (orig_change - df["amount"]).abs() <= tolerance,
            "orig_zero_before": df["old_balance_orig"] == 0,
            "customer_dest": is_customer_dest,
            "dest_match": ((dest_change - df["amount"]).abs() <= tolerance) & is_customer_dest,
            "dest_zero_zero": (
                (df["old_balance_dest"] == 0)
                & (df["new_balance_dest"] == 0)
                & (df["amount"] > 0)
                & is_customer_dest
            ),
        }
    )

    grouped = flags.groupby("type", observed=True)
    n = grouped.size()
    n_customer_dest = grouped["customer_dest"].sum().replace(0, np.nan)

    report = pd.DataFrame(
        {
            "transactions": n,
            "orig_change_matches_amount_pct": grouped["orig_match"].sum() / n * 100,
            "orig_zero_before_pct": grouped["orig_zero_before"].sum() / n * 100,
            "customer_dest_rows": grouped["customer_dest"].sum(),
            "dest_change_matches_amount_pct": grouped["dest_match"].sum() / n_customer_dest * 100,
            "dest_zero_before_and_after_pct": grouped["dest_zero_zero"].sum() / n_customer_dest * 100,
        }
    )
    return report.astype("float64").round(2)


def flagged_vs_actual_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate PaySim's built-in rule flag against actual fraud."""
    return pd.crosstab(
        df["is_flagged_fraud"], df[TARGET_COLUMN], margins=True, margins_name="total"
    )


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    """Run every check and return the results keyed by check name."""
    schema = check_schema(df)
    if not schema["is_valid"]:
        raise ValueError(f"Schema check failed: missing {schema['missing_columns']}")

    results = {
        "shape": df.shape,
        "schema": schema,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_value_report(df),
        "duplicates": duplicate_report(df),
        "class_distribution": class_distribution(df),
        "imbalance_ratio": imbalance_ratio(df),
        "fraud_by_type": fraud_rate_by_type(df),
        "impossible_values": impossible_value_report(df),
        "account_prefixes": account_prefix_report(df),
        "balance_consistency": balance_consistency_report(df),
        "flagged_vs_actual": flagged_vs_actual_fraud(df),
    }
    logger.info(
        "Data-quality checks complete: %s rows, %s duplicates, imbalance %.1f:1",
        f"{df.shape[0]:,}",
        results["duplicates"]["duplicate_rows"],
        results["imbalance_ratio"],
    )
    return results
