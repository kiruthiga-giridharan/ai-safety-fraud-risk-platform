"""Tests for the data loader and data-quality checks.

These use tiny hand-built frames whose correct answers are known in advance,
so each check can be verified independently of the real dataset.
"""

import pandas as pd
import pytest

from src import data_validation as dv
from src.data_loader import DatasetNotFoundError, EXPECTED_COLUMNS, load_paysim


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 3, 800],
            "type": pd.Categorical(["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER", "PAYMENT"]),
            "amount": [100.0, 100.0, 50.0, 200.0, 0.0],
            "name_orig": ["C1", "C2", "C3", "C4", "C5"],
            # row 0: consistent; row 3: origin balance 0 -> 0 despite sending 200
            "old_balance_orig": [100.0, 300.0, 80.0, 0.0, 10.0],
            "new_balance_orig": [0.0, 200.0, 30.0, 0.0, 10.0],
            "name_dest": ["C9", "C8", "M1", "C7", "M2"],
            # row 0: dest 0 -> 0 despite receiving 100; row 1: consistent
            "old_balance_dest": [0.0, 50.0, 0.0, 10.0, 0.0],
            "new_balance_dest": [0.0, 150.0, 0.0, 210.0, 0.0],
            "is_fraud": [1, 0, 0, 1, 0],
            "is_flagged_fraud": [0, 0, 0, 0, 0],
        }
    )


def test_check_schema_valid(transactions):
    result = dv.check_schema(transactions)
    assert result["is_valid"]
    assert result["missing_columns"] == []


def test_check_schema_detects_missing_column(transactions):
    result = dv.check_schema(transactions.drop(columns="amount"))
    assert not result["is_valid"]
    assert result["missing_columns"] == ["amount"]


def test_missing_value_report_counts_nulls(transactions):
    transactions.loc[0, "amount"] = None
    report = dv.missing_value_report(transactions)
    assert report.loc["amount", "missing_count"] == 1
    assert report.loc["amount", "missing_pct"] == 20.0


def test_duplicate_report(transactions):
    with_dup = pd.concat([transactions, transactions.iloc[[0]]], ignore_index=True)
    assert dv.duplicate_report(transactions)["duplicate_rows"] == 0
    assert dv.duplicate_report(with_dup)["duplicate_rows"] == 1


def test_class_distribution_and_imbalance(transactions):
    dist = dv.class_distribution(transactions)
    assert dist.loc[0, "count"] == 3
    assert dist.loc[1, "count"] == 2
    assert dv.imbalance_ratio(transactions) == pytest.approx(1.5)


def test_imbalance_ratio_requires_two_classes(transactions):
    with pytest.raises(ValueError):
        dv.imbalance_ratio(transactions[transactions["is_fraud"] == 0])


def test_fraud_rate_by_type(transactions):
    report = dv.fraud_rate_by_type(transactions)
    assert report.loc["TRANSFER", "fraud_rate_pct"] == 100.0
    assert report.loc["PAYMENT", "fraud_rate_pct"] == 0.0
    assert report.loc["TRANSFER", "share_of_all_fraud_pct"] == 100.0


def test_impossible_value_report(transactions):
    transactions.loc[1, "amount"] = -5.0
    report = dv.impossible_value_report(transactions)["count"]
    assert report["negative_amount"] == 1
    assert report["zero_amount"] == 1
    assert report["step_out_of_range"] == 1  # step 800 > 744


def test_balance_consistency_report(transactions):
    report = dv.balance_consistency_report(transactions)
    transfer = report.loc["TRANSFER"]
    # TRANSFER rows: row 0 origin matches, row 3 origin (0 -> 0, amount 200) does not
    assert transfer["orig_change_matches_amount_pct"] == 50.0
    assert transfer["orig_zero_before_pct"] == 50.0
    # destinations: row 0 is 0 -> 0 (mismatch), row 3 changes by 200 (match)
    assert transfer["dest_change_matches_amount_pct"] == 50.0
    assert transfer["dest_zero_before_and_after_pct"] == 50.0
    # PAYMENT rows go to merchants only, so destination checks are undefined
    assert pd.isna(report.loc["PAYMENT", "dest_change_matches_amount_pct"])


def test_run_data_quality_checks_rejects_bad_schema(transactions):
    with pytest.raises(ValueError):
        dv.run_data_quality_checks(transactions.drop(columns="is_fraud"))


def test_load_paysim_missing_file_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_paysim(tmp_path / "does_not_exist.csv")


def test_load_paysim_renames_columns(tmp_path):
    csv = tmp_path / "paysim.csv"
    csv.write_text(
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,"
        "oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        "1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0\n"
    )
    df = load_paysim(csv)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.loc[0, "amount"] == pytest.approx(9839.64)
