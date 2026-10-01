"""Tests for fraud feature engineering and the train/validation/test split."""

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as fe


def make_tx(**overrides) -> dict:
    row = {
        "transaction_id": 1, "step": 1, "type": "TRANSFER", "amount": 100.0,
        "name_orig": "C1", "old_balance_orig": 100.0, "new_balance_orig": 0.0,
        "name_dest": "C2", "old_balance_dest": 50.0, "new_balance_dest": 150.0,
        "is_fraud": 0, "is_flagged_fraud": 0,
    }
    row.update(overrides)
    return row


def features_for(**overrides) -> pd.Series:
    return fe.build_features(pd.DataFrame([make_tx(**overrides)])).iloc[0]


# --- individual features -----------------------------------------------------

def test_consistent_full_drain():
    f = features_for()  # 100 -> 0 sending 100; dest 50 -> 150
    assert f["orig_balance_error"] == 0
    assert f["dest_balance_error"] == 0
    assert f["orig_fully_drained"] == 1
    assert f["log_amount_to_orig_balance"] == pytest.approx(0.0)
    assert f["amount_exceeds_orig_balance"] == 0
    assert f["is_transfer"] == 1


def test_overspend_floored_at_zero():
    f = features_for(type="CASH_OUT", amount=500.0, old_balance_orig=100.0, new_balance_orig=0.0)
    assert f["amount_exceeds_orig_balance"] == 1
    assert f["orig_fully_drained"] == 0
    assert f["orig_balance_error"] < 0          # 100 - 500 - 0 = -400
    assert f["log_amount_to_orig_balance"] > 0
    assert f["is_transfer"] == 0


def test_zero_balance_flags():
    f = features_for(old_balance_orig=0.0, new_balance_orig=0.0, old_balance_dest=0.0, new_balance_dest=0.0)
    assert f["orig_zero_before"] == 1
    assert f["orig_fully_drained"] == 0          # nothing to drain
    assert f["dest_zero_before_and_after"] == 1
    assert f["dest_balance_error"] > 0           # 0 + 100 - 0 = +100 missing at destination


def test_signed_log1p_keeps_sign_and_zero():
    out = fe.signed_log1p(pd.Series([-np.e + 1, 0.0, np.e - 1]))
    assert out.tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_feature_columns_match_full_set():
    out = fe.build_features(pd.DataFrame([make_tx()]))
    assert list(out.columns) == fe.FEATURE_SETS["full"]


def test_no_nan_or_inf_on_edge_values():
    rows = [make_tx(amount=0.01, old_balance_orig=0.0, new_balance_orig=0.0),
            make_tx(amount=9.2e7, old_balance_orig=0.0, new_balance_orig=0.0,
                    old_balance_dest=3.5e8, new_balance_dest=0.0)]
    out = fe.build_features(pd.DataFrame(rows))
    assert np.isfinite(out.to_numpy(dtype=float)).all()


# --- leakage guards ----------------------------------------------------------

@pytest.mark.parametrize("feature_set", list(fe.FEATURE_SETS))
def test_excluded_columns_never_in_feature_sets(feature_set):
    assert not set(fe.EXCLUDED_COLUMNS) & set(fe.FEATURE_SETS[feature_set])


def test_no_origin_balance_set_has_no_origin_features():
    assert not set(fe.ORIGIN_BALANCE_FEATURES) & set(fe.FEATURE_SETS["no_origin_balance"])


# --- scope and split ---------------------------------------------------------

@pytest.fixture
def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    rows = [
        make_tx(transaction_id=i, type=rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"]),
                is_fraud=int(i % 50 == 0))
        for i in range(n)
    ]
    return fe.build_model_table(pd.DataFrame(rows))


def test_model_table_only_in_scope_types(table):
    # PAYMENT rows were dropped, and identifiers/target are carried through
    assert table["transaction_id"].is_unique
    assert {"transaction_id", "is_fraud"} <= set(table.columns)
    assert len(table) < 2000


def test_split_is_disjoint_complete_and_stratified(table):
    splits = fe.split_train_validation_test(table)
    ids = [set(part["transaction_id"]) for part in splits.values()]
    assert sum(map(len, ids)) == len(table)
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    overall = table["is_fraud"].mean()
    for part in splits.values():
        assert part["is_fraud"].mean() == pytest.approx(overall, abs=0.01)


def test_split_is_reproducible(table):
    a = fe.split_train_validation_test(table)["test"]["transaction_id"]
    b = fe.split_train_validation_test(table)["test"]["transaction_id"]
    pd.testing.assert_series_equal(a, b)
