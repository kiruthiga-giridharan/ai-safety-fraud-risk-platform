"""Tests for cleaning rules, the SQLite loader and the named-query runner."""

import sqlite3

import pandas as pd
import pytest

from src import database as db
from src.preprocessing import (
    add_transaction_id,
    clean_transactions,
    format_transaction_id,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 2],
            "type": ["TRANSFER", "CASH_OUT", "CASH_OUT", "PAYMENT"],
            "amount": [500.0, 500.0, 0.0, 20.0],
            "name_orig": ["C1", "C2", "C3", "C4"],
            "old_balance_orig": [500.0, 900.0, 0.0, 100.0],
            "new_balance_orig": [0.0, 400.0, 0.0, 80.0],
            "name_dest": ["C2", "C9", "C8", "M1"],
            "old_balance_dest": [0.0, 0.0, 0.0, 0.0],
            "new_balance_dest": [0.0, 500.0, 0.0, 0.0],
            "is_fraud": [1, 1, 1, 0],
            "is_flagged_fraud": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def clean(raw) -> pd.DataFrame:
    cleaned, _, _ = clean_transactions(add_transaction_id(raw))
    return cleaned


@pytest.fixture
def conn(clean):
    connection = sqlite3.connect(":memory:")
    db.load_transactions_to_sqlite(clean, connection)
    yield connection
    connection.close()


# --- preprocessing -----------------------------------------------------------

def test_transaction_id_is_one_based_row_position(raw):
    ids = add_transaction_id(raw)["transaction_id"].tolist()
    assert ids == [1, 2, 3, 4]


def test_format_transaction_id():
    assert format_transaction_id(42) == "TX-0000042"


def test_clean_removes_only_zero_amount_rows(raw):
    cleaned, excluded, report = clean_transactions(add_transaction_id(raw))
    assert report.input_rows == 4
    assert report.output_rows == 3
    assert report.excluded == {"zero_amount": 1}
    assert excluded["transaction_id"].tolist() == [3]
    assert (excluded["reason"] == "zero_amount").all()
    # IDs still point at the original raw rows after removal
    assert cleaned["transaction_id"].tolist() == [1, 2, 4]


def test_clean_does_not_modify_kept_rows(raw):
    with_ids = add_transaction_id(raw)
    cleaned, _, _ = clean_transactions(with_ids)
    expected = with_ids[with_ids["amount"] != 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, expected)


def test_clean_requires_transaction_id(raw):
    with pytest.raises(ValueError, match="add_transaction_id"):
        clean_transactions(raw)


# --- database ----------------------------------------------------------------

def test_load_to_sqlite_row_count(conn, clean):
    (n,) = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    assert n == len(clean)


def test_schema_rejects_zero_amount(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO transactions VALUES "
            "(99, 1, 'CASH_OUT', 0, 'C1', 0, 0, 'C2', 0, 0, 1, 0)"
        )


def test_parse_named_queries():
    text = """
-- name: first
-- question: What is one?
SELECT 1 AS one;

-- name: second
SELECT 2 AS two -- inline comment survives
;
"""
    queries = db.parse_named_queries(text)
    assert list(queries) == ["first", "second"]
    assert queries["first"].question == "What is one?"
    assert queries["first"].sql == "SELECT 1 AS one"
    assert queries["second"].question == ""


def test_parse_named_queries_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        db.parse_named_queries("-- name: a\nSELECT 1;\n-- name: a\nSELECT 2;")


def test_every_project_query_runs(conn):
    """Each query in sql/*.sql must be valid SQL against the real schema."""
    queries = db.load_named_queries()
    assert len(queries) >= 10
    for name, query in queries.items():
        params = {"account": "C2"} if ":account" in query.sql else None
        result = db.run_query(conn, query, params)
        assert isinstance(result, pd.DataFrame), name
        assert query.question, f"{name} is missing a '-- question:' header"


def test_fraud_rate_by_type_values(conn):
    result = db.run_query(conn, db.load_named_queries()["fraud_rate_by_type"]).set_index("type")
    assert result.loc["TRANSFER", "fraud_rate_pct"] == 100.0
    assert result.loc["PAYMENT", "fraud"] == 0


def test_account_activity_finds_both_directions(conn):
    result = db.run_query(conn, db.load_named_queries()["account_activity"], {"account": "C2"})
    assert sorted(result["direction"]) == ["received", "sent"]
