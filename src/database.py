"""SQLite storage for cleaned transactions and a runner for named SQL queries.

SQL lives in ``sql/*.sql`` files, not in Python strings. Each query is preceded
by a header in the form

    -- name: fraud_rate_by_type
    -- question: Which transaction types have the highest fraud rate?
    SELECT ...;

so the notebook, dashboard and API can all run the same query by name.

Run as a script to (re)build the database from the cleaned Parquet file:

    python -m src.database
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_loader import PROCESSED_DATA_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("FRAUD_DB_PATH", PROCESSED_DATA_DIR / "paysim.db"))
SQL_DIR = PROJECT_ROOT / "sql"
TABLE_NAME = "transactions"

# Explicit schema rather than letting pandas guess: proper types, a primary
# key, and CHECK constraints that encode the validation rules in the database.
CREATE_TABLE_SQL = f"""
CREATE TABLE {TABLE_NAME} (
    transaction_id    INTEGER PRIMARY KEY,
    step              INTEGER NOT NULL CHECK (step BETWEEN 1 AND 744),
    type              TEXT    NOT NULL,
    amount            REAL    NOT NULL CHECK (amount > 0),
    name_orig         TEXT    NOT NULL,
    old_balance_orig  REAL    NOT NULL CHECK (old_balance_orig >= 0),
    new_balance_orig  REAL    NOT NULL CHECK (new_balance_orig >= 0),
    name_dest         TEXT    NOT NULL,
    old_balance_dest  REAL    NOT NULL CHECK (old_balance_dest >= 0),
    new_balance_dest  REAL    NOT NULL CHECK (new_balance_dest >= 0),
    is_fraud          INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud  INTEGER NOT NULL CHECK (is_flagged_fraud IN (0, 1))
)
"""

# Indexes chosen for the access patterns in sql/*.sql: grouping by type/step,
# looking up accounts, and joining TRANSFERs to CASH_OUTs on (step, amount).
INDEX_SQL = [
    f"CREATE INDEX idx_type_fraud ON {TABLE_NAME} (type, is_fraud)",
    f"CREATE INDEX idx_step ON {TABLE_NAME} (step)",
    f"CREATE INDEX idx_name_orig ON {TABLE_NAME} (name_orig)",
    f"CREATE INDEX idx_name_dest ON {TABLE_NAME} (name_dest)",
    f"CREATE INDEX idx_type_step_amount ON {TABLE_NAME} (type, step, amount)",
]


@dataclass(frozen=True)
class NamedQuery:
    name: str
    question: str
    sql: str


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def load_transactions_to_sqlite(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    chunksize: int = 200_000,
) -> int:
    """Replace the transactions table with ``df`` and build indexes.

    Indexes are created *after* the bulk insert, which is much faster than
    maintaining them row by row.
    """
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn.execute(CREATE_TABLE_SQL)

    columns = [
        "transaction_id", "step", "type", "amount", "name_orig", "old_balance_orig",
        "new_balance_orig", "name_dest", "old_balance_dest", "new_balance_dest",
        "is_fraud", "is_flagged_fraud",
    ]
    to_load = df[columns].astype({"type": "string"})
    to_load.to_sql(TABLE_NAME, conn, if_exists="append", index=False, chunksize=chunksize)

    for statement in INDEX_SQL:
        conn.execute(statement)
    conn.execute("ANALYZE")  # refresh query-planner statistics
    conn.commit()

    (n_rows,) = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
    logger.info("Loaded %s rows into %s", f"{n_rows:,}", TABLE_NAME)
    return n_rows


_HEADER = re.compile(r"^--\s*name:\s*(?P<name>\w+)\s*$", re.MULTILINE)
_QUESTION = re.compile(r"^--\s*question:\s*(?P<question>.+)$", re.MULTILINE)


def parse_named_queries(text: str) -> dict[str, NamedQuery]:
    """Split a .sql file into named queries using ``-- name:`` headers."""
    headers = list(_HEADER.finditer(text))
    queries: dict[str, NamedQuery] = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end():end]
        question_match = _QUESTION.search(body)
        sql = "\n".join(
            line for line in body.strip().splitlines() if not line.lstrip().startswith("--")
        ).strip().rstrip(";")
        name = header.group("name")
        if name in queries:
            raise ValueError(f"Duplicate query name: {name}")
        if not sql:
            raise ValueError(f"Query '{name}' has no SQL body")
        queries[name] = NamedQuery(
            name=name,
            question=question_match.group("question").strip() if question_match else "",
            sql=sql,
        )
    return queries


def load_named_queries(sql_dir: Path = SQL_DIR) -> dict[str, NamedQuery]:
    """Load every named query from all .sql files in ``sql_dir``."""
    queries: dict[str, NamedQuery] = {}
    for path in sorted(sql_dir.glob("*.sql")):
        for name, query in parse_named_queries(path.read_text()).items():
            if name in queries:
                raise ValueError(f"Query '{name}' is defined in more than one file")
            queries[name] = query
    return queries


def run_query(
    conn: sqlite3.Connection,
    query: NamedQuery | str,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """Run a NamedQuery (or raw SQL string) and return the result as a DataFrame."""
    sql = query.sql if isinstance(query, NamedQuery) else query
    return pd.read_sql_query(sql, conn, params=params)


def build_database(db_path: Path = DB_PATH) -> int:
    """Build the SQLite database from the cleaned Parquet file."""
    from src.preprocessing import load_clean_transactions

    df = load_clean_transactions()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)  # sqlite3's own context manager commits but never closes
    try:
        return load_transactions_to_sqlite(df, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    build_database()
