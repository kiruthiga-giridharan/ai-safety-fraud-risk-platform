"""Build the small, committed ``demo/`` folder used by the hosted dashboard.

The full dashboard reads a 1 GB SQLite database and git-ignored model files,
which a free hosting platform cannot hold. This script copies the deployed
models and precomputes everything else the dashboard shows, from the same
pipeline outputs, so the hosted demo displays identical numbers.

Run after the full pipeline (README "How to Run"):

    python -m src.demo_artifacts
"""

from __future__ import annotations

import json
import logging
import shutil

import pandas as pd

from src import database as db
from src import evaluation as ev
from src import explainability as xai
from src import fraud_model as fm
from src import safety_model as sm
from src.data_loader import PROJECT_ROOT
from src.preprocessing import load_clean_transactions

logger = logging.getLogger(__name__)

DEMO_DIR = PROJECT_ROOT / "demo"

# Named SQL queries the dashboard displays (results are small aggregates).
DEMO_QUERIES = [
    "dataset_overview", "fraud_rate_by_type", "fraud_rate_by_amount_band", "fraud_by_day",
    "full_balance_drain_rule", "existing_rule_performance", "suspicious_destination_accounts",
]

# Transactions available for lookup in the demo: the documented example cases
# plus a random sample of held-out *test* transactions.
EXAMPLE_TRANSACTION_IDS = [6259966, 3928776, 408956, 3]
SAMPLE_FRAUD, SAMPLE_LEGIT, SEED = 300, 700, 42


def _scores(bundle: dict, split: str) -> tuple:
    frame = fm.load_split_frame(split)
    return bundle["model"].predict_proba(frame[bundle["features"]])[:, 1], frame["is_fraud"].to_numpy(), frame


def build_demo_artifacts() -> dict:
    """Write demo/ and return a summary of what was written."""
    (DEMO_DIR / "sql").mkdir(parents=True, exist_ok=True)

    for src_path in (fm.BUNDLE_PATH, fm.COMPARISON_PATH, sm.SAFETY_BUNDLE_PATH, sm.SAFETY_METRICS_PATH):
        shutil.copy2(src_path, DEMO_DIR / src_path.name)

    conn = db.get_connection()
    try:
        queries = db.load_named_queries()
        for name in DEMO_QUERIES:
            db.run_query(conn, queries[name]).to_json(DEMO_DIR / "sql" / f"{name}.json", orient="records", indent=1)
    finally:
        conn.close()

    bundle = fm.load_bundle()
    val_scores, y_val, validation = _scores(bundle, "validation")
    test_scores, y_test, test = _scores(bundle, "test")
    ev.threshold_table(y_val, val_scores).to_json(DEMO_DIR / "thresholds_validation.json", orient="records", indent=1)
    ev.risk_level_table(y_test, test_scores, bundle["risk_boundaries"]).reset_index().to_json(
        DEMO_DIR / "risk_levels_test.json", orient="records", indent=1)

    sample = pd.concat([validation[validation["is_fraud"] == 1],
                        validation[validation["is_fraud"] == 0].sample(20_000, random_state=SEED)])
    xai.global_importance(bundle, sample).to_json(DEMO_DIR / "shap_global_importance.json", orient="records", indent=1)

    clean = load_clean_transactions()
    picked = pd.concat([
        test[test["is_fraud"] == 1].sample(SAMPLE_FRAUD, random_state=SEED)["transaction_id"],
        test[test["is_fraud"] == 0].sample(SAMPLE_LEGIT, random_state=SEED)["transaction_id"],
        pd.Series(EXAMPLE_TRANSACTION_IDS),
    ]).drop_duplicates()
    transactions = clean[clean["transaction_id"].isin(picked)].reset_index(drop=True)
    transactions.to_parquet(DEMO_DIR / "transactions_sample.parquet", index=False)

    summary = {
        "sql_queries": DEMO_QUERIES,
        "sample_transactions": len(transactions),
        "sample_fraud": int(transactions["is_fraud"].sum()),
        "fraud_model": bundle["version"],
        "safety_model": sm.load_safety_bundle()["version"],
    }
    (DEMO_DIR / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    logger.info("Demo artefacts written to %s: %s", DEMO_DIR, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print(json.dumps(build_demo_artifacts(), indent=2))
