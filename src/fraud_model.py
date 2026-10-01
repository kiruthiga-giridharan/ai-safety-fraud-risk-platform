"""Train, compare, select and serve fraud models.

Pipeline (``python -m src.fraud_model``):

1. Train Logistic Regression, Random Forest and XGBoost on both feature sets
   (training split only).
2. Compare them, and two rule baselines, on the validation split.
3. Select the deployment model by validation PR-AUC (``full`` feature set,
   matching the post-transaction-monitoring framing).
4. Derive the risk-level boundaries on validation (see src/evaluation.py).
5. Evaluate the selected model **once** on the test split.
6. Save a single bundle (model + features + boundaries + metrics + version).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import evaluation as ev
from src.data_loader import PROJECT_ROOT, TARGET_COLUMN
from src.feature_engineering import (
    FEATURE_SETS,
    MODEL_TYPES,
    RANDOM_STATE,
    SPLIT_PATHS,
    build_features,
)

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "fraud_model.joblib"
COMPARISON_PATH = MODELS_DIR / "fraud_model_comparison.json"
MODEL_ORDER = ("logistic_regression", "random_forest", "xgboost")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def make_logistic_regression(scale_pos_weight: float) -> Pipeline:
    """Interpretable baseline.

    Scaling is required because features are on different scales; L2
    regularisation (the default) stabilises the collinear origin features.
    ``class_weight="balanced"`` weights each fraud ~337x a legitimate row.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def make_random_forest(scale_pos_weight: float) -> RandomForestClassifier:
    """Bagged trees: captures non-linear interactions, robust to collinearity.

    ``min_samples_leaf=20`` limits tree size (memory, overfitting);
    ``balanced_subsample`` re-weights classes within each bootstrap sample.
    """
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """Boosted trees: usually the strongest model on tabular data.

    ``scale_pos_weight`` = negatives / positives in the training data, which is
    XGBoost's equivalent of class weighting.
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


MODEL_FACTORIES: dict[str, Callable[[float], Any]] = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}


def model_path(model_name: str, feature_set: str) -> Path:
    return MODELS_DIR / f"fraud_{model_name}_{feature_set}.joblib"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split_frame(name: str) -> pd.DataFrame:
    path = SPLIT_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.feature_engineering` first.")
    return pd.read_parquet(path)


def rule_baselines(split: pd.DataFrame) -> dict[str, np.ndarray]:
    """Binary predictions of the two rule baselines for a split.

    - existing_rule_flag: PaySim's ``is_flagged_fraud`` (looked up by transaction_id)
    - balance_drain_rule: "moves the entire origin balance" (notebook 01b, query 9)
    """
    from src.preprocessing import load_clean_transactions

    flags = load_clean_transactions()[["transaction_id", "is_flagged_fraud"]]
    flagged = split[["transaction_id"]].merge(flags, on="transaction_id", how="left")
    return {
        "existing_rule_flag": flagged["is_flagged_fraud"].to_numpy(),
        "balance_drain_rule": split["orig_fully_drained"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Training & comparison (Phase 4)
# ---------------------------------------------------------------------------

def train_and_compare(retrain: bool = True) -> pd.DataFrame:
    """Train every model on every feature set and compare on validation.

    Returns a comparison table and writes it to ``models/fraud_model_comparison.json``.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    train, validation = load_split_frame("train"), load_split_frame("validation")
    y_train, y_val = train[TARGET_COLUMN], validation[TARGET_COLUMN]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("Training rows: %s, scale_pos_weight=%.1f", f"{len(train):,}", scale_pos_weight)

    rows = []
    for name, preds in rule_baselines(validation).items():
        rows.append({"model": name, "feature_set": "rule", "train_seconds": None,
                     **ev.classification_metrics(y_val, preds, threshold=0.5)})

    for feature_set, features in FEATURE_SETS.items():
        for model_name in MODEL_ORDER:
            path = model_path(model_name, feature_set)
            start = time.perf_counter()
            trained = retrain or not path.is_file()
            if trained:
                model = MODEL_FACTORIES[model_name](scale_pos_weight)
                model.fit(train[features], y_train)
                joblib.dump(model, path, compress=3)
            else:
                model = joblib.load(path)
            # Only record a duration when we actually trained (loading time is not training time).
            seconds = time.perf_counter() - start if trained else float("nan")
            scores = model.predict_proba(validation[features])[:, 1]
            metrics = ev.classification_metrics(y_val, scores, threshold=0.5)
            rows.append({"model": model_name, "feature_set": feature_set,
                         "train_seconds": None if np.isnan(seconds) else round(seconds, 1), **metrics})
            logger.info("%-20s %-18s PR-AUC=%.4f recall=%.4f precision=%.4f (%.0fs)",
                        model_name, feature_set, metrics["pr_auc"], metrics["recall"],
                        metrics["precision"], seconds)

    comparison = pd.DataFrame(rows)
    COMPARISON_PATH.write_text(comparison.to_json(orient="records", indent=2))
    return comparison


# ---------------------------------------------------------------------------
# Selection, thresholds and final evaluation (Phase 5)
# ---------------------------------------------------------------------------

PR_AUC_TIE_TOLERANCE = 0.001


def select_model(comparison: pd.DataFrame, feature_set: str = "full") -> str:
    """Pick the deployment model for a feature set using validation results.

    Primary criterion: PR-AUC. Differences smaller than 0.001 are treated as a
    tie (with ~1,200 validation frauds they are within noise); ties are broken
    by F1 at the default 0.5 threshold.
    """
    candidates = comparison[comparison["feature_set"] == feature_set]
    best = candidates["pr_auc"].max()
    tied = candidates[candidates["pr_auc"] >= best - PR_AUC_TIE_TOLERANCE]
    return str(tied.sort_values("f1", ascending=False).iloc[0]["model"])


def finalise_model(comparison: pd.DataFrame, feature_set: str = "full") -> dict:
    """Choose boundaries on validation, evaluate once on test, and save the bundle."""
    model_name = select_model(comparison, feature_set)
    features = FEATURE_SETS[feature_set]
    model = joblib.load(model_path(model_name, feature_set))

    validation, test = load_split_frame("validation"), load_split_frame("test")
    val_scores = model.predict_proba(validation[features])[:, 1]
    boundaries = ev.choose_risk_boundaries(validation[TARGET_COLUMN], val_scores)

    test_scores = model.predict_proba(test[features])[:, 1]
    test_metrics = ev.classification_metrics(test[TARGET_COLUMN], test_scores, boundaries["medium"])
    test_metrics_high = ev.classification_metrics(test[TARGET_COLUMN], test_scores, boundaries["high"])

    version = f"fraud-{model_name}-{feature_set}-{datetime.now(timezone.utc):%Y%m%d}"
    bundle = {
        "model": model,
        "model_name": model_name,
        "feature_set": feature_set,
        "features": features,
        "model_types": list(MODEL_TYPES),
        "risk_boundaries": boundaries,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "validation_at_medium": ev.classification_metrics(validation[TARGET_COLUMN], val_scores, boundaries["medium"]),
            "test_at_medium": test_metrics,
            "test_at_high": test_metrics_high,
        },
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    (MODELS_DIR / "fraud_model_metadata.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2, default=float)
    )
    logger.info("Saved %s (%s); boundaries=%s", BUNDLE_PATH, version, boundaries)
    return bundle


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.fraud_model` first.")
    return joblib.load(path)


def score_transactions(bundle: dict, transactions: pd.DataFrame) -> np.ndarray:
    """Fraud scores for in-scope transactions (raises on out-of-scope types)."""
    out_of_scope = ~transactions["type"].isin(bundle["model_types"])
    if out_of_scope.any():
        raise ValueError(f"Types outside model scope: {sorted(transactions.loc[out_of_scope, 'type'].unique())}")
    features = build_features(transactions)[bundle["features"]]
    return bundle["model"].predict_proba(features)[:, 1]


def predict_transaction(bundle: dict, transaction: dict) -> dict:
    """Score one transaction and map it to a risk level.

    Out-of-scope types (CASH_IN, DEBIT, PAYMENT) are not scored by the model:
    they contained no fraud in the training data, so they return LOW with an
    explicit ``in_model_scope = False`` rather than a made-up probability.
    """
    if transaction["type"] not in bundle["model_types"]:
        return {
            "fraud_probability": None,
            "risk_level": "LOW",
            "in_model_scope": False,
            "model_version": bundle["version"],
        }
    score = float(score_transactions(bundle, pd.DataFrame([transaction]))[0])
    return {
        "fraud_probability": score,
        "risk_level": ev.assign_risk_level(score, bundle["risk_boundaries"]),
        "in_model_scope": True,
        "model_version": bundle["version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and finalise fraud models.")
    parser.add_argument("--reuse", action="store_true", help="Reuse saved models instead of retraining.")
    args = parser.parse_args()

    comparison = train_and_compare(retrain=not args.reuse)
    cols = ["model", "feature_set", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"]
    print(comparison[cols].round(4).to_string(index=False))
    bundle = finalise_model(comparison)
    print(f"\nDeployed: {bundle['version']}\nBoundaries: {bundle['risk_boundaries']}")
    print(json.dumps(bundle["metrics"]["test_at_medium"], indent=2, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    main()
