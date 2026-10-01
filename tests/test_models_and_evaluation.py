"""Tests for evaluation metrics, risk levels, fraud serving and SHAP explanations."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src import evaluation as ev
from src import explainability as xai
from src import fraud_model as fm
from src.feature_engineering import FEATURE_SETS, build_features

# --- evaluation --------------------------------------------------------------

Y = np.array([0, 0, 0, 0, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9])


def test_classification_metrics_counts():
    m = ev.classification_metrics(Y, S, threshold=0.5)
    # flagged: 0.8 (legit), 0.7 (fraud), 0.9 (fraud)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0
    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["fnr"] == 0.0
    assert "accuracy" not in m


def test_threshold_table_trades_recall_for_precision():
    table = ev.threshold_table(Y, S, thresholds=(0.5, 0.85))
    assert table["recall"].tolist() == [1.0, 0.5]
    assert table["precision"].tolist() == pytest.approx([2 / 3, 1.0])


def test_threshold_helpers():
    assert ev.threshold_for_precision(Y, S, 1.0) == pytest.approx(0.9)
    assert ev.threshold_for_recall(Y, S, 1.0) == pytest.approx(0.7)
    assert ev.threshold_for_precision(Y, S, 1.01) is None


def test_risk_boundaries_ordered_and_assigned():
    b = ev.choose_risk_boundaries(Y, S, high_min_precision=1.0, medium_min_precision=0.6)
    assert b["medium"] <= b["high"]
    assert ev.assign_risk_level(0.95, b) == "HIGH"
    assert ev.assign_risk_level(0.0, b) == "LOW"


def test_risk_boundaries_require_exactly_one_medium_target():
    with pytest.raises(ValueError):
        ev.choose_risk_boundaries(Y, S, medium_min_precision=0.5, medium_min_recall=0.9)


# --- fraud model serving and SHAP --------------------------------------------

def _tx(i: int, fraud: bool) -> dict:
    balance = 1000.0 + i
    return {
        "type": "TRANSFER" if i % 2 else "CASH_OUT",
        "amount": balance if fraud else 50.0 + i,
        "old_balance_orig": balance,
        "new_balance_orig": 0.0 if fraud else balance - 50.0 - i,
        "old_balance_dest": 0.0 if fraud else 500.0,
        "new_balance_dest": 0.0 if fraud else 550.0 + i,
    }


@pytest.fixture(scope="module")
def bundle() -> dict:
    rows = [_tx(i, fraud=i % 5 == 0) for i in range(200)]
    y = [int(i % 5 == 0) for i in range(200)]
    X = build_features(pd.DataFrame(rows))[FEATURE_SETS["full"]]
    model = XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1).fit(X, y)
    return {
        "model": model, "features": FEATURE_SETS["full"], "model_types": ["TRANSFER", "CASH_OUT"],
        "risk_boundaries": {"medium": 0.3, "high": 0.8}, "version": "test-v1",
    }


def test_select_model_breaks_pr_auc_ties_with_f1():
    comparison = pd.DataFrame([
        {"model": "a", "feature_set": "full", "pr_auc": 0.9968, "f1": 0.990},
        {"model": "b", "feature_set": "full", "pr_auc": 0.9967, "f1": 0.995},
        {"model": "c", "feature_set": "full", "pr_auc": 0.9900, "f1": 0.999},
    ])
    assert fm.select_model(comparison) == "b"


def test_predict_transaction_in_scope(bundle):
    out = fm.predict_transaction(bundle, _tx(0, fraud=True))
    assert out["in_model_scope"] is True
    assert 0.0 <= out["fraud_probability"] <= 1.0
    assert out["risk_level"] in ev.RISK_LEVELS
    assert out["model_version"] == "test-v1"


def test_predict_transaction_out_of_scope_has_no_probability(bundle):
    out = fm.predict_transaction(bundle, {**_tx(1, fraud=False), "type": "PAYMENT"})
    assert out == {"fraud_probability": None, "risk_level": "LOW",
                   "in_model_scope": False, "model_version": "test-v1"}


def test_shap_values_sum_to_model_output(bundle):
    tx = _tx(10, fraud=True)
    explanation = xai.explain_transaction(bundle, tx, top_n=len(bundle["features"]))
    X = build_features(pd.DataFrame([tx]))[bundle["features"]]
    raw_margin = float(bundle["model"].predict(X, output_margin=True)[0])
    total = explanation["base_value"] + sum(f["shap_value"] for f in explanation["top_factors"])
    assert total == pytest.approx(raw_margin, abs=1e-4)
    assert explanation["model_output_log_odds"] == pytest.approx(raw_margin, abs=1e-4)


def test_explanation_factors_are_ranked_and_described(bundle):
    factors = xai.explain_transaction(bundle, _tx(10, fraud=True), top_n=3)["top_factors"]
    magnitudes = [abs(f["shap_value"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for f in factors:
        assert f["label"] == xai.FEATURE_LABELS[f["feature"]]
        assert f["direction"] in {"increases risk", "decreases risk"}
        assert f["value_description"]
