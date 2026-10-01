"""SHAP explanations for the deployed fraud model.

SHAP assigns each feature a contribution to one prediction, measured in the
model's log-odds space: contributions plus the base value sum exactly to the
model's raw output. Positive = pushes towards fraud, negative = away.

Every "reason" produced here is generated from (a) the SHAP value the model
actually assigned and (b) the transaction's real field values. Nothing is
inferred or invented, which is what the investigation assistant relies on.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import shap

from src.feature_engineering import build_features

# Plain-English names for each feature, used in explanations and charts.
FEATURE_LABELS = {
    "log_amount": "Transaction amount",
    "is_transfer": "Transaction type",
    "log_old_balance_orig": "Sender balance before",
    "log_new_balance_orig": "Sender balance after",
    "orig_balance_error": "Sender balance inconsistency",
    "log_amount_to_orig_balance": "Amount relative to sender balance",
    "orig_zero_before": "Sender started with zero balance",
    "amount_exceeds_orig_balance": "Amount exceeds sender balance",
    "orig_fully_drained": "Sender account fully drained",
    "log_old_balance_dest": "Recipient balance before",
    "log_new_balance_dest": "Recipient balance after",
    "dest_balance_error": "Recipient balance inconsistency",
    "dest_zero_before_and_after": "Recipient balance stayed at zero",
}


def _money(x: float) -> str:
    return f"{x:,.2f}"


def describe_feature(feature: str, tx: dict) -> str:
    """Factual description of a feature's value, built from raw transaction fields."""
    amount, old_o, new_o = tx["amount"], tx["old_balance_orig"], tx["new_balance_orig"]
    old_d, new_d = tx["old_balance_dest"], tx["new_balance_dest"]
    yes_no = lambda flag: "yes" if flag else "no"
    descriptions = {
        "log_amount": f"amount {_money(amount)}",
        "is_transfer": f"type {tx['type']}",
        "log_old_balance_orig": f"sender balance before {_money(old_o)}",
        "log_new_balance_orig": f"sender balance after {_money(new_o)}",
        "orig_balance_error": f"sender balance changed by {_money(old_o - new_o)} vs amount {_money(amount)}",
        "log_amount_to_orig_balance": f"amount {_money(amount)} vs sender balance {_money(old_o)}",
        "orig_zero_before": f"sender balance before was zero: {yes_no(old_o == 0)}",
        "amount_exceeds_orig_balance": f"amount exceeds sender balance: {yes_no(amount > old_o + 0.01)}",
        "orig_fully_drained": f"entire sender balance moved: {yes_no(old_o > 0 and abs(amount - old_o) <= 0.01)}",
        "log_old_balance_dest": f"recipient balance before {_money(old_d)}",
        "log_new_balance_dest": f"recipient balance after {_money(new_d)}",
        "dest_balance_error": f"recipient balance changed by {_money(new_d - old_d)} vs amount {_money(amount)}",
        "dest_zero_before_and_after": f"recipient balance 0 before and after: {yes_no(old_d == 0 and new_d == 0)}",
    }
    return descriptions[feature]


def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


@lru_cache(maxsize=4)
def _tree_explainer(model_id: int, model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def get_explainer(bundle: dict, background: Optional[pd.DataFrame] = None):
    """SHAP explainer for the bundle's model (tree models use the exact TreeExplainer)."""
    estimator = _final_estimator(bundle["model"])
    if hasattr(estimator, "estimators_") or estimator.__class__.__name__.startswith("XGB"):
        return _tree_explainer(id(estimator), estimator)
    if background is None:
        raise ValueError("Linear models need a background sample for SHAP.")
    return shap.LinearExplainer(bundle["model"].steps[-1][1],
                                bundle["model"][:-1].transform(background))


def shap_values(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values (rows x features) for the positive class, plus the base value."""
    explainer = get_explainer(bundle)
    values = explainer.shap_values(X)
    base = explainer.expected_value
    if isinstance(values, list):          # older RF API: one array per class
        values, base = values[1], base[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # newer RF API: (rows, features, classes)
        values, base = values[:, :, 1], np.asarray(base)[1]
    return values, float(np.ravel(base)[0])


def global_importance(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature over a sample: how much each feature moves predictions on average."""
    values, _ = shap_values(bundle, X[bundle["features"]])
    return (
        pd.DataFrame({"feature": bundle["features"], "mean_abs_shap": np.abs(values).mean(axis=0)})
        .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_transaction(bundle: dict, transaction: dict, top_n: int = 5) -> dict:
    """Top contributing factors for one in-scope transaction.

    Returns the base value, the model's raw log-odds output, and the ``top_n``
    features ranked by |SHAP|, each with a factual description.
    """
    features = build_features(pd.DataFrame([transaction]))[bundle["features"]]
    values, base = shap_values(bundle, features)
    contributions = values[0]

    order = np.argsort(-np.abs(contributions))[:top_n]
    factors = [
        {
            "feature": bundle["features"][i],
            "label": FEATURE_LABELS[bundle["features"][i]],
            "value_description": describe_feature(bundle["features"][i], transaction),
            "shap_value": float(contributions[i]),
            "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
        }
        for i in order
    ]
    return {
        "base_value": base,
        "model_output_log_odds": float(base + contributions.sum()),
        "top_factors": factors,
    }
