"""Evaluation utilities for imbalanced binary classification.

Accuracy is deliberately absent: with ~0.3% fraud, predicting "legitimate" for
everything scores 99.7% accuracy and catches nothing. We report metrics that
focus on the fraud class instead.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


def classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict:
    """Threshold-dependent and threshold-free metrics for the positive (fraud) class.

    - precision: of the transactions we flag, how many are fraud
    - recall: of all fraud, how much we flag
    - FPR: share of legitimate transactions wrongly flagged (customer friction)
    - FNR: share of fraud missed (direct financial loss)
    - roc_auc: ranking quality over all thresholds (optimistic under heavy imbalance)
    - pr_auc: average precision; the headline metric for rare-event detection
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "alerts": int(tp + fp),
    }


def threshold_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Metrics at each candidate threshold, to show the precision/recall trade-off."""
    rows = [classification_metrics(y_true, y_score, t) for t in thresholds]
    cols = ["threshold", "precision", "recall", "f1", "fpr", "fnr", "tp", "fp", "fn", "alerts"]
    return pd.DataFrame(rows)[cols]


def threshold_for_precision(y_true, y_score, min_precision: float) -> Optional[float]:
    """Lowest threshold whose precision is >= ``min_precision`` (maximises recall at that precision)."""
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    # precision has one more element than thresholds; drop the final (recall=0) point
    ok = np.where(precision[:-1] >= min_precision)[0]
    return float(thresholds[ok[0]]) if len(ok) else None


def threshold_for_recall(y_true, y_score, min_recall: float) -> Optional[float]:
    """Highest threshold whose recall is >= ``min_recall`` (fewest alerts at that recall)."""
    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = np.where(recall[:-1] >= min_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else None


def choose_risk_boundaries(
    y_true,
    y_score,
    high_min_precision: float = 0.99,
    medium_min_precision: Optional[float] = 0.90,
    medium_min_recall: Optional[float] = None,
) -> dict:
    """Derive LOW / MEDIUM / HIGH score boundaries from validation data.

    The business targets are the inputs; the score boundaries are computed from
    data, never picked by hand.

    - HIGH starts at the lowest score where >= ``high_min_precision`` of alerts
      are positive: safe to act on automatically (e.g. hold the funds).
    - MEDIUM starts at either
        * the lowest score where alert precision is still >= ``medium_min_precision``
          (fraud: maximise caught fraud while analysts' queue stays >= 90% fraud), or
        * the highest score that still reaches ``medium_min_recall``
          (LLM safety: catch nearly everything; a review costs little).
    - Everything below MEDIUM is LOW.
    """
    if (medium_min_precision is None) == (medium_min_recall is None):
        raise ValueError("Set exactly one of medium_min_precision / medium_min_recall.")
    high = threshold_for_precision(y_true, y_score, high_min_precision)
    if medium_min_precision is not None:
        medium = threshold_for_precision(y_true, y_score, medium_min_precision)
    else:
        medium = threshold_for_recall(y_true, y_score, medium_min_recall)
    if high is None or medium is None:
        raise ValueError("Targets not achievable on this data; relax them.")
    medium = min(medium, high)  # MEDIUM must sit below HIGH
    targets = {"high_min_precision": high_min_precision}
    targets.update({"medium_min_precision": medium_min_precision} if medium_min_precision is not None
                   else {"medium_min_recall": medium_min_recall})
    return {"medium": medium, "high": high, "targets": targets}


def assign_risk_level(score: float, boundaries: dict) -> str:
    """Map a model score to LOW / MEDIUM / HIGH using validated boundaries."""
    if score >= boundaries["high"]:
        return "HIGH"
    if score >= boundaries["medium"]:
        return "MEDIUM"
    return "LOW"


def risk_level_table(y_true, y_score, boundaries: dict) -> pd.DataFrame:
    """Transactions, fraud and fraud rate within each risk level."""
    levels = pd.Series([assign_risk_level(s, boundaries) for s in y_score], name="risk_level")
    df = pd.DataFrame({"risk_level": levels, "is_fraud": np.asarray(y_true)})
    table = df.groupby("risk_level")["is_fraud"].agg(transactions="size", fraud="sum")
    table = table.reindex(RISK_LEVELS, fill_value=0)
    table["fraud_rate_pct"] = (table["fraud"] / table["transactions"].replace(0, np.nan) * 100).round(3)
    table["share_of_all_fraud_pct"] = (table["fraud"] / max(table["fraud"].sum(), 1) * 100).round(2)
    return table


def calibration_table(y_true, y_score, bins: Iterable[float] = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)) -> pd.DataFrame:
    """Mean predicted score vs observed fraud rate per score bin.

    Class weighting pushes scores upward, so a score of 0.8 need not mean an
    80% chance of fraud. This table shows how far scores are from probabilities.
    """
    df = pd.DataFrame({"score": np.asarray(y_score), "y": np.asarray(y_true)})
    df["bin"] = pd.cut(df["score"], bins=list(bins), right=False)
    return df.groupby("bin", observed=True).agg(
        transactions=("y", "size"), mean_score=("score", "mean"), observed_fraud_rate=("y", "mean")
    )
