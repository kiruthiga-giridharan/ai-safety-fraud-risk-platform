"""LLM prompt-safety classifier: SAFE / PROMPT_INJECTION / JAILBREAK.

Data (both Apache-2.0, downloaded from the authors' Hugging Face repositories
into ``data/raw/llm_safety/``):

- deepset/prompt-injections: label 1 = prompt injection, 0 = benign.
  Short prompts, English and German.
- jackhhao/jailbreak-classification: "jailbreak" vs "benign".
  Mostly long role-play / "DAN"-style jailbreak prompts.

Categories we deliberately do NOT predict: PROMPT_EXTRACTION and SUSPICIOUS.
Neither dataset labels them, and inventing labels would mean fabricating
training data. "Suspicious" is instead expressed through the REVIEW action band.

Run ``python -m src.safety_model`` to train, evaluate and save the model.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data_loader import RAW_DATA_DIR
from src import evaluation as ev

logger = logging.getLogger(__name__)

SAFETY_RAW_DIR = RAW_DATA_DIR / "llm_safety"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SAFETY_BUNDLE_PATH = MODELS_DIR / "safety_model.joblib"
SAFETY_METRICS_PATH = MODELS_DIR / "safety_model_metrics.json"

LABELS = ["SAFE", "PROMPT_INJECTION", "JAILBREAK"]
RANDOM_STATE = 42
ACTIONS = ("ALLOW", "REVIEW", "FLAG")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Whitespace/case normalisation used only for de-duplication."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_safety_data(raw_dir: Path = SAFETY_RAW_DIR) -> pd.DataFrame:
    """Combine both datasets into one frame with columns text, label, source, split."""
    files = {
        "deepset_train.parquet": ("deepset", "train"),
        "deepset_test.parquet": ("deepset", "test"),
        "jackhhao_train.csv": ("jackhhao", "train"),
        "jackhhao_test.csv": ("jackhhao", "test"),
    }
    missing = [f for f in files if not (raw_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing safety data files in {raw_dir}: {missing} (see README).")

    parts = []
    for filename, (source, split) in files.items():
        path = raw_dir / filename
        if source == "deepset":
            df = pd.read_parquet(path)
            df["label"] = df["label"].map({0: "SAFE", 1: "PROMPT_INJECTION"})
        else:
            df = pd.read_csv(path).rename(columns={"prompt": "text"})
            df["label"] = df["type"].map({"benign": "SAFE", "jailbreak": "JAILBREAK"})
        parts.append(df[["text", "label"]].assign(source=source, split=split))
    data = pd.concat(parts, ignore_index=True)
    if data["label"].isna().any():
        raise ValueError("Unexpected label values in safety data")
    return data


def deduplicate(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove empty texts, exact duplicates, and test texts that also appear in train.

    A test prompt that is also a training prompt would be scored on memory, not
    generalisation, so it is removed from test (train/test leakage).
    """
    report = {"input_rows": len(data)}
    data = data.assign(_norm=data["text"].map(normalise_text))
    data = data[data["_norm"].str.len() > 0]
    report["empty_removed"] = report["input_rows"] - len(data)

    conflicting = data.groupby("_norm")["label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    report["conflicting_label_texts_removed"] = int(data["_norm"].isin(conflicting).sum())
    data = data[~data["_norm"].isin(conflicting)]

    before = len(data)
    data = data.sort_values("split", ascending=False)  # keep the 'train' copy first
    data = data.drop_duplicates(subset=["_norm", "split"])
    report["within_split_duplicates_removed"] = before - len(data)

    train_texts = set(data.loc[data["split"] == "train", "_norm"])
    leak = (data["split"] == "test") & data["_norm"].isin(train_texts)
    report["test_rows_also_in_train_removed"] = int(leak.sum())
    data = data[~leak].drop(columns="_norm").reset_index(drop=True)
    report["output_rows"] = len(data)
    return data, report


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_baseline() -> Pipeline:
    """TF-IDF word unigrams+bigrams -> logistic regression (the required baseline)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=10.0)),
    ])


def make_word_char_model() -> Pipeline:
    """Word + character n-grams -> logistic regression.

    Character n-grams (within word boundaries) are robust to the tricks attacks
    use: misspellings, spacing, leetspeak, and mixed English/German text.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=200_000)),
    ])
    return Pipeline([
        ("features", features),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=10.0)),
    ])


def make_length_only_model() -> Pipeline:
    """Diagnostic, not a candidate: predicts the label from text length alone.

    If this scores well, the classes are separable by a dataset artefact
    (injections are short, jailbreaks are long) and text models may be
    partly learning that artefact.
    """
    from sklearn.preprocessing import FunctionTransformer

    def length_features(texts):
        lengths = np.array([len(str(t)) for t in texts], dtype=float)
        return np.column_stack([np.log1p(lengths)])

    return Pipeline([
        ("length", FunctionTransformer(length_features)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


CANDIDATES = {
    "tfidf_word_logreg": make_baseline,
    "tfidf_word_char_logreg": make_word_char_model,
}


# ---------------------------------------------------------------------------
# Scoring and actions
# ---------------------------------------------------------------------------

def risk_scores(model: Pipeline, texts) -> np.ndarray:
    """Risk = 1 - P(SAFE): the probability mass on any unsafe class."""
    proba = model.predict_proba(list(texts))
    safe_idx = list(model.classes_).index("SAFE")
    return 1.0 - proba[:, safe_idx]


def choose_action_thresholds(is_unsafe, scores, flag_min_precision=0.95, review_min_recall=0.95) -> dict:
    """FLAG / REVIEW boundaries from out-of-fold training predictions.

    Same method as fraud risk levels: FLAG where >= 95% of flagged prompts are
    unsafe; REVIEW down to the score that still catches >= 95% of unsafe prompts.
    """
    bounds = ev.choose_risk_boundaries(is_unsafe, scores, high_min_precision=flag_min_precision,
                                       medium_min_precision=None, medium_min_recall=review_min_recall)
    return {"review": bounds["medium"], "flag": bounds["high"], "targets": bounds["targets"]}


def action_for(score: float, thresholds: dict) -> str:
    if score >= thresholds["flag"]:
        return "FLAG"
    if score >= thresholds["review"]:
        return "REVIEW"
    return "ALLOW"


def analyse_prompt(bundle: dict, prompt: str) -> dict:
    """Classify one prompt and recommend an action."""
    model = bundle["model"]
    proba = model.predict_proba([prompt])[0]
    classes = list(model.classes_)
    score = float(1.0 - proba[classes.index("SAFE")])
    return {
        "classification": classes[int(np.argmax(proba))],
        "risk_score": score,
        "action": action_for(score, bundle["action_thresholds"]),
        "class_probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "model_version": bundle["version"],
    }


def load_safety_bundle(path: Path = SAFETY_BUNDLE_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.safety_model` first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def multiclass_report(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def train_and_evaluate() -> dict:
    """Cross-validate candidates on train, pick one, set thresholds, test once, save."""
    data, dedup_report = deduplicate(load_safety_data())
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {"dedup": dedup_report, "cv": {}, "diagnostics": {}}
    oof = {}
    for name, factory in {**CANDIDATES, "length_only": make_length_only_model}.items():
        pred = cross_val_predict(factory(), train["text"], train["label"], cv=cv, method="predict_proba")
        classes = sorted(LABELS)  # cross_val_predict orders columns by sorted class label
        labels = np.array(classes)[pred.argmax(axis=1)]
        target = results["diagnostics"] if name == "length_only" else results["cv"]
        target[name] = multiclass_report(train["label"], labels)
        oof[name] = (pred, classes)
        logger.info("CV %-24s macro-F1 = %.4f", name, target[name]["macro_f1"])

    best = max(CANDIDATES, key=lambda n: results["cv"][n]["macro_f1"])
    pred, classes = oof[best]
    oof_risk = 1.0 - pred[:, classes.index("SAFE")]
    thresholds = choose_action_thresholds((train["label"] != "SAFE").astype(int), oof_risk)

    model = CANDIDATES[best]().fit(train["text"], train["label"])
    test_pred = model.predict(test["text"])
    test_risk = risk_scores(model, test["text"])
    results["selected_model"] = best
    results["action_thresholds"] = thresholds
    results["test"] = multiclass_report(test["label"], test_pred)
    # Within one source, both classes share the same writing style, so this shows
    # whether the model learned *content* rather than "which dataset is this from".
    # Macro-F1 is averaged over the classes present in that source only.
    results["test"]["by_source"] = {}
    for src in ("deepset", "jackhhao"):
        mask = (test["source"] == src).to_numpy()
        y_src = test.loc[mask, "label"]
        results["test"]["by_source"][src] = {
            "rows": int(mask.sum()),
            "macro_f1_on_present_classes": f1_score(y_src, test_pred[mask], labels=sorted(y_src.unique()), average="macro"),
            "per_class_recall": {lab: float((test_pred[mask][y_src.to_numpy() == lab] == lab).mean())
                                 for lab in sorted(y_src.unique())},
        }
    unsafe_test = (test["label"] != "SAFE").astype(int)
    results["test"]["binary_unsafe_at_review"] = ev.classification_metrics(unsafe_test, test_risk, thresholds["review"])
    results["test"]["binary_unsafe_at_flag"] = ev.classification_metrics(unsafe_test, test_risk, thresholds["flag"])
    results["data_summary"] = {
        split: {f"{src}:{label}": int(n)
                for (src, label), n in data[data["split"] == split].groupby(["source", "label"]).size().items()}
        for split in ("train", "test")
    }

    version = f"safety-{best}-{datetime.now(timezone.utc):%Y%m%d}"
    results["version"] = version
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "action_thresholds": thresholds, "version": version,
                 "labels": LABELS, "metrics": results}, SAFETY_BUNDLE_PATH, compress=3)
    SAFETY_METRICS_PATH.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", SAFETY_BUNDLE_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    res = train_and_evaluate()
    print(json.dumps({k: res[k] for k in ("dedup", "selected_model", "action_thresholds", "data_summary")}, indent=2, default=float))
    print("CV macro-F1:", {k: round(v["macro_f1"], 4) for k, v in res["cv"].items()},
          "| length-only:", round(res["diagnostics"]["length_only"]["macro_f1"], 4))
    print("Test macro-F1:", round(res["test"]["macro_f1"], 4), "| by source:", res["test"]["by_source"])
    print("Test confusion (rows=true", LABELS, "):", res["test"]["confusion_matrix"])
