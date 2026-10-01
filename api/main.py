"""Unified risk API: fraud scoring, prompt-safety analysis and grounded investigations.

Run locally:

    uvicorn api.main:app --reload

Interactive docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request

from api.schemas import (
    FraudPrediction,
    HealthResponse,
    InvestigationResponse,
    PromptRequest,
    SafetyAnalysis,
    TransactionRequest,
)
from src import explainability as xai
from src import fraud_model as fm
from src import investigation_assistant as assistant
from src import safety_model as sm

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("risk_api")

FRAUD_MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", fm.BUNDLE_PATH))
SAFETY_MODEL_PATH = Path(os.getenv("SAFETY_MODEL_PATH", sm.SAFETY_BUNDLE_PATH))


def _try_load(loader, path: Path, name: str) -> Optional[dict]:
    """Load a model bundle; on failure log it and keep serving (health reports 'degraded')."""
    try:
        bundle = loader(path)
        logger.info("Loaded %s model %s", name, bundle["version"])
        return bundle
    except FileNotFoundError as exc:
        logger.warning("%s model unavailable: %s", name, exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not per request.
    app.state.fraud = _try_load(fm.load_bundle, FRAUD_MODEL_PATH, "fraud")
    app.state.safety = _try_load(sm.load_safety_bundle, SAFETY_MODEL_PATH, "safety")
    yield


app = FastAPI(
    title="AI Safety & Fraud Risk API",
    version="1.0.0",
    description="Fraud risk scoring with SHAP explanations, LLM prompt-safety screening, "
                "and grounded investigation summaries.",
    lifespan=lifespan,
)


def _require(request: Request, name: str) -> dict:
    bundle = getattr(request.app.state, name, None)
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"{name} model is not loaded; see /health")
    return bundle


def _predict(bundle: dict, tx: TransactionRequest, explain: bool) -> FraudPrediction:
    transaction = tx.model_dump(exclude={"transaction_id"})
    result = fm.predict_transaction(bundle, transaction)
    factors = None
    if explain and result["in_model_scope"]:
        factors = xai.explain_transaction(bundle, transaction)["top_factors"]
    return FraudPrediction(transaction_id=tx.transaction_id, top_factors=factors, **result)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    models = {
        "fraud": request.app.state.fraud["version"] if request.app.state.fraud else None,
        "safety": request.app.state.safety["version"] if request.app.state.safety else None,
    }
    return HealthResponse(status="ok" if all(models.values()) else "degraded", models=models)


@app.post("/fraud/predict", response_model=FraudPrediction, tags=["fraud"])
def fraud_predict(
    tx: TransactionRequest,
    request: Request,
    explain: bool = Query(False, description="Include the top SHAP contributing factors"),
) -> FraudPrediction:
    """Score a transaction and map it to LOW / MEDIUM / HIGH risk."""
    prediction = _predict(_require(request, "fraud"), tx, explain)
    logger.info("fraud/predict id=%s type=%s risk=%s", tx.transaction_id, tx.type, prediction.risk_level)
    return prediction


@app.post("/fraud/investigate", response_model=InvestigationResponse, tags=["fraud"])
def fraud_investigate(tx: TransactionRequest, request: Request) -> InvestigationResponse:
    """Answer "why was this transaction flagged?" using only the model's own outputs."""
    bundle = _require(request, "fraud")
    prediction = _predict(bundle, tx, explain=True)
    if not prediction.in_model_scope:
        raise HTTPException(status_code=422, detail=f"{tx.type} transactions are outside the fraud model's scope.")
    transaction = tx.model_dump(exclude={"transaction_id"})
    explanation = {"top_factors": [f.model_dump() for f in prediction.top_factors]}
    facts = assistant.build_case_facts(transaction, prediction.model_dump(), explanation, tx.transaction_id)
    result = assistant.explain_case(facts)
    logger.info("fraud/investigate id=%s generated_by=%s", tx.transaction_id, result["generated_by"])
    return InvestigationResponse(prediction=prediction, **result)


@app.post("/safety/analyse", response_model=SafetyAnalysis, tags=["llm safety"])
def safety_analyse(body: PromptRequest, request: Request) -> SafetyAnalysis:
    """Classify a prompt and recommend ALLOW / REVIEW / FLAG."""
    result = sm.analyse_prompt(_require(request, "safety"), body.prompt)
    # Log the decision, never the prompt text (it may contain user data).
    logger.info("safety/analyse chars=%d class=%s action=%s", len(body.prompt), result["classification"], result["action"])
    return SafetyAnalysis(**result)
