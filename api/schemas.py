"""Request and response models for the risk API (Pydantic v2)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

TransactionType = Literal["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class TransactionRequest(BaseModel):
    """One transaction, in the same units as the PaySim data."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "transaction_id": "TX-0000003", "type": "TRANSFER", "amount": 181.0,
        "old_balance_orig": 181.0, "new_balance_orig": 0.0,
        "old_balance_dest": 0.0, "new_balance_dest": 0.0,
    }})

    transaction_id: Optional[str] = Field(None, max_length=64)
    type: TransactionType
    amount: float = Field(..., gt=0, description="Must be positive; zero-amount rows were excluded from training.")
    old_balance_orig: float = Field(..., ge=0)
    new_balance_orig: float = Field(..., ge=0)
    old_balance_dest: float = Field(..., ge=0)
    new_balance_dest: float = Field(..., ge=0)


class Factor(BaseModel):
    feature: str
    label: str
    value_description: str
    shap_value: float
    direction: Literal["increases risk", "decreases risk"]


class FraudPrediction(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: Optional[float] = Field(
        None, description="Class-weighted model score in [0, 1]; a ranking, not a calibrated probability. "
                          "Null when the type is outside model scope.")
    risk_level: RiskLevel
    in_model_scope: bool
    model_version: str
    top_factors: Optional[List[Factor]] = None


class InvestigationReason(BaseModel):
    feature: str
    label: str
    direction: Literal["increases risk", "decreases risk"]
    shap_value: float
    explanation: str


class InvestigationResponse(BaseModel):
    prediction: FraudPrediction
    summary: str
    reasons: List[InvestigationReason]
    caveats: List[str]
    generated_by: Literal["llm", "template"]
    grounding_problems: List[str]
    llm_unavailable_reason: Optional[str]


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000)


class SafetyAnalysis(BaseModel):
    classification: Literal["SAFE", "PROMPT_INJECTION", "JAILBREAK"]
    risk_score: float = Field(..., description="1 - P(SAFE)")
    action: Literal["ALLOW", "REVIEW", "FLAG"]
    class_probabilities: Dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models: Dict[str, Optional[str]]
