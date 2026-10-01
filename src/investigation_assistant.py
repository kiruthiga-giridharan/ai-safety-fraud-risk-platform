"""Grounded investigation assistant: "Why was this transaction flagged?"

The assistant turns *structured model outputs* (prediction, risk level, SHAP
factors, transaction fields) into a short analyst-readable explanation.

Grounding guarantees:
1. Claude only receives the case facts. It is told to use nothing else.
2. Structured output: each reason must name a feature from an enum built
   from *this case's* SHAP factors, so it cannot cite anything else.
3. Code then re-checks every reason: the feature must be in the facts and
   the stated direction must match the SHAP sign. Failing reasons are dropped
   and reported.
4. If Claude is unavailable (no credentials, network error, refusal), a
   deterministic template produces the explanation from the same facts.

LangChain/LangGraph are not used: this is one LLM call with a fixed input
and output, so a framework would add dependencies without adding capability.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import anthropic

logger = logging.getLogger(__name__)

MODEL = os.getenv("INVESTIGATION_MODEL", "claude-opus-5-5")
LLM_ENABLED = os.getenv("INVESTIGATION_LLM", "on").lower() != "off"

SYSTEM_PROMPT = """You are a fraud-investigation assistant writing for a bank analyst.

You will receive CASE FACTS as JSON: a model's fraud score and risk level, the
transaction's fields, and the model's top contributing factors from SHAP
(each with a direction and a factual value description).

Rules:
- Use ONLY the case facts. Do not add reasons, typical fraud patterns, or
  assumptions that are not in the facts.
- Explain each factor using its value description; keep the direction
  ("increases risk" / "decreases risk") exactly as given.
- The fraud score is a model score, not a calibrated probability; do not
  describe it as a certainty.
- If the facts are insufficient to explain something, say so in caveats.
- Plain, neutral language. No accusations about any person."""


def build_case_facts(transaction: dict, prediction: dict, explanation: dict,
                     transaction_id: Optional[str] = None) -> dict:
    """Assemble the only information the assistant is allowed to use."""
    return {
        "transaction_id": transaction_id,
        "transaction": {k: transaction[k] for k in (
            "type", "amount", "old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest")},
        "fraud_score": prediction["fraud_probability"],
        "risk_level": prediction["risk_level"],
        "model_version": prediction["model_version"],
        "factors": [
            {"feature": f["feature"], "label": f["label"], "direction": f["direction"],
             "shap_value": round(f["shap_value"], 3), "value_description": f["value_description"]}
            for f in explanation["top_factors"]
        ],
    }


def output_schema(facts: dict) -> dict:
    """JSON schema whose feature enum contains only this case's SHAP factors."""
    features = [f["feature"] for f in facts["factors"]]
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string", "enum": features},
                        "direction": {"type": "string", "enum": ["increases risk", "decreases risk"]},
                        "explanation": {"type": "string"},
                    },
                    "required": ["feature", "direction", "explanation"],
                    "additionalProperties": False,
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "reasons", "caveats"],
        "additionalProperties": False,
    }


def validate_grounding(answer: dict, facts: dict) -> tuple[dict, list[str]]:
    """Drop any reason not backed by the facts; return (clean answer, problems)."""
    by_feature = {f["feature"]: f for f in facts["factors"]}
    kept, problems = [], []
    for reason in answer.get("reasons", []):
        fact = by_feature.get(reason.get("feature"))
        if fact is None:
            problems.append(f"dropped reason citing unknown feature {reason.get('feature')!r}")
        elif reason.get("direction") != fact["direction"]:
            problems.append(f"dropped reason for {reason['feature']!r}: direction does not match SHAP")
        else:
            kept.append({**reason, "label": fact["label"], "shap_value": fact["shap_value"]})
    return {**answer, "reasons": kept}, problems


def template_explanation(facts: dict) -> dict:
    """Deterministic explanation built directly from the facts (no LLM)."""
    score = facts["fraud_score"]
    tx = facts["transaction"]
    summary = (
        f"{tx['type']} of {tx['amount']:,.2f} scored {score:.1%} by model {facts['model_version']}, "
        f"risk level {facts['risk_level']}."
    )
    reasons = [
        {"feature": f["feature"], "label": f["label"], "direction": f["direction"], "shap_value": f["shap_value"],
         "explanation": f"{f['label']} ({f['value_description']}) {f['direction']}."}
        for f in facts["factors"]
    ]
    caveats = ["The score is a class-weighted model score, not a calibrated probability."]
    return {"summary": summary, "reasons": reasons, "caveats": caveats}


def _call_claude(facts: dict, client: Any) -> dict:
    """One grounded structured-output call. Raises on any failure; caller falls back."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "CASE FACTS:\n" + json.dumps(facts, indent=2)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": output_schema(facts)}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response truncated")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def explain_case(facts: dict, client: Any = None) -> dict:
    """Produce a grounded analyst explanation, via Claude when available.

    Returns the explanation plus provenance: ``generated_by`` ("llm" or
    "template"), any grounding problems, and why the LLM was not used.
    """
    if not LLM_ENABLED:
        return {**template_explanation(facts), "generated_by": "template",
                "grounding_problems": [], "llm_unavailable_reason": "disabled by INVESTIGATION_LLM=off"}

    try:
        client = client or anthropic.Anthropic()
        answer = _call_claude(facts, client)
    except anthropic.APIStatusError as exc:          # 4xx/5xx from the API (auth, rate limit, server)
        reason = f"API error {exc.status_code}: {type(exc).__name__}"
    except anthropic.APIConnectionError:             # network failure or timeout
        reason = "could not reach the Anthropic API"
    except TypeError as exc:                         # SDK found no credentials
        reason = f"no Anthropic credentials configured ({exc})"
    except (RuntimeError, json.JSONDecodeError, StopIteration) as exc:  # unusable response
        reason = f"unusable model response: {exc}"
    else:
        clean, problems = validate_grounding(answer, facts)
        if not clean["reasons"]:
            problems.append("no grounded reasons left; using template")
            return {**template_explanation(facts), "generated_by": "template",
                    "grounding_problems": problems, "llm_unavailable_reason": None}
        return {**clean, "generated_by": "llm", "grounding_problems": problems, "llm_unavailable_reason": None}

    logger.info("Investigation assistant using template: %s", reason)
    return {**template_explanation(facts), "generated_by": "template",
            "grounding_problems": [], "llm_unavailable_reason": reason}
