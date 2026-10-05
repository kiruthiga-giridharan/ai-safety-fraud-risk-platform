# AI Safety & Fraud Risk Analytics Platform

An end-to-end analytics and machine-learning platform with two connected risk systems:

1. **Financial fraud detection.** It scores 6.3M mobile-money transactions, maps each score to a LOW / MEDIUM / HIGH risk level, and explains every score with SHAP.
2. **LLM safety screening.** It classifies prompts as SAFE, PROMPT_INJECTION or JAILBREAK and recommends ALLOW / REVIEW / FLAG.

Both systems sit behind one **FastAPI** service and one **Streamlit** dashboard. A **grounded investigation assistant** turns model outputs into an analyst-readable explanation, and it can only cite factors the model actually used.

> Every metric in this README was computed by the code in this repository. The fraud data is **synthetic**, so read the results as a demonstration of method, not as real-world fraud performance (see [Limitations](#limitations)).

**Python · Pandas · NumPy · SQL (SQLite) · scikit-learn · XGBoost · SHAP · Matplotlib · Plotly · FastAPI · Pydantic · Streamlit · Joblib · Pytest · Claude API**

---

## Screenshots

**Executive overview:** headline figures from the held-out test data, with the honest comparison against a one-line rule.
![Executive overview](docs/images/overview.png)

**Transaction investigation:** score, risk level and the SHAP factors behind one real transaction.
![Transaction investigation](docs/images/investigation.png)

| Model performance | LLM safety |
|---|---|
| ![Model performance](docs/images/model-performance.png) | ![LLM safety](docs/images/llm-safety.png) |

| Fraud analytics |
|---|
| ![Fraud analytics](docs/images/fraud-analytics.png) |

---

## Contents
[Screenshots](#screenshots) · [Business Problem](#business-problem) · [Architecture](#architecture) · [Dataset](#dataset) · [Data Pipeline](#data-pipeline) · [SQL Analysis](#sql-analysis) · [EDA Findings](#eda-findings) · [Feature Engineering](#feature-engineering) · [Machine Learning](#machine-learning) · [Model Evaluation](#model-evaluation) · [Explainability](#explainability) · [LLM Safety](#llm-safety) · [API](#api) · [Dashboard](#dashboard) · [Investigation Assistant](#investigation-assistant) · [Installation](#installation) · [How to Run](#how-to-run) · [Repository Structure](#repository-structure) · [Key Findings](#key-findings) · [Limitations](#limitations) · [Future Improvements](#future-improvements) · [Design Decisions](#design-decisions-interview-notes)

---

## Business Problem

**Fraud.** Payment providers must stop fraudulent transfers without blocking legitimate customers. Fraud is rare (0.13% of transactions here), so a model that predicts "not fraud" for everything is 99.87% accurate and useless. The platform therefore:
- ranks transactions by risk;
- explains why each one was flagged;
- lets analysts choose the trade-off between **false negatives** (fraud classified as legitimate, so money is lost) and **false positives** (legitimate transactions flagged, so analyst time is wasted and customers are annoyed).

**LLM safety.** Applications built on LLMs receive prompts that try to override instructions or bypass safeguards. The question here is whether cheap, traditional NLP can screen these prompts *before* they reach an expensive LLM.

---

## Architecture

```
 FRAUD                                                     LLM SAFETY
 data/raw/PaySim CSV (6.36M rows)                          data/raw/llm_safety/ (2 public datasets)
   │ data_loader → data_validation (report only)             │ load → de-duplicate (incl. train/test leaks)
   │ preprocessing (16 rows excluded, audited)               │ TF-IDF word + char n-grams
   ├──► SQLite + 15 named SQL queries                        │ logistic regression (5-fold CV on train)
   │ feature_engineering (13 features, 2 feature sets)       │ action thresholds on out-of-fold scores
   │ train / validation / test  70 / 15 / 15                 ▼
   │ LogReg · Random Forest · XGBoost  (+ 2 rule baselines)  safety_model.joblib
   │ risk boundaries from validation; test used once            │
   │ SHAP TreeExplainer                                         │
   ▼                                                            │
 fraud_model.joblib ───────────────┬────────────────────────────┘
                                   │
                    FastAPI  /health  /fraud/predict  /fraud/investigate  /safety/analyse
                    Streamlit  6-page dashboard
                    Investigation assistant (Claude, grounded in SHAP facts; template fallback)
```

---

## Dataset

### Fraud: PaySim
A **synthetic** mobile-money dataset produced by a simulator calibrated on real transaction logs (E. A. Lopez-Rojas, A. Elmir, S. Axelsson, *PaySim: A financial mobile money simulator for fraud detection*, EMSS 2016). Source: Kaggle `ealaxi/paysim1`, file `PS_20174392719_1491204439457_log.csv`. You download it yourself; it is not committed.

| Property | Value |
|---|---|
| Rows × columns | 6,362,620 × 11 |
| Missing values / exact duplicates | 0 / 0 |
| Fraudulent transactions | 8,213 (0.1291%), i.e. 773.7 legitimate per fraud |
| Types containing fraud | TRANSFER (0.769% fraud rate) and CASH_OUT (0.184%) only |
| Simulated time span | steps 1–743 (one step = one hour) |

### LLM safety
| Dataset | Labels | Licence | Notes |
|---|---|---|---|
| [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | injection / benign | Apache-2.0 | 662 short prompts, English and German |
| [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | jailbreak / benign | Apache-2.0 | 1,998 prompts, mostly long role-play jailbreaks |

PROMPT_EXTRACTION and SUSPICIOUS are **not** modelled. No labelled data exists for them here, and inventing labels would mean fabricating training data. "Suspicious" is expressed through the REVIEW action instead.

---

## Data Pipeline

```
raw CSV ─► data_loader.py       explicit dtypes (money stays float64), consistent snake_case names
        ─► data_validation.py   schema, nulls, duplicates, impossible values, balance checks; never drops rows
        ─► preprocessing.py     adds transaction_id (TX-0000001); excludes 16 zero-amount "frauds" → excluded_rows.csv
        ─► database.py          SQLite with CHECK constraints + indexes
        ─► feature_engineering  13 features, train/validation/test parquet files
```

Cleaning is deliberately minimal. In 90.1% of legitimate TRANSFER/CASH_OUT rows the amount is larger than the sender's balance, and the balance is floored at 0. That is **how the simulator behaves**, not corruption. Dropping those rows would delete most of the data and much of the signal, so they are kept and documented.

---

## SQL Analysis

15 named queries live in [`sql/fraud_analysis.sql`](sql/fraud_analysis.sql) and [`sql/investigation_queries.sql`](sql/investigation_queries.sql). Each query is annotated with the business question it answers, and all of them run in [`notebooks/01b_fraud_sql_analysis.ipynb`](notebooks/01b_fraud_sql_analysis.ipynb).

| Finding | Result |
|---|---|
| Existing rule flag (`isFlaggedFraud`) | 16 of 8,197 frauds caught: precision 100%, **recall 0.20%** |
| One-line rule "moves the entire sender balance" | **100% precision, 97.82% recall** (simulator artefact) |
| Balance consistency | Fraud is the *consistent* ledger behaviour: 3.02% fraud rate vs 0.0012% where the balance was floored at 0 |
| Fraud by amount | 8.95% fraud rate at exactly 10M; 0% above 10M (fraud is capped) |
| Fraud over time | Steady 216–319 frauds/day; legitimate volume collapses from day 18; day 31 is 100% fraud |
| TRANSFER → CASH_OUT chains | 99.5% of fraud transfers have a matching cash-out, linked only by timing and row order, never by account |
| Mule accounts / repeat victims | No account received more than 2 frauds; no sender was defrauded twice |

## EDA Findings
From [`notebooks/01_fraud_eda.ipynb`](notebooks/01_fraud_eda.ipynb):
- **Imbalance:** 773.7 legitimate transactions per fraud, so accuracy cannot be the metric.
- **Fraud occurs only in TRANSFER and CASH_OUT.** TRANSFER is 8.4% of transactions but 49.9% of fraud.
- **Fraud amounts are larger but overlap heavily:** median 441K vs 171K. Fraud is capped at exactly 10M.
- **Balance patterns separate the classes most strongly.** The whole sender balance is moved in 97.8% of frauds vs 0.0001% of legitimate transactions.
- **The fraud rate peaks at hours 3–5 only because legitimate volume drops.** Fraud counts per hour are flat (274–375).

---

## Feature Engineering

Framing: **post-transaction monitoring**. Transactions are scored after they are processed, so post-transaction balances are valid inputs. All logic lives in [`src/feature_engineering.py`](src/feature_engineering.py) and is shared with the API, so training and serving compute features identically. Details are in [`notebooks/02_fraud_feature_engineering.ipynb`](notebooks/02_fraud_feature_engineering.ipynb).

- **Scope:** TRANSFER and CASH_OUT only. That is 2,770,393 rows and all 8,197 frauds; the other types contain no fraud.
- **13 features:**
  - log amount and transaction type;
  - log sender and recipient balances;
  - signed ledger-error terms;
  - amount-to-balance ratio;
  - zero-balance, overspend and "fully drained" flags.
- **Two feature sets:**
  - `full` (13 features);
  - `no_origin_balance` (6 features), which removes the origin-balance artefact.
- **Excluded for leakage or artefacts:**
  - `step` (time; 320 of 743 hours are 100% fraud);
  - account IDs;
  - `isFlaggedFraud`;
  - `transaction_id` (encodes file order);
  - TRANSFER→CASH_OUT chain features (they use a future row);
  - whole-dataset account counts.
- **Split:** 70/15/15 train/validation/test, stratified and random. A time-based split is not used because legitimate traffic disappears from day 18.

---

## Machine Learning

| Model | Role | Imbalance handling |
|---|---|---|
| Logistic Regression | Interpretable baseline (scaled inputs, L2) | `class_weight="balanced"` |
| Random Forest | Non-linear, robust to collinear features | `class_weight="balanced_subsample"` |
| XGBoost | Strongest tabular learner; exact SHAP | `scale_pos_weight` = 337 |

Hyperparameters are sensible defaults and deliberately not tuned against validation. SMOTE was not needed, because class weighting worked.

## Model Evaluation

**Validation PR-AUC** is the headline metric. It focuses on the fraud class and is not inflated by 414K easy negatives.

| | Existing rule | Drain rule | LogReg | Random Forest | XGBoost |
|---|---|---|---|---|---|
| `full` features | 0.005 | 0.973 | 0.9966 | 0.9968 | **0.9967** |
| `no_origin_balance` | | | 0.3496 | 0.6902 | **0.7010** |

- **With all features the three models tie.** The data is so separable that model choice barely matters. XGBoost is deployed: PR-AUC differences under 0.001 count as a tie, and XGBoost has the best F1.
- **What the model adds beyond the drain rule:** on validation, 27 more frauds caught (1,223 vs 1,196) for 6 false alarms.
- **Without origin-balance features, trees far outperform logistic regression** (0.70 vs 0.35). That is where model choice actually matters.

**Threshold analysis.** For the deployed model, thresholds 0.30–0.70 give *identical* results, because scores are bimodal: almost all are near 0 or near 1. For the `no_origin_balance` XGBoost, moving from 0.30 to 0.70 cuts alerts from 64,834 to 3,945 while recall falls from 82.8% to 72.1%. That is the real trade-off a fraud team would face.

**Risk levels.** These come from business targets applied to validation data, not from hand-picked numbers:
- **HIGH** = the lowest score where alert precision is at least 99% (safe to hold funds automatically). Boundary: **≥ 0.138**.
- **MEDIUM** = the lowest score where alert precision is still at least 90% (the analyst review queue). Boundary: **≥ 0.0014**.

**Test split (used once):**

| System | Precision | Recall | False positives | Missed fraud |
|---|---|---|---|---|
| Existing rule flag | 100% | 0.08% | 0 | 1,229 |
| Drain rule | 100% | 97.56% | 0 | 30 |
| **Model @ MEDIUM** | **90.96%** | **99.84%** | 122 (FPR 0.03%) | 2 |
| Model @ HIGH | 99.35% | 99.67% | 8 | 4 |

Validation and test agree closely. **Scores are rankings, not calibrated probabilities:** class weighting inflates them (the SHAP base value is +8.83 log-odds), so the risk level is the field to act on.

## Explainability

SHAP `TreeExplainer` gives exact per-transaction contributions, and they sum to the model output. This is checked in the tests.
- **Global importance:** "sender account fully drained" dominates (mean |SHAP| 5.91), followed by the recipient balances (3.99, 3.25).
- **A counter-intuitive result:** TRANSFER pushes scores slightly *down*, even though TRANSFER has the higher raw fraud rate. SHAP measures a feature's effect *given the other features*, which is not the same as its raw correlation.
- **Local explanations** cover a caught fraud, a false alarm and a missed fraud ([`notebooks/04_fraud_explainability.ipynb`](notebooks/04_fraud_explainability.ipynb)). The missed fraud looked exactly like a legitimate overspend, so the model cannot catch fraud that does not follow the simulator's pattern.

---

## LLM Safety

The deployed model is TF-IDF (word + character n-grams) with logistic regression. Candidates were compared with 5-fold cross-validation on the training data; the test split was used once ([`notebooks/05`](notebooks/05_llm_safety_eda.ipynb), [`06`](notebooks/06_llm_safety_modelling.ipynb)).

| | Macro-F1 |
|---|---|
| TF-IDF words (baseline), CV | 0.900 |
| TF-IDF words + characters, CV | **0.913** |
| Length-only diagnostic, CV | 0.458 |
| **Deployed model, test** | **0.905** |

| Test | Precision | Recall |
|---|---|---|
| SAFE | 0.931 | **1.000** |
| PROMPT_INJECTION | 0.952 | **0.667** |
| JAILBREAK | 0.985 | 0.949 |

- **The source confound was checked.** Within the deepset dataset alone (one writing style), safe vs injection reaches macro-F1 0.831. That is the honest measure of injection detection.
- **Actions** use thresholds chosen on out-of-fold training scores: REVIEW ≥ 0.254, FLAG ≥ 0.358. On test, REVIEW catches 91.9% of unsafe prompts at 95.8% precision, short of the 95% recall target.
- **Weaknesses:**
  - injections appended to harmless questions;
  - role-play framing ("roleplay as" is learned as a SAFE signal);
  - topic-specific vocabulary.
- **Why not classify with an LLM?** Each prompt would need an extra paid call with seconds of latency, and an LLM classifier can itself be prompt-injected. The TF-IDF model runs in milliseconds on a CPU and is fully measurable. It works as a **first filter**, not a complete defence.

---

## API

`uvicorn api.main:app --reload`, with interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | | status `ok`/`degraded`, loaded model versions |
| `POST /fraud/predict?explain=true` | transaction | `fraud_probability`, `risk_level`, `in_model_scope`, `model_version`, optional SHAP `top_factors` |
| `POST /fraud/investigate` | transaction | prediction + grounded analyst explanation |
| `POST /safety/analyse` | `{"prompt": "..."}` | `classification`, `risk_score`, `action`, class probabilities |

Implementation notes:
- **Pydantic validation:** amount must be > 0, balances ≥ 0, and the type must be a known value. Anything else returns 422.
- **Models load once at startup.** If a model is missing, `/health` reports `degraded` and that endpoint returns 503.
- **Out-of-scope types** (PAYMENT, CASH_IN, DEBIT) return `in_model_scope: false` with no made-up probability.
- **Prompt text is never logged.**

```bash
curl -X POST localhost:8000/fraud/predict?explain=true -H 'content-type: application/json' \
  -d '{"type":"TRANSFER","amount":181.0,"old_balance_orig":181.0,"new_balance_orig":0.0,"old_balance_dest":0.0,"new_balance_dest":0.0}'
# → "risk_level": "HIGH", top factor "Sender account fully drained" (+10.19)
```

## Dashboard

`streamlit run dashboard/app.py`. Pages: **Executive Overview · Fraud Analytics · Transaction Investigation · Model Performance · LLM Safety · About**. Every figure is read from the SQLite queries and model artefacts. You can deep-link to a page with `?page=model-performance`.

## Investigation Assistant

"Why was this transaction flagged?" The assistant receives **only** structured case facts: the score, the risk level, the transaction fields and the SHAP factors. Grounding is enforced, not merely requested:
1. The structured-output schema restricts each reason's `feature` to an **enum of that case's SHAP features**, so Claude cannot cite anything else.
2. Code re-checks every reason: the feature must exist in the facts and the direction must match the SHAP sign. Mismatches are dropped and reported.
3. If no credentials are configured, or the API errors or declines, a **deterministic template** writes the explanation from the same facts.

It uses `claude-opus-5-5` at low effort, with server-side refusal fallbacks. LangChain/LangGraph were not used: this is a single call with a fixed input and output, so a framework would add dependencies without adding capability.

> The LLM path is unit-tested with a fake client. It has **not yet been run against the live API**, because no credentials were configured during development. Set `ANTHROPIC_API_KEY` in `.env` to enable it.

---

## Installation

Python 3.11 is recommended. The code also runs on 3.9+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp        # macOS only: OpenMP runtime required by XGBoost
cp .env.example .env       # optional: Anthropic key for the investigation assistant
```

## How to Run

```bash
# 1. Data (manual download)
#    PaySim: Kaggle ealaxi/paysim1 → data/raw/PS_20174392719_1491204439457_log.csv
#    Safety: deepset/prompt-injections + jackhhao/jailbreak-classification → data/raw/llm_safety/
#            (deepset_{train,test}.parquet, jackhhao_{train,test}.csv)

# 2. Pipeline (~10 min on a laptop)
python -m src.preprocessing        # clean → data/processed/transactions_clean.parquet
python -m src.database             # → data/processed/paysim.db
python -m src.feature_engineering  # → train / validation / test splits
python -m src.fraud_model          # train 6 models, choose boundaries, test once → models/
python -m src.safety_model         # → models/safety_model.joblib

# 3. Use it
pytest                             # 72 tests
uvicorn api.main:app --reload
streamlit run dashboard/app.py
jupyter notebook notebooks/
```

## Repository Structure

```
ai-safety-fraud-risk-platform/
├── data/raw/, data/processed/          # git-ignored
├── notebooks/
│   ├── 01_fraud_eda.ipynb                 data quality + EDA
│   ├── 01b_fraud_sql_analysis.ipynb       all 15 SQL queries with findings
│   ├── 02_fraud_feature_engineering.ipynb features, leakage review, splits
│   ├── 03_fraud_modelling.ipynb           comparison, thresholds, risk levels, test
│   ├── 04_fraud_explainability.ipynb      SHAP global + local
│   ├── 05_llm_safety_eda.ipynb
│   └── 06_llm_safety_modelling.ipynb
├── sql/                                   fraud_analysis.sql, investigation_queries.sql
├── src/
│   ├── data_loader.py, data_validation.py, preprocessing.py, database.py
│   ├── feature_engineering.py, fraud_model.py, evaluation.py, explainability.py
│   ├── safety_model.py
│   └── investigation_assistant.py
├── api/            main.py, schemas.py
├── dashboard/      app.py
├── models/         git-ignored trained artefacts
├── tests/          72 pytest tests
├── requirements.txt, .env.example, .gitignore
```

---

## Key Findings

1. **On PaySim, a one-line rule nearly solves fraud detection.** "Moves the entire sender balance" gets 97.6% recall at 100% precision on test. The ML model adds about 2 points of recall. Reporting that honestly matters more than reporting 0.997 PR-AUC.
2. **Without that artefact, fraud detection is genuinely hard** (best PR-AUC 0.70), and tree models clearly beat logistic regression.
3. **Intuition can point the wrong way.** In PaySim, *consistent* ledgers indicate fraud and inconsistent ones indicate legitimate customers.
4. **Class weighting makes scores rankings, not probabilities.** Risk levels derived from precision targets are what analysts should act on.
5. **Cheap NLP is a credible first-line prompt filter** (jailbreak F1 0.967), but it misses a third of prompt injections.

## Limitations

- **Synthetic fraud data with strong artefacts:** account draining, a 10M cap on fraud, and legitimate traffic disappearing from day 18. Results will not transfer to real banking data.
- **Fraud scores are not calibrated probabilities.**
- **Post-transaction framing:** a real-time pre-authorisation system would not have post-transaction balances. The `no_origin_balance` results are a closer guide for that case.
- **Small safety datasets:** 60 injections and 137 jailbreaks in test, so the numbers have wide uncertainty. There is a source confound (every injection comes from one dataset), some deepset labels are ambiguous, and there is no labelled data for prompt extraction.
- **The investigation assistant's LLM path has not been run against the live API yet.**
- **Development environment note:** XGBoost on this macOS machine was linked to scikit-learn's bundled OpenMP because Homebrew was unavailable. `brew install libomp` is the supported setup.

## Future Improvements

- Calibrate fraud scores (isotonic or Platt scaling on a held-out set) so the API can return real probabilities.
- Validate on real or less artefact-prone data, and add point-in-time account-history features (prior activity only).
- Tune hyperparameters with a separate tuning split or nested cross-validation.
- Safety: a fine-tuned small transformer (e.g. DeBERTa) on a larger, more varied labelled set, compared on the same within-source metrics. Use an LLM only for the REVIEW queue.
- Monitoring: log score distributions and alert rates per model version; alert on drift.
- Containerise (Docker) and add CI running `pytest` on every push.

---

## Design Decisions (interview notes)

| Decision | Why | Alternative |
|---|---|---|
| PR-AUC, not accuracy | With 0.3% fraud, accuracy rewards predicting "legitimate" for everything | ROC-AUC (inflated by easy negatives) |
| Logistic regression first | An interpretable baseline: if it suffices, complexity is not justified | Starting with XGBoost |
| XGBoost deployed | Tied best PR-AUC, best F1, fast, exact SHAP | Random forest (tied, larger, slower) |
| Class weights, not SMOTE | Encodes "missing fraud is costly" without synthetic transactions | SMOTE / undersampling |
| Train/validation/test | Thresholds are parameters; choosing them on test would leak | 80/20 with test-tuned thresholds |
| Benchmark against rules | Shows what the model actually adds | Reporting model metrics alone |
| SHAP | Exact, additive, per-transaction explanations | Built-in importance (global only), LIME (approximate) |
| TF-IDF before LLMs | Milliseconds, cheap, measurable, cannot be prompt-injected | An LLM classifier on every prompt |
| Grounded assistant via schema enum + code checks | The LLM is structurally unable to cite unused factors | Prompt instructions alone |
