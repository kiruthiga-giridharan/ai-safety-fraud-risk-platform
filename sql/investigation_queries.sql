-- =============================================================================
-- Investigation queries: rules, suspicious accounts and high-risk patterns.
-- Dialect: SQLite. Table: transactions (see src/database.py for the schema).
--
-- These are retrospective investigation queries. Several look at the whole
-- dataset, including activity *after* a given transaction, which is fine for an
-- analyst looking back but must NOT be reused as model features (leakage).
-- =============================================================================


-- name: top_fraud_transactions
-- question: What are the highest-value fraudulent transactions, and what do their balances show?
SELECT
    transaction_id, step, type, amount,
    name_orig, old_balance_orig, new_balance_orig,
    name_dest, old_balance_dest, new_balance_dest,
    is_flagged_fraud
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC, transaction_id
LIMIT 20;


-- name: existing_rule_performance
-- question: How well does PaySim's existing rule flag (is_flagged_fraud) catch fraud? This is the bar any model must beat.
SELECT
    SUM(is_flagged_fraud = 1 AND is_fraud = 1)   AS true_positives,
    SUM(is_flagged_fraud = 1 AND is_fraud = 0)   AS false_positives,
    SUM(is_flagged_fraud = 0 AND is_fraud = 1)   AS false_negatives,
    SUM(is_flagged_fraud = 0 AND is_fraud = 0)   AS true_negatives,
    ROUND(100.0 * SUM(is_flagged_fraud = 1 AND is_fraud = 1)
          / NULLIF(SUM(is_flagged_fraud), 0), 2) AS precision_pct,
    ROUND(100.0 * SUM(is_flagged_fraud = 1 AND is_fraud = 1)
          / NULLIF(SUM(is_fraud), 0), 2)         AS recall_pct
FROM transactions;


-- name: full_balance_drain_rule
-- question: If an analyst flagged every TRANSFER/CASH_OUT that moves the entire origin balance, what precision and recall would that simple rule get?
WITH flagged AS (
    SELECT
        is_fraud,
        (old_balance_orig > 0 AND ABS(amount - old_balance_orig) < 0.01) AS drains_account
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    CASE drains_account WHEN 1 THEN 'moves entire balance' ELSE 'other' END    AS pattern,
    COUNT(*)                                                                    AS transactions,
    SUM(is_fraud)                                                               AS fraud,
    ROUND(100.0 * AVG(is_fraud), 4)                                             AS fraud_rate_pct,  -- = precision of the rule
    ROUND(100.0 * SUM(is_fraud) / (SELECT SUM(is_fraud) FROM flagged), 2)       AS share_of_fraud_pct  -- = recall of the rule
FROM flagged
GROUP BY drains_account
ORDER BY drains_account DESC;


-- name: origin_balance_inconsistency
-- question: How often does the origin balance fail to change by exactly the amount, and how risky is each kind of inconsistency?
-- note: debit types should satisfy old - amount = new. A tolerance of 0.01 absorbs rounding.
WITH classified AS (
    SELECT
        is_fraud,
        CASE
            WHEN ABS(old_balance_orig - amount - new_balance_orig) <= 0.01
                THEN '1. consistent (old - amount = new)'
            WHEN amount > old_balance_orig AND new_balance_orig = 0
                THEN '2. amount exceeds balance, balance floored at 0'
            ELSE '3. other inconsistency'
        END AS origin_balance_state
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    origin_balance_state,
    COUNT(*)                          AS transactions,
    SUM(is_fraud)                     AS fraud,
    ROUND(100.0 * AVG(is_fraud), 4)   AS fraud_rate_pct
FROM classified
GROUP BY origin_balance_state
ORDER BY origin_balance_state;


-- name: high_risk_pattern_matrix
-- question: How do the two strongest balance signals combine? Fraud rate for every combination of 'drains origin account' and 'destination stays at 0'.
WITH flags AS (
    SELECT
        is_fraud,
        (old_balance_orig > 0 AND ABS(amount - old_balance_orig) < 0.01)  AS drains_origin,
        (old_balance_dest = 0 AND new_balance_dest = 0)                    AS dest_stays_zero
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    drains_origin,
    dest_stays_zero,
    COUNT(*)                          AS transactions,
    SUM(is_fraud)                     AS fraud,
    ROUND(100.0 * AVG(is_fraud), 4)   AS fraud_rate_pct
FROM flags
GROUP BY drains_origin, dest_stays_zero
ORDER BY fraud_rate_pct DESC;


-- name: transfer_to_cashout_chains
-- question: How often is a TRANSFER followed by a CASH_OUT of the same amount in the same hour, and is that chain specific to fraud?
-- note: this is the documented PaySim fraud scenario (take over account -> transfer to mule -> cash out).
SELECT
    t.is_fraud              AS transfer_is_fraud,
    c.is_fraud              AS cashout_is_fraud,
    COUNT(*)                AS matched_pairs,
    COUNT(DISTINCT t.transaction_id) AS distinct_transfers
FROM transactions AS t
JOIN transactions AS c
  ON  c.type   = 'CASH_OUT'
  AND c.step   = t.step
  AND c.amount = t.amount
WHERE t.type = 'TRANSFER'
GROUP BY t.is_fraud, c.is_fraud
ORDER BY t.is_fraud DESC, c.is_fraud DESC;


-- name: suspicious_destination_accounts
-- question: Which destination accounts received the most fraudulent transfers? These are candidate mule accounts.
SELECT
    name_dest,
    SUM(is_fraud)                                                    AS fraud_received,
    COUNT(*) - SUM(is_fraud)                                         AS legitimate_received,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)     AS fraud_amount_received,
    MIN(CASE WHEN is_fraud = 1 THEN step END)                        AS first_fraud_step,
    MAX(CASE WHEN is_fraud = 1 THEN step END)                        AS last_fraud_step
FROM transactions
GROUP BY name_dest
HAVING SUM(is_fraud) >= 2
ORDER BY fraud_received DESC, fraud_amount_received DESC
LIMIT 20;


-- name: suspicious_origin_accounts
-- question: Do any origin accounts make more than one transaction including fraud? Repeat victims would show up here.
SELECT
    name_orig,
    COUNT(*)                         AS transactions,
    SUM(is_fraud)                    AS fraud,
    GROUP_CONCAT(DISTINCT type)      AS types_used,
    ROUND(SUM(amount), 2)            AS total_amount,
    MIN(step)                        AS first_step,
    MAX(step)                        AS last_step
FROM transactions
GROUP BY name_orig
HAVING COUNT(*) > 1 AND SUM(is_fraud) >= 1
ORDER BY fraud DESC, total_amount DESC
LIMIT 20;


-- name: account_activity
-- question: What is the full transaction history of one account, as sender or receiver? (Parameter: :account)
SELECT
    transaction_id, step, type, amount,
    CASE WHEN name_orig = :account THEN 'sent' ELSE 'received' END AS direction,
    name_orig, old_balance_orig, new_balance_orig,
    name_dest, old_balance_dest, new_balance_dest,
    is_fraud
FROM transactions
WHERE name_orig = :account OR name_dest = :account
ORDER BY step, transaction_id;
