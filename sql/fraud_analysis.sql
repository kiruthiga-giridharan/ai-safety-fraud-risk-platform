-- =============================================================================
-- Fraud analysis: descriptive queries over the cleaned PaySim transactions.
-- Dialect: SQLite. Table: transactions (see src/database.py for the schema).
--
-- Each query has a `-- name:` (used to run it from Python) and a
-- `-- question:` (the business question it answers).
-- =============================================================================


-- name: dataset_overview
-- question: How many transactions and how much money does the dataset contain, and what share of each is fraud?
SELECT
    COUNT(*)                                                         AS transactions,
    SUM(is_fraud)                                                    AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)                       AS fraud_rate_pct,
    ROUND(SUM(amount), 2)                                            AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)     AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)
          / SUM(amount), 4)                                          AS fraud_amount_pct
FROM transactions;


-- name: fraud_rate_by_type
-- question: Which transaction types carry fraud, how risky is each one, and how much of all fraud does each account for?
-- note: rate = risk of a single transaction of that type; share = where the fraud workload comes from.
SELECT
    type,
    COUNT(*)                                                                    AS transactions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 2)            AS share_of_transactions_pct,
    SUM(is_fraud)                                                               AS fraud,
    ROUND(100.0 * AVG(is_fraud), 4)                                             AS fraud_rate_pct,
    ROUND(100.0 * SUM(is_fraud) / (SELECT SUM(is_fraud) FROM transactions), 2)  AS share_of_fraud_pct
FROM transactions
GROUP BY type
ORDER BY fraud_rate_pct DESC, transactions DESC;


-- name: amount_profile_by_class
-- question: How do fraudulent and legitimate transaction amounts compare within the types where fraud occurs?
-- note: SQLite has no MEDIAN; medians are computed in pandas (notebook 01).
SELECT
    type,
    CASE is_fraud WHEN 1 THEN 'fraud' ELSE 'legitimate' END  AS class,
    COUNT(*)                                                 AS transactions,
    ROUND(AVG(amount), 2)                                    AS avg_amount,
    ROUND(MIN(amount), 2)                                    AS min_amount,
    ROUND(MAX(amount), 2)                                    AS max_amount,
    ROUND(SUM(amount), 2)                                    AS total_amount
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY type, is_fraud
ORDER BY type, is_fraud;


-- name: fraud_rate_by_amount_band
-- question: How does fraud risk change with transaction size, and does the 10,000,000 cap on fraud amounts show up?
SELECT
    CASE
        WHEN amount <  10000      THEN '1. under 10K'
        WHEN amount <  100000     THEN '2. 10K to 100K'
        WHEN amount <  1000000    THEN '3. 100K to 1M'
        WHEN amount <  10000000   THEN '4. 1M to under 10M'
        WHEN amount =  10000000   THEN '5. exactly 10M'
        ELSE                           '6. over 10M'
    END                                       AS amount_band,
    COUNT(*)                                  AS transactions,
    SUM(is_fraud)                             AS fraud,
    ROUND(100.0 * AVG(is_fraud), 4)           AS fraud_rate_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY amount_band
ORDER BY amount_band;


-- name: fraud_by_day
-- question: Is fraud volume stable across the simulated month, and on which days does legitimate activity disappear?
-- note: 1 step = 1 hour, so day = (step - 1) / 24 + 1 (integer division).
WITH per_step AS (
    SELECT step, COUNT(*) AS transactions, SUM(is_fraud) AS fraud
    FROM transactions
    GROUP BY step
)
SELECT
    (step - 1) / 24 + 1                               AS day,
    COUNT(*)                                          AS active_hours,
    SUM(transactions)                                 AS transactions,
    SUM(fraud)                                        AS fraud,
    ROUND(100.0 * SUM(fraud) / SUM(transactions), 4)  AS fraud_rate_pct,
    SUM(transactions = fraud)                         AS fraud_only_hours
FROM per_step
GROUP BY day
ORDER BY day;


-- name: fraud_by_hour_of_day
-- question: Does fraud cluster at particular hours, or does the fraud rate move only because legitimate volume changes?
-- note: step % 24 assumes the simulation starts at midnight (same convention as notebook 01).
SELECT
    step % 24                          AS hour_of_day,
    COUNT(*)                           AS transactions,
    SUM(is_fraud)                      AS fraud,
    COUNT(*) - SUM(is_fraud)           AS legitimate,
    ROUND(100.0 * AVG(is_fraud), 4)    AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;
