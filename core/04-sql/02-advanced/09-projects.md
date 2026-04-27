# Advanced 9 — SQL Projects

## Two projects in this module

1. **Finance & Top-N Insights** — UDFs, stored procedures, views in a CPG / consumer-goods finance context
2. **Supply Chain Analytics & Model Optimisation** — helper tables, triggers, query tuning

Both come from Codebasics' **AtliQ Hardware** dataset (or a comparable CPG schema).

---

## Project A — Finance & Top-N Insights

### Domain
A consumer goods company sells through multiple channels (retailers, e-commerce, direct). They need:
- Net sales and gross margin per market and per quarter
- Top-N customers / products per market
- Year-over-year growth metrics
- Automated, refreshed reports

### Skills demonstrated
- Building a star-shaped analytical schema (or working on AtliQ's existing one)
- Encapsulating "net sales" formula in a UDF
- Stored procedures for parameterized reports
- Views as the analyst-facing layer
- Window functions for top-N + YoY

### Sample artifacts you'll build

#### A UDF for "net sales"
```sql
DELIMITER $$
CREATE FUNCTION net_sales(gross DECIMAL(12,2), discount_pct DECIMAL(4,2))
RETURNS DECIMAL(12,2)
DETERMINISTIC
NO SQL
BEGIN
    RETURN gross * (1 - discount_pct / 100);
END$$
DELIMITER ;
```

#### A view of clean transactions
```sql
CREATE OR REPLACE VIEW v_sales_clean AS
SELECT s.date, s.product_id, s.customer_id, s.market,
       net_sales(s.gross_price * s.quantity, c.discount_pct) AS net_sales
FROM fact_sales_monthly s
JOIN dim_customer c ON s.customer_id = c.customer_id;
```

#### A procedure for top-N customers per market
```sql
DELIMITER $$
CREATE PROCEDURE top_customers_per_market(IN n INT, IN year_in INT)
BEGIN
    WITH ranked AS (
        SELECT market, customer_id,
               SUM(net_sales) AS total,
               ROW_NUMBER() OVER (PARTITION BY market ORDER BY SUM(net_sales) DESC) AS rk
        FROM v_sales_clean
        WHERE YEAR(date) = year_in
        GROUP BY market, customer_id
    )
    SELECT * FROM ranked WHERE rk <= n;
END$$
DELIMITER ;

CALL top_customers_per_market(5, 2025);
```

#### A YoY growth view
```sql
CREATE VIEW v_yoy_market AS
SELECT market,
       YEAR(date) AS yr,
       SUM(net_sales) AS total_net_sales,
       LAG(SUM(net_sales)) OVER (PARTITION BY market ORDER BY YEAR(date)) AS prev,
       (SUM(net_sales) - LAG(SUM(net_sales)) OVER (PARTITION BY market ORDER BY YEAR(date)))
            / LAG(SUM(net_sales)) OVER (PARTITION BY market ORDER BY YEAR(date)) * 100 AS yoy_pct
FROM v_sales_clean
GROUP BY market, YEAR(date);
```

### Deliverables
- 4–6 views
- 2–4 stored procedures
- 1–2 UDFs
- A `README.md` documenting each (purpose, parameters, sample query)
- Sample output screenshots
- ERD of the schema

This is a strong "data analyst" portfolio piece. Recruiters in retail/CPG read it.

---

## Project B — Supply Chain Analytics & Optimisation

### Domain
The same company tracks **forecast accuracy** — how well demand was predicted vs. actual sales. They need:
- Forecast Error % per product per month
- Net Error and Absolute Error metrics
- Aggregations across different hierarchies (market, segment, region)
- Helper tables to speed up dashboards
- Triggers to keep them fresh
- Query plans tuned for sub-second response

### Skills demonstrated
- Helper / aggregation tables
- Triggers (or scheduled events) for refresh
- Index strategy + EXPLAIN-based tuning
- Window functions for moving averages of error
- Performance comparisons (before/after benchmarks)

### Sample artifacts

#### Forecast error calculation
```sql
SELECT customer_id, product_id, date,
       sold_quantity, forecast_quantity,
       forecast_quantity - sold_quantity AS net_error,
       ABS(forecast_quantity - sold_quantity) AS abs_error,
       ABS(forecast_quantity - sold_quantity) / NULLIF(sold_quantity, 0) * 100 AS abs_error_pct
FROM fact_act_est;
```

#### A helper table for fast dashboard reads
```sql
CREATE TABLE helper_forecast_accuracy_monthly AS
SELECT YEAR(date) AS yr, MONTH(date) AS mn, market,
       SUM(forecast_quantity) AS forecast_units,
       SUM(sold_quantity) AS sold_units,
       SUM(ABS(forecast_quantity - sold_quantity)) AS abs_error_units
FROM fact_act_est
GROUP BY YEAR(date), MONTH(date), market;

CREATE INDEX idx_yr_mn_market ON helper_forecast_accuracy_monthly(yr, mn, market);
```

#### Trigger to refresh on insert
(In practice you'd batch this — but as a teaching pattern):
```sql
DELIMITER $$
CREATE TRIGGER refresh_helper_after_insert
AFTER INSERT ON fact_act_est
FOR EACH ROW
BEGIN
    -- naive: re-aggregate the (year, month, market) slice
    DELETE FROM helper_forecast_accuracy_monthly
    WHERE yr = YEAR(NEW.date) AND mn = MONTH(NEW.date) AND market = NEW.market;

    INSERT INTO helper_forecast_accuracy_monthly
    SELECT YEAR(date), MONTH(date), market,
           SUM(forecast_quantity), SUM(sold_quantity),
           SUM(ABS(forecast_quantity - sold_quantity))
    FROM fact_act_est
    WHERE YEAR(date) = YEAR(NEW.date) AND MONTH(date) = MONTH(NEW.date) AND market = NEW.market
    GROUP BY YEAR(date), MONTH(date), market;
END$$
DELIMITER ;
```

#### Query optimisation steps
1. Run `EXPLAIN <query>` — find the bottleneck
2. Add an index on the highest-cardinality filter column
3. Re-run; compare row scans
4. Move repeated subqueries into CTEs / views
5. Pre-aggregate hot dashboards into helper tables (with refresh strategy)
6. Document the before/after timings in your README

### Deliverables
- Helper tables + triggers / events
- Indexes added (with rationale)
- `EXPLAIN` outputs before / after, side by side
- Stored procedure for the main forecast accuracy report
- A `performance.md` documenting the optimization journey

Recruiters in **supply chain analytics** specifically value this kind of project — it's exactly what working analysts do.

---

## Repo template (for either project)

```
finance-top-n/
├── data/
│   └── seed.sql                        # schema + sample data
├── sql/
│   ├── 01_schema.sql
│   ├── 02_views.sql
│   ├── 03_procedures.sql
│   ├── 04_functions.sql
│   └── 05_indexes.sql
├── notebooks/                          # optional pandas verification
│   └── verify.ipynb
├── reports/
│   └── insights.md                     # business answers from running the procedures
├── README.md
└── ERD.png
```

---

## Self-check

- [ ] Project A: have I defined at least one UDF + 2 procedures + 3 views?
- [ ] Project A: do my views encapsulate the "net sales" formula so analysts don't reinvent it?
- [ ] Project B: do I have a helper table and a refresh strategy?
- [ ] Project B: did I document an EXPLAIN before-and-after?
- [ ] Both: do my READMEs include sample CALL / SELECT statements with output?
- [ ] Both: posted on LinkedIn with screenshot and 3-bullet "what I learned"?
- [ ] Both: ERD image included in the repo?
