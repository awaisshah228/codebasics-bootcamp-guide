# Advanced 5 — Data Warehouse, ETL, OLAP/OLTP, Schemas

## Lectures covered
- ETL · Data Warehouse
- OLTP vs OLAP · Data Catalog
- Fact vs Dimension Table
- Star vs Snowflake Schema
- Data Import (recap)
- "Simplified: What is Kanban?"

---

## 1. OLTP vs OLAP — two different worlds

| | OLTP | OLAP |
|---|---|---|
| **Purpose** | Run the business (writes) | Understand the business (reads) |
| **Workload** | Many small txns | Few large analytical queries |
| **Schema** | Highly normalized (3NF+) | Denormalized (star/snowflake) |
| **Latency** | Milliseconds | Seconds–minutes |
| **Examples** | MySQL behind a checkout, Postgres behind an app | Snowflake, BigQuery, Redshift, Databricks |

The same company often runs both: OLTP captures, OLAP analyzes. ETL bridges them.

---

## 2. ETL — Extract / Transform / Load

```
Sources             Staging               Warehouse
─────────           ───────               ─────────
ERP DB     ──┐
CRM DB     ──┤  E   raw tables   T   clean,    L   star schema
S3 files   ──┤ ───>            ───>  conformed ───>  facts + dims
APIs       ──┤      (sometimes  T               L
Logs       ──┘       skipped)
```

### Extract
Pull data from sources. Patterns: full snapshots (slow), incremental (CDC), event streams (Kafka).

### Transform
- Type cleaning (string → date, etc.)
- Deduplication
- Joining + denormalizing
- Aggregation
- Business-rule application (e.g., "exclude internal test orders")

### Load
Write into the warehouse, partitioned + indexed for analytical access.

### ETL vs ELT
- **ETL**: transform *before* loading. Classic. Used when warehouse compute is expensive.
- **ELT**: load raw, then transform inside the warehouse using SQL. Modern (Snowflake, BigQuery). Tools like dbt embody this.

For Codebasics' projects, you'll see ETL inside Python (pandas) writing to MySQL — small but real.

---

## 3. Data warehouse vs data lake vs data mart

| | Data warehouse | Data lake | Data mart |
|---|---|---|---|
| **Format** | structured (tables) | any (JSON, Parquet, CSV, images) | structured |
| **Scope** | enterprise-wide analytics | raw catch-all | one team / domain |
| **Tools** | Snowflake, BigQuery, Redshift | S3, ADLS, GCS | a slice of the warehouse |
| **Schema** | upfront (schema-on-write) | flexible (schema-on-read) | upfront |

**Data lakehouse** = lake + warehouse hybrid (Databricks, Iceberg, Delta Lake). Modern dominant pattern.

---

## 4. Star schema — the analytical default

### Layout
```
                 dim_customer
                       │
dim_date  ──── fact_sales ──── dim_product
                       │
                  dim_store
```

- **Fact table** in the middle: numerical, additive measures + foreign keys to dimensions
- **Dimension tables** around it: descriptive, denormalized

### Fact table
```sql
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    date_id INT,
    customer_id INT,
    product_id INT,
    store_id INT,
    quantity INT,
    revenue DECIMAL(12, 2),
    discount DECIMAL(12, 2)
);
```

Measures (`quantity`, `revenue`, `discount`) are summable — that's the point of a fact table.

### Dimension table
```sql
CREATE TABLE dim_product (
    product_id INT PRIMARY KEY,
    sku VARCHAR(50),
    name VARCHAR(100),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    brand VARCHAR(50),
    list_price DECIMAL(10, 2)
);
```

Descriptive attributes, low row count, denormalized (category + sub-category live in the same row).

### Why star wins for OLAP
- Few joins per query (just fact ↔ dim)
- Easy to understand
- Fast on columnar warehouses (each column compressed)

---

## 5. Snowflake schema — normalized dimensions

```
                              dim_subcategory
                                    │
dim_product ─── dim_category ───────┘
```

A `dim_product` references `dim_category`, which references `dim_subcategory`. More normalized; saves disk; harder to query.

### Star vs snowflake — pick which?
- **Star** (default for analytics): fewer joins, faster, more readable
- **Snowflake**: when dimension data is huge and changes often (e.g., complex product hierarchies)

Most teams default to star and only normalize where storage / change-frequency forces it.

---

## 6. Fact table flavors

### Transactional fact
One row per business event (`fact_sales`, `fact_clicks`). Most common.

### Periodic snapshot
One row per entity per period (`fact_account_balance_daily`). Useful for time-series.

### Accumulating snapshot
One row per process, updated as it progresses (`fact_order_lifecycle` with `placed`, `shipped`, `delivered`, `returned` timestamps).

### Factless fact table
A fact table with no measures, just FK relationships. Used to track *occurrences* (e.g., student attendance — what was attended, by whom, when).

---

## 7. Slowly Changing Dimensions (SCDs) — handling history in dim tables

| Type | Behavior | When |
|---|---|---|
| **SCD Type 1** | Overwrite (no history) | Typo fixes |
| **SCD Type 2** | New row per change, with `valid_from` / `valid_to` columns | Track history (most common) |
| **SCD Type 3** | One column for "previous value" | Rare; partial history |

### SCD2 example
```sql
CREATE TABLE dim_customer (
    customer_sk BIGINT PRIMARY KEY,            -- surrogate key (changes per version)
    customer_id INT,                            -- natural / business key (stable)
    name VARCHAR(100),
    tier VARCHAR(20),
    valid_from DATE,
    valid_to DATE,
    is_current BOOLEAN
);
```

When a customer's tier changes, insert a new row with the new tier and bump the previous row's `valid_to`. Fact rows reference `customer_sk` so a 2024 sale stays linked to the correct *historical* customer.

---

## 8. Data Catalog — knowing what's where

A **data catalog** is a metadata directory of every dataset in your org:
- Table name + schema + owner
- Column descriptions
- Data lineage (what feeds into what)
- Data classification (PII, financial, etc.)

Tools: **DataHub** (open-source), **Atlan**, **Alation**, **Collibra**.

For your bootcamp projects: even a `data_dictionary.md` in the repo is a starter catalog. Recruiters notice this.

---

## 9. "Simplified: What is Kanban?"

Codebasics drops in a 5-min Kanban primer here because their stats project (AtliQo Bank, Module 5) uses Kanban / JIRA.

### Kanban in 60 seconds
- A board with columns: **To Do · In Progress · Done**
- Cards (tasks) flow left → right
- WIP limits prevent multitasking
- Daily 10-minute standup: "what did I do yesterday / today / blockers?"

It's the lightest-weight project management framework — perfect for a personal bootcamp.

### Apply to your bootcamp
A simple Trello / Notion / GitHub Projects board with columns: **Up Next**, **Studying**, **Practicing**, **Done**. Add each module / project / practice room as a card.

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Storing analytical metrics in OLTP | slows the app | replicate to a warehouse |
| Designing facts without grain | sums double-count | define exactly what one row means |
| Snowflaking dimensions prematurely | extra joins, slower queries | start star, snowflake only when needed |
| No data dictionary | "what does this column mean?" Discord questions | document at least every fact table |
| ETL with no monitoring | silent failures | log row counts, alert on anomalies |

## Self-check

- [ ] Difference between OLTP and OLAP?
- [ ] Walk through the ETL pipeline for a sales dashboard.
- [ ] Star vs snowflake — when to pick which?
- [ ] What's the "grain" of a fact table?
- [ ] Three types of SCD and when to use each?
- [ ] What's a data catalog and why do orgs need one?
- [ ] Set up a Kanban board for your bootcamp progress.
