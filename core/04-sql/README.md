# Module 4 — SQL for Data Science (Beginner → Advanced)

> **Status**: Basics ✅ Complete (100%) · Advanced 🔒 Locked (after Math/Stats)
> **Domain projects**: Finance & Top-N Insights · Supply Chain Analytics & Optimization

## Why SQL still matters in 2025

Despite Python's rise, SQL is *the* language of data in every company:
- Every BI tool runs SQL underneath (Looker, Tableau, Power BI)
- Every modern data warehouse speaks SQL (Snowflake, BigQuery, Redshift, Databricks)
- The "Data Analyst" interview is mostly SQL
- "Data Scientist" interviews include SQL rounds even for ML-heavy roles
- LLM agents now write SQL — but you have to *verify* what they wrote

Bottom line: in 2025, SQL is *more* important, not less, because it's the universal contract between analytical layers.

## Folder layout

```
04-sql/
├── README.md                           ← you are here
├── 01-basics/                          ← Sections 1–4 (data retrieval, joins)
│   ├── README.md
│   ├── 01-installation-import.md
│   ├── 02-single-table-retrieval.md
│   ├── 03-multiple-tables-joins.md
│   └── 04-final-quiz.md
└── 02-advanced/                        ← Sections 5–end (CTE, windows, perf, projects)
    ├── README.md
    ├── 01-subqueries-cte.md
    ├── 02-data-types.md
    ├── 03-keys-erd-normalization.md
    ├── 04-dml-statements.md
    ├── 05-data-warehouse-etl.md
    ├── 06-functions-procedures-views.md
    ├── 07-window-functions.md
    ├── 08-triggers-events-indexes.md
    └── 09-projects.md
```

## Curriculum (verbatim from public outline)

### Beginner sections
- Welcome / "How much SQL is needed?"
- Install MySQL · Import movies dataset
- Retrieve text-based: `SELECT`, `WHERE`, `DISTINCT`, `LIKE`
- Retrieve numeric-based: `BETWEEN`, `IN`, `ORDER BY`, `LIMIT`, `OFFSET`
- Aggregates: `MIN`, `MAX`, `AVG`, `GROUP BY`, `HAVING`
- Calculated columns: `IF`, `CASE`, `YEAR`, `CURYEAR`
- Joins: `INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`
- Multi-table joins · Final Quiz

### Advanced sections (from brochure)
- Subqueries: `ANY`, `ALL`, correlated subquery
- Common Table Expressions (CTE) — including recursive
- Database normalization & data integrity
- Data types: numeric, string, date, JSON, spatial
- Keys: PK, FK · ERD diagrams
- DML: `INSERT`, `UPDATE`, `DELETE`
- ETL · OLAP vs OLTP · Data Catalog
- Star vs snowflake · Fact vs dimension
- User-defined functions, stored procedures, views
- **Window functions**: `OVER`, `ROW_NUMBER`, `RANK`, `DENSE_RANK`
- Triggers, events, accounts/privileges, indexes
- Projects: Finance & Top-N Insights · Supply Chain Analytics

## Module-level goal

After this module:
- Write any analytical query (filter, aggregate, join, window) without docs
- Design a normalized schema and an ERD for a small business
- Write stored procedures + UDFs for reusable analytics
- Tune slow queries with indexes
- Pass any "SQL round" entry-level data interview

## Self-check

- [ ] Can I write a query that ranks the top 3 products per category? (window function)
- [ ] Difference between `WHERE` and `HAVING`?
- [ ] When would I pick a CTE over a subquery?
- [ ] What's a star schema and why is it good for analytics?
- [ ] How do indexes speed up reads, and what's the cost?
- [ ] When should I write a stored procedure vs an application-side function?
- [ ] Can I draw an ERD for a 5-table e-commerce schema?
