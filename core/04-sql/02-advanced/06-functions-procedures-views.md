# Advanced 6 — User-Defined Functions, Stored Procedures, Views

## Lectures covered
- User-Defined Functions (UDF)
- Stored Procedures
- Database Views

---

## 1. Views — saved queries you can SELECT from

A **view** is a stored `SELECT` statement that you can query like a table.

### Creating
```sql
CREATE VIEW v_top_movies AS
SELECT m.title, m.imdb_rating, f.revenue
FROM movies m
JOIN financials f ON m.movie_id = f.movie_id
WHERE m.imdb_rating > 8;
```

### Using
```sql
SELECT * FROM v_top_movies WHERE revenue > 100000000;
```

### Dropping / replacing
```sql
DROP VIEW v_top_movies;
CREATE OR REPLACE VIEW v_top_movies AS ...;
```

### Why use views
- **Encapsulate complex joins** — analysts SELECT from views, not raw tables
- **Security** — give a user access to a view that filters PII out
- **Stable interface** — underlying tables can change, view stays the same

### Materialized views
A **materialized view** stores the result on disk (refreshed on schedule or on demand). Fast reads, stale writes.

MySQL doesn't support materialized views natively. Workarounds:
- A scheduled `INSERT INTO summary_table SELECT ...` job
- An event (covered in `08-triggers-events-indexes.md`)
- A real warehouse (Snowflake, Postgres+pg_cron) supports them natively

---

## 2. Stored Procedures — saved SQL blocks (no return value)

```sql
DELIMITER $$

CREATE PROCEDURE get_top_movies(IN min_rating DECIMAL(2,1))
BEGIN
    SELECT title, imdb_rating
    FROM movies
    WHERE imdb_rating >= min_rating
    ORDER BY imdb_rating DESC;
END$$

DELIMITER ;

CALL get_top_movies(8.5);
```

### Why `DELIMITER $$`?
Inside the procedure, statements end with `;`. We need a different delimiter for the *outer* `CREATE PROCEDURE` to know when the body ends.

### Parameter modes
- `IN` — read-only input (default)
- `OUT` — written by the procedure, returned to caller
- `INOUT` — both

```sql
DELIMITER $$
CREATE PROCEDURE add_two(IN a INT, IN b INT, OUT result INT)
BEGIN
    SET result = a + b;
END$$
DELIMITER ;

CALL add_two(2, 3, @sum);
SELECT @sum;          -- 5
```

### Conditional logic + loops
```sql
DELIMITER $$
CREATE PROCEDURE classify(IN movie_id INT)
BEGIN
    DECLARE r DECIMAL(2,1);
    SELECT imdb_rating INTO r FROM movies WHERE movie_id = movie_id;

    IF r >= 9 THEN
        SELECT "excellent" AS rating;
    ELSEIF r >= 7 THEN
        SELECT "good" AS rating;
    ELSE
        SELECT "average" AS rating;
    END IF;
END$$
DELIMITER ;
```

### When to use stored procedures
- **Reusable analytics** — same complex query called from many places
- **Encapsulating business rules** in one place
- **Performance** — server-side; no round-trips for multi-step logic
- **Permission control** — grant CALL access without granting raw table reads

### When *not* to use them
- Application logic that should live in app code (Python, Node)
- Anything that needs to be tested with unit tests
- Anything that crosses microservice boundaries

Modern teams keep procedures for **pure-SQL workflows** (data warehousing) and put business logic in app code.

---

## 3. User-Defined Functions (UDFs)

A **function** in SQL is like a procedure but **returns a single value** and can be used inside SELECT.

```sql
DELIMITER $$
CREATE FUNCTION movie_age(release_year INT)
RETURNS INT
DETERMINISTIC
READS SQL DATA
BEGIN
    RETURN YEAR(CURDATE()) - release_year;
END$$
DELIMITER ;

SELECT title, movie_age(release_year) AS age FROM movies;
```

### Required clauses
- `RETURNS <type>` — the return type
- `DETERMINISTIC` — same input always gives same output (lets MySQL cache)
- `READS SQL DATA` / `MODIFIES SQL DATA` / `NO SQL` — what side effects, if any

### When to use UDFs
- Reusable column calculations (e.g., "fiscal week" from a date)
- Domain-specific formulas (BMI, tax, profit margin)

---

## 4. Procedure vs Function vs View — which to use

| You want to... | Use |
|---|---|
| Save a complex SELECT for reuse | View |
| Compute one value per row in a SELECT | Function |
| Encapsulate a multi-step DML routine | Procedure |
| Filter sensitive columns from analysts | View |
| Run on a schedule | Event (next file) |

---

## 5. Real example — Codebasics' "Finance Top-N" project preview

A typical advanced-SQL project shape:

### Step 1 — A view for clean, joined data
```sql
CREATE VIEW v_sales_clean AS
SELECT s.sale_id, s.date, s.amount, c.name AS customer, p.name AS product
FROM sales s
JOIN customers c ON s.customer_id = c.id
JOIN products p ON s.product_id = p.id
WHERE s.status = "confirmed";
```

### Step 2 — A procedure for top-N by category
```sql
DELIMITER $$
CREATE PROCEDURE top_n_per_category(IN n INT)
BEGIN
    SELECT category, name, total
    FROM (
        SELECT p.category, p.name, SUM(s.amount) AS total,
               ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(s.amount) DESC) AS rk
        FROM v_sales_clean s
        JOIN products p ON s.product = p.name
        GROUP BY p.category, p.name
    ) t
    WHERE rk <= n;
END$$
DELIMITER ;

CALL top_n_per_category(3);
```

This is the kind of build that's portfolio-worthy and walks the line between SQL and "data engineering."

---

## 6. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot `DELIMITER` change | "no statement after BEGIN" | wrap procedure body with `DELIMITER $$ ... $$ DELIMITER ;` |
| Procedure with side effects + `DETERMINISTIC` | wrong cache | be honest about side effects |
| View with `ORDER BY` | useless — the consumer can re-order | drop the ORDER BY in the view |
| Procedure too large | hard to test | split into smaller procedures |
| Returning multiple result sets from a function | not supported | use a procedure |

## Self-check

- [ ] Difference between view, function, procedure?
- [ ] When would you use a procedure over a view?
- [ ] What does `DELIMITER` do in CREATE PROCEDURE?
- [ ] What's the difference between `IN`, `OUT`, `INOUT` parameters?
- [ ] What's `DETERMINISTIC` and why does it matter?
- [ ] Write a function `discount(amount DECIMAL, pct DECIMAL)` that returns the post-discount amount.
- [ ] Why doesn't MySQL have native materialized views, and how do teams emulate them?
