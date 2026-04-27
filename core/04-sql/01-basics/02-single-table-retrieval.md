# Section 2 — Data Retrieval: Single Table

## Lectures covered
- Retrieve Data Using Text Query (`SELECT`, `WHERE`, `DISTINCT`, `LIKE`)
- Retrieve Data Using Numeric Query (`BETWEEN`, `IN`, `ORDER BY`, `LIMIT`, `OFFSET`)
- Summary Analytics (`MIN`, `MAX`, `AVG`, `GROUP BY`)
- `HAVING` clause
- Calculated Columns (`IF`, `CASE`, `YEAR`, `CURYEAR`)
- "The Data God's Blessing" + Quiz

---

## 1. The mental model — order of execution

This is the single most useful thing to memorize:

```
1.  FROM        which table(s)
2.  WHERE       filter rows
3.  GROUP BY    bucket rows
4.  HAVING      filter buckets
5.  SELECT      pick / compute columns
6.  ORDER BY    sort
7.  LIMIT       trim
```

You **write** `SELECT` first; the engine **executes** it 5th. This explains 90% of "why is this not working?" moments.

---

## 2. SELECT, WHERE, DISTINCT, LIKE

### The basic shape
```sql
SELECT col1, col2
FROM table
WHERE condition
ORDER BY col1
LIMIT 10;
```

### Selecting columns
```sql
SELECT * FROM movies;
SELECT title, release_year FROM movies;
SELECT title AS movie_name, release_year AS year FROM movies;
SELECT title, release_year, release_year * 2 FROM movies;   -- expressions allowed
```

### Filtering with WHERE
```sql
SELECT * FROM movies WHERE industry = "Bollywood";
SELECT * FROM movies WHERE imdb_rating > 8.0;
SELECT * FROM movies WHERE imdb_rating BETWEEN 7 AND 9;     -- inclusive
SELECT * FROM movies WHERE industry IN ("Bollywood", "Hollywood");
SELECT * FROM movies WHERE industry NOT IN ("Bollywood");
SELECT * FROM movies WHERE imdb_rating IS NULL;
SELECT * FROM movies WHERE imdb_rating IS NOT NULL;
```

### Combining conditions
```sql
SELECT * FROM movies
WHERE industry = "Bollywood" AND imdb_rating > 8;

SELECT * FROM movies
WHERE industry = "Bollywood" OR imdb_rating > 9;

SELECT * FROM movies
WHERE NOT (industry = "Bollywood" AND release_year < 2000);
```

> Use parentheses to be explicit about precedence: `AND` binds tighter than `OR`.

### DISTINCT — unique values
```sql
SELECT DISTINCT industry FROM movies;
SELECT DISTINCT industry, language_id FROM movies;          -- distinct combinations
```

### LIKE — pattern match
```sql
SELECT * FROM movies WHERE title LIKE "The%";          -- starts with "The"
SELECT * FROM movies WHERE title LIKE "%love%";        -- contains "love"
SELECT * FROM movies WHERE title LIKE "_oy";           -- single-char wildcard: matches "Boy", "Toy"
```
Wildcards: `%` = any number of chars; `_` = exactly one.

### Case-insensitive matching
MySQL is **case-insensitive by default for non-binary string columns**. To force case-sensitivity:
```sql
SELECT * FROM movies WHERE BINARY title = "the godfather";
```

---

## 3. BETWEEN, IN, ORDER BY, LIMIT, OFFSET

### Range
```sql
SELECT * FROM movies WHERE release_year BETWEEN 2010 AND 2020;
```

### Set membership
```sql
SELECT * FROM movies WHERE studio IN ("Marvel Studios", "Pixar");
```

### Ordering
```sql
SELECT title, imdb_rating FROM movies ORDER BY imdb_rating DESC;
SELECT * FROM movies ORDER BY industry ASC, imdb_rating DESC;
```

### Pagination
```sql
SELECT * FROM movies ORDER BY imdb_rating DESC LIMIT 10;
SELECT * FROM movies ORDER BY imdb_rating DESC LIMIT 10 OFFSET 20;     -- rows 21–30
```

> Always pair `LIMIT` with an explicit `ORDER BY` — without ordering, the "first 10" rows are *whatever the engine returns*, which is non-deterministic.

---

## 4. Summary analytics

### Aggregate functions
```sql
SELECT MIN(imdb_rating) FROM movies;
SELECT MAX(imdb_rating) FROM movies;
SELECT AVG(imdb_rating) FROM movies;
SELECT COUNT(*) FROM movies;
SELECT COUNT(imdb_rating) FROM movies;       -- ignores NULLs
SELECT COUNT(DISTINCT industry) FROM movies;
SELECT SUM(revenue) FROM financials;
```

### COUNT(*) vs COUNT(col) — the gotcha
- `COUNT(*)` — number of rows (NULLs included)
- `COUNT(col)` — number of *non-NULL* values in that column

### GROUP BY — collapse rows by a key
```sql
-- avg rating per industry
SELECT industry, AVG(imdb_rating) AS avg_rating
FROM movies
GROUP BY industry;

-- multi-key
SELECT industry, release_year, COUNT(*) AS movie_count
FROM movies
GROUP BY industry, release_year;
```

### Rule: every non-aggregated SELECT column must appear in GROUP BY
```sql
-- WRONG — title isn't aggregated and not in GROUP BY
SELECT title, AVG(imdb_rating) FROM movies GROUP BY industry;
```

---

## 5. HAVING — filtering groups

`WHERE` filters rows *before* grouping. `HAVING` filters groups *after*.

```sql
-- studios with more than 5 movies
SELECT studio, COUNT(*) AS n
FROM movies
GROUP BY studio
HAVING n > 5;

-- industries whose average rating exceeds 7
SELECT industry, AVG(imdb_rating) AS avg_r
FROM movies
GROUP BY industry
HAVING avg_r > 7;
```

> Common mistake: trying `WHERE COUNT(*) > 5` — this fails because `COUNT(*)` is computed *after* `WHERE`.

---

## 6. Calculated columns

### IF (single condition → 2 outcomes)
```sql
SELECT title,
       imdb_rating,
       IF(imdb_rating > 8, "good", "average") AS rating_tier
FROM movies;
```

### CASE (multi-branch, more general)
```sql
SELECT title, imdb_rating,
       CASE
         WHEN imdb_rating >= 9 THEN "excellent"
         WHEN imdb_rating >= 7 THEN "good"
         WHEN imdb_rating >= 5 THEN "average"
         ELSE "poor"
       END AS rating_tier
FROM movies;
```

### Date functions — `YEAR`, `CURDATE`, etc.
```sql
SELECT title, release_year, YEAR(CURDATE()) - release_year AS age_years
FROM movies;

SELECT title FROM movies WHERE release_year = YEAR(CURDATE());

-- common date functions
SELECT NOW();              -- current datetime
SELECT CURDATE();          -- today's date
SELECT YEAR(NOW()), MONTH(NOW()), DAY(NOW());
SELECT DATE_ADD(NOW(), INTERVAL 7 DAY);
SELECT DATEDIFF("2025-12-31", NOW());
```

### Numeric / string helpers
```sql
SELECT ROUND(avg_rating, 2), CEIL(price), FLOOR(price), ABS(-5);
SELECT UPPER(title), LOWER(title), CONCAT(title, " (", release_year, ")");
SELECT SUBSTRING(title, 1, 5);
SELECT REPLACE(title, "The ", "");
```

---

## 7. Putting it together — typical analyst query

```sql
-- "Top 5 highest-grossing Bollywood movies released after 2010,
--  excluding those without revenue data"
SELECT m.title, m.release_year, f.revenue
FROM movies m
JOIN financials f ON m.movie_id = f.movie_id   -- (joins next section)
WHERE m.industry = "Bollywood"
  AND m.release_year > 2010
  AND f.revenue IS NOT NULL
ORDER BY f.revenue DESC
LIMIT 5;
```

> "Question to query" translation is the core analyst skill. Practice it on Codebasics' exercises until it's reflexive.

---

## 8. "The Data God's Blessing" — Codebasics' aside

Cinematic side-story; the takeaway: **read every business question twice before writing SQL**. Half of mis-written queries are mis-read questions, not bad SQL.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| `WHERE col = NULL` | always evaluates to NULL → 0 rows | `WHERE col IS NULL` |
| Using `WHERE COUNT(*) > 5` | `COUNT` not yet computed at WHERE | use `HAVING` |
| `SELECT title, AVG(rating) GROUP BY industry` | mixes raw + aggregate | add `title` to `GROUP BY` or remove from `SELECT` |
| `LIMIT 10` without `ORDER BY` | non-deterministic results | always order |
| `LIKE 'foo'` thinking it's pattern | needs `%` to be a pattern | `LIKE 'foo%'` |
| Comparing strings with `BINARY` accidentally | unwanted case-sensitivity | drop `BINARY` |

## Self-check

- [ ] What's the order of execution: WHERE / GROUP BY / SELECT / ORDER BY / LIMIT?
- [ ] Difference between `COUNT(*)` and `COUNT(col)`?
- [ ] Difference between `WHERE` and `HAVING`?
- [ ] Why does `WHERE col = NULL` return zero rows?
- [ ] Write a query: average rating per industry, only industries with >3 movies, sorted descending.
- [ ] Convert "Show me movies older than 20 years" into SQL using `YEAR(CURDATE())`.
- [ ] How do I bucket ratings into "good / average / poor" with one SELECT?
