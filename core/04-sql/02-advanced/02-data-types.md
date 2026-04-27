# Advanced 2 — MySQL Data Types

## Lectures covered
- Data Types: Numeric (`INT`, `DECIMAL`, `FLOAT`, `DOUBLE`)
- Data Types: String (`VARCHAR`, `CHAR`, `ENUM`)
- Data Types: Date / Time (`DATETIME`, `DATE`, `TIME`, `YEAR`, `TIMESTAMP`)
- Data Types: JSON, Spatial (`JSON`, `GEOMETRY`)

---

## Why types matter

Types affect:
- **Storage size** — `BIGINT` vs `TINYINT` is 7 bytes per row
- **Index efficiency** — `VARCHAR(255)` indexes slower than `INT`
- **Correctness** — `FLOAT` for money is a bug
- **Validation** — `ENUM` rejects bad values at insert

Picking the right type is half schema design.

---

## 1. Numeric types

| Type | Bytes | Range (signed) | When to use |
|---|---|---|---|
| `TINYINT` | 1 | -128 to 127 | flags, very small counts |
| `SMALLINT` | 2 | -32k to 32k | medium counts |
| `MEDIUMINT` | 3 | -8M to 8M | rarely used |
| `INT` (`INTEGER`) | 4 | ±2.1B | default for IDs and counts |
| `BIGINT` | 8 | ±9.2 × 10^18 | huge IDs, sums of small values |
| `DECIMAL(p, s)` | varies | exact | money, percentages — anything where precision matters |
| `FLOAT` | 4 | approximate | scientific data; avoid for money |
| `DOUBLE` | 8 | approximate | scientific data |

### `DECIMAL` for money
```sql
CREATE TABLE orders (
    id INT PRIMARY KEY,
    amount DECIMAL(12, 2)         -- 12 digits, 2 after decimal point: up to 9,999,999,999.99
);
```

`FLOAT`/`DOUBLE` round (`0.1 + 0.2 ≠ 0.3`). Don't use them for money.

### `UNSIGNED` doubles the positive range
```sql
INT UNSIGNED         -- 0 to 4.2B (good for IDs)
TINYINT UNSIGNED     -- 0 to 255
```

### `AUTO_INCREMENT`
```sql
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY
```
Each insert without specifying `id` gets the next number.

---

## 2. String types

| Type | Storage | When to use |
|---|---|---|
| `CHAR(n)` | fixed n chars | rare; only for truly fixed-width (e.g., ISO codes) |
| `VARCHAR(n)` | variable, up to n | default for short strings (names, emails) |
| `TEXT` | variable, up to 64KB | long free-form text |
| `MEDIUMTEXT` | up to 16MB | very long content |
| `LONGTEXT` | up to 4GB | rarely needed; use a file/object store |
| `ENUM(...)` | 1–2 bytes | fixed set of allowed values |
| `SET(...)` | 1–8 bytes | multi-valued enum (like checkboxes) |
| `BINARY(n)` / `VARBINARY` / `BLOB` | byte data | files, hashes — but prefer object storage |

### `VARCHAR(255)` is *not* always best
- `VARCHAR(10)` for a 5-char column wastes nothing on disk
- But `VARCHAR(255)` indexes are bigger; pick the smallest reasonable cap
- For unbounded text → `TEXT` (allows full-text search via FULLTEXT index)

### `ENUM` — for restricted vocabularies
```sql
CREATE TABLE bookings (
    id INT PRIMARY KEY,
    status ENUM("pending", "confirmed", "cancelled") DEFAULT "pending"
);
```
- Stored as a small int internally
- Fast and safe — invalid values are rejected
- **Downside**: changing the list requires `ALTER TABLE`

### Charset & collation
- Use `utf8mb4` (real UTF-8, supports emoji)
- Default collation: `utf8mb4_0900_ai_ci` (accent-insensitive, case-insensitive)
- Avoid `utf8` (it's actually 3-byte UTF-8 — broken for emoji)

```sql
CREATE TABLE t (
    name VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
);
```

---

## 3. Date and time types

| Type | Format | Use |
|---|---|---|
| `DATE` | `YYYY-MM-DD` | calendar date |
| `TIME` | `HH:MM:SS` | time of day, no date |
| `DATETIME` | `YYYY-MM-DD HH:MM:SS` | full timestamp, no timezone |
| `TIMESTAMP` | UTC, range 1970–2038 | logging, "updated at" |
| `YEAR` | `YYYY` | year only |

### `DATETIME` vs `TIMESTAMP` — important
- `DATETIME` stores wall-clock time **as you wrote it** (no timezone awareness)
- `TIMESTAMP` converts to UTC for storage and back to session timezone on read
- `TIMESTAMP` is bounded to 2038 (Y2038 problem) — for new schemas, `DATETIME` + explicit UTC is safer

### Auto-fill for created/updated
```sql
CREATE TABLE posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### Common date functions (refresher)
```sql
NOW()                                  -- current datetime
CURDATE() / CURRENT_DATE               -- today's date
CURTIME() / CURRENT_TIME               -- current time
YEAR(d), MONTH(d), DAY(d), HOUR(d)
DATE_ADD(d, INTERVAL 7 DAY)
DATE_SUB(d, INTERVAL 1 MONTH)
DATEDIFF(end, start)
TIMESTAMPDIFF(MINUTE, start, end)
DATE_FORMAT(d, "%Y-%m")               -- "2025-04"
STR_TO_DATE("2025-04-15", "%Y-%m-%d") -- parse string into DATE
```

---

## 4. JSON

Modern MySQL stores JSON natively (since 5.7) — typed, validated, and queryable.

```sql
CREATE TABLE events (
    id INT PRIMARY KEY,
    payload JSON
);

INSERT INTO events VALUES (1, '{"user": "Awais", "action": "login", "ip": "10.0.0.1"}');

-- access fields
SELECT payload->"$.user", payload->>"$.action"      -- ->> unquotes the string
FROM events;

-- filter
SELECT * FROM events WHERE payload->>"$.action" = "login";

-- update a field
UPDATE events SET payload = JSON_SET(payload, "$.user", "Awais Shah") WHERE id = 1;
```

Useful for semi-structured event logs / API responses. **Don't use it as a substitute for normalization** — if a field is queried often, give it its own column.

---

## 5. Spatial types — geometry & GIS

For maps, locations, polygons.

```sql
CREATE TABLE places (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    location POINT,
    SPATIAL INDEX (location)
);

INSERT INTO places VALUES
    (1, "Lahore", ST_PointFromText("POINT(74.3587 31.5204)"));

-- distance in meters between two points
SELECT ST_Distance_Sphere(
    ST_PointFromText("POINT(74.3587 31.5204)"),    -- Lahore
    ST_PointFromText("POINT(67.0011 24.8607)")     -- Karachi
);
-- ~1027 km
```

For serious GIS, **PostGIS (PostgreSQL)** is the industry standard — but MySQL handles point-distance and bounding-box queries well enough for many apps.

---

## 6. Picking types — checklist for new tables

- [ ] IDs: `INT UNSIGNED AUTO_INCREMENT` (or `BIGINT` if you'll overflow ~2B rows)
- [ ] Money: `DECIMAL(p, s)` with explicit precision
- [ ] Names / emails: `VARCHAR(n)` with a sane `n`
- [ ] Status / category with fixed list: `ENUM`
- [ ] Big text: `TEXT`; mark for `FULLTEXT INDEX` if you'll search it
- [ ] Created/updated: `DATETIME DEFAULT CURRENT_TIMESTAMP [ON UPDATE ...]`
- [ ] Charset: `utf8mb4` everywhere; never plain `utf8`
- [ ] JSON only when truly schema-flexible — promote frequently-queried fields to columns

## Self-check

- [ ] What's the difference between `DECIMAL` and `FLOAT` and which for money?
- [ ] Why is `utf8mb4` preferred over `utf8`?
- [ ] When use `CHAR` vs `VARCHAR`?
- [ ] What's the Y2038 problem with `TIMESTAMP`?
- [ ] How do I extract a JSON field as a usable string?
- [ ] What's `AUTO_INCREMENT` and why is it useful?
- [ ] How would you store a user's "preferred languages" — array, JSON, separate table?
