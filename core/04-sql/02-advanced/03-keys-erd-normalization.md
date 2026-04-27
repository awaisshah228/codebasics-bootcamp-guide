# Advanced 3 — Keys, ERDs, Normalization, Data Integrity

## Lectures covered
- Primary Key · Foreign Key
- Entity Relationship Diagram (ERD)
- Database Normalization · Data Integrity

---

## 1. Primary Keys (PK)

A **primary key** uniquely identifies each row.

```sql
CREATE TABLE customers (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL
);
```

### Properties
- **Unique** across the table
- **Not NULL**
- Implicitly **indexed** (gives O(log n) lookup)
- Should be **stable** — don't change after creation

### Surrogate vs natural keys
- **Surrogate**: an artificial column (`id INT AUTO_INCREMENT`) — *recommended default*
- **Natural**: a real-world identifier (`email`, `ssn`) — risky if it can change

### Composite PK (multi-column)
```sql
CREATE TABLE order_items (
    order_id INT,
    line_no INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, line_no)
);
```

Each row identified by the *combination*. Common in junction tables.

---

## 2. Foreign Keys (FK)

A **foreign key** in one table references the PK of another, enforcing **referential integrity**.

```sql
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
```

### What it enforces
- You can't insert an order with `customer_id = 999` if no customer 999 exists
- You can't delete a customer who still has orders (unless you allow cascade)

### Cascade options
```sql
FOREIGN KEY (customer_id) REFERENCES customers(id)
    ON DELETE CASCADE                  -- delete a customer → delete their orders
    ON UPDATE CASCADE                  -- change customer.id → propagate to orders.customer_id
```

Other options: `ON DELETE SET NULL`, `ON DELETE RESTRICT` (default — block).

### Why some teams skip FKs
- Performance: enforces extra checks on insert/delete
- Flexibility: easier ETL into the table
- Many-write systems sometimes enforce in app code

For OLTP and analytical sandboxes both: **start with FKs on**, turn off only if proven necessary.

---

## 3. Entity Relationship Diagrams (ERD)

A diagram showing tables (entities) and the relationships between them.

### Notation (Crow's foot)
```
customers ──╾─── orders
            ↑
        one customer has many orders (1:N)
```

### Cardinality
| Notation | Meaning |
|---|---|
| `1:1` | one-to-one (e.g. `users` ↔ `user_profiles`) |
| `1:N` | one-to-many (e.g. `customers` → `orders`) |
| `N:M` | many-to-many (e.g. `students` ↔ `courses`) — needs junction table |

### Drawing tools
- **MySQL Workbench** → File → New Model → reverse-engineer from existing schema → produces an ERD automatically
- **dbdiagram.io** — text-based, fast
- **Lucidchart, draw.io, Whimsical** — generic diagram tools

### Example for the Movies dataset

```
movies (movie_id PK)
   1 ─── 1   financials (movie_id PK, FK)
   1 ─── N   movie_actor (movie_id FK, actor_id FK) ── N ─── 1   actors (actor_id PK)
   N ─── 1   languages (language_id PK)
   N ─── 1   studios (studio PK)
```

Drawing this out before writing complex queries reveals the join paths.

---

## 4. Normalization — when to split tables

Normalization = organizing data to **eliminate redundancy** and **prevent anomalies**.

### The anomalies we're avoiding
- **Update anomaly** — changing a customer's name in 100 orders
- **Insertion anomaly** — can't add a product without an order
- **Deletion anomaly** — deleting the last order of a customer wipes the customer's data

### 1NF — atomic values, no repeating groups
**Bad** (a phone column with "555-1, 555-2"):
```
| id | name  | phones        |
| 1  | Alice | 555-1, 555-2  |
```

**1NF** — split into separate rows or a separate phone table:
```
phones: (customer_id, phone)
| 1 | 555-1 |
| 1 | 555-2 |
```

### 2NF — no partial dependency on part of a composite PK
If your PK is `(order_id, product_id)` and `product_name` depends only on `product_id`, that violates 2NF — `product_name` belongs in a `products` table.

### 3NF — no transitive dependency
If `zipcode` determines `city`, then in a `customers (id, zipcode, city)` table, `city` transitively depends on `id` *through* `zipcode` — split `zipcodes (zipcode, city)`.

### BCNF / 4NF / 5NF
Stricter forms. Rarely needed in practice.

### When to denormalize (intentionally)
For analytical / OLAP workloads, you **denormalize** for speed:
- Pre-join into a wide fact table
- Embed dimension attributes in the fact table
- Trade write-time cost for read-time speed

This is the **star schema** (covered in `05-data-warehouse-etl.md`).

---

## 5. Data integrity — the layers

### 1. Data type integrity
`age INT` rejects `"hello"`. `email VARCHAR(255) NOT NULL` rejects nulls.

### 2. Domain integrity (CHECK constraints)
```sql
CREATE TABLE orders (
    id INT PRIMARY KEY,
    quantity INT,
    CHECK (quantity > 0)
);
```

(MySQL fully honors `CHECK` since 8.0.16.)

### 3. Entity integrity
PK ensures each row is uniquely identifiable. `NOT NULL` + `UNIQUE`.

### 4. Referential integrity
FKs ensure relationships stay valid.

### 5. Application / business rule integrity
Things like "an order can't be cancelled if shipped" → enforced in app code or triggers.

---

## 6. A working example — designing a schema

### Requirement
Track customers, orders, order items, products.

### Tables
```sql
CREATE TABLE customers (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0)
);

CREATE TABLE orders (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    placed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status ENUM("pending", "shipped", "delivered", "cancelled") DEFAULT "pending",
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE order_items (
    order_id INT UNSIGNED NOT NULL,
    line_no INT NOT NULL,
    product_id INT UNSIGNED NOT NULL,
    quantity INT NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL,                  -- snapshot at order time
    PRIMARY KEY (order_id, line_no),
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

This schema is in **3NF**, with full referential and domain integrity.

> Note `unit_price` snapshotted in `order_items` — even if `products.price` changes later, the historical order is preserved. Common pattern.

---

## 7. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| No PK on a table | hard to update specific rows; replication breaks | always have a PK |
| Using natural key for PK | breaks if value changes | surrogate `id` PK |
| FK without index | slow joins / cascade | MySQL auto-creates index for FKs (good) |
| Storing comma-separated values | violates 1NF; can't query | split into separate rows / table |
| Forgetting `NOT NULL` | accidental nulls leak in | require fields explicitly |
| Mixing units in one column | "100" — pounds? dollars? | one unit per column |

## Self-check

- [ ] Difference between PK and FK?
- [ ] What's a junction table and when do I need one?
- [ ] Walk through 1NF → 2NF → 3NF with an example.
- [ ] When is denormalization the right call?
- [ ] What does `ON DELETE CASCADE` do?
- [ ] Why would I snapshot `unit_price` in `order_items`?
- [ ] Draw an ERD for: `users`, `posts`, `comments`, `tags`, `post_tags`.
