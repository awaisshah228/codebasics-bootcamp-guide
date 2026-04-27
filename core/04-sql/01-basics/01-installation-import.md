# Section 1+2 — MySQL Setup & Movies Dataset Import

## Lectures covered
- How much SQL is Needed for a Data Scientist / AI Engineer?
- Install MySQL: Windows / Linux / Mac
- Import Movies Dataset in MySQL

---

## How much SQL do you need?

For data roles in 2025:
- **Data Analyst**: SQL is your daily driver — comfort + speed required.
- **Data Scientist**: comfortable with intermediate SQL (joins, aggregates, windows). Most companies pull data with SQL before any Python.
- **ML Engineer**: read-comfort with SQL; rarely write complex queries.
- **AI / GenAI Engineer**: SQL appears in RAG-over-databases and tool-use scenarios. Read-comfort + ability to verify what an LLM agent wrote.

The bootcamp targets ~80% mastery — enough for any analytical interview.

---

## MySQL installation

### Windows
1. Download "MySQL Installer" from https://dev.mysql.com/downloads/installer/
2. Choose "Developer Default" (installs server + Workbench + sample DBs)
3. Set a strong root password — *write it down*
4. Verify in PowerShell: `mysql --version`

### macOS
```bash
brew install mysql
brew services start mysql       # auto-start on boot
mysql_secure_installation        # set root password, drop test DB
mysql -u root -p
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
sudo mysql_secure_installation
sudo mysql -u root -p
```

---

## Clients — what to use

| Tool | When |
|---|---|
| **MySQL CLI** (`mysql -u root -p`) | Quick, scriptable; how production engineers connect |
| **MySQL Workbench** | GUI, free, official; great for ERDs + visual query results |
| **DBeaver** | GUI, free, multi-DB (works with Postgres, BigQuery too) |
| **VS Code SQLTools extension** | If you live in VS Code |

The bootcamp demos Workbench. Use whichever you like — the SQL is the same.

---

## Importing the Movies dataset (Codebasics standard)

### What's in it
A `moviesdb` schema with 6 tables: `movies`, `actors`, `movie_actor`, `studios`, `financials`, `languages`. Sandbox-style — small enough to query interactively, real enough to teach joins.

### Two import paths

#### Path A — Using a `.sql` dump (fastest)
1. Download `moviesdb.sql` from Codebasics' GitHub
2. In MySQL Workbench: Server → Data Import → "Import from Self-Contained File" → choose the `.sql` → Default Target Schema (or new) → Start Import
3. Or via CLI:
   ```bash
   mysql -u root -p < moviesdb.sql
   ```

#### Path B — Using individual CSVs
1. Create the schema: `CREATE DATABASE moviesdb;`
2. Run `CREATE TABLE` statements per CSV
3. Load each CSV: `LOAD DATA INFILE '/path/to/movies.csv' INTO TABLE movies FIELDS TERMINATED BY ',' IGNORE 1 LINES;`

> `LOAD DATA INFILE` requires a `--secure-file-priv` setting; if it errors, copy the CSV into the MySQL upload folder shown by `SHOW VARIABLES LIKE "secure_file_priv";`.

---

## Verifying the import

```sql
USE moviesdb;
SHOW TABLES;

-- typical output:
-- actors
-- financials
-- languages
-- movie_actor
-- movies
-- studios

SELECT COUNT(*) FROM movies;        -- ~50 rows
SELECT * FROM movies LIMIT 5;
```

If counts look right and `SELECT *` returns sensible rows, you're good.

---

## Quick orientation — what each table looks like

```sql
-- movies (the fact-ish table)
DESCRIBE movies;
-- movie_id, title, industry, release_year, imdb_rating, studio, language_id

-- actors
DESCRIBE actors;
-- actor_id, name, birth_year

-- movie_actor (junction table — many-to-many)
DESCRIBE movie_actor;
-- movie_id, actor_id

-- financials
DESCRIBE financials;
-- movie_id, budget, revenue, unit, currency

-- studios
DESCRIBE studios;
-- studio (PK), city, founded

-- languages
DESCRIBE languages;
-- language_id, name
```

This is the dataset every later lecture references. Worth a 5-minute browse.

---

## Workbench — the 4 tabs you'll use

1. **Schemas panel (left)** — your databases & tables
2. **Query editor (top)** — write SQL, hit Ctrl+Enter to run
3. **Results grid (bottom)** — query output
4. **Output / Action panel** — error messages, execution time

Tip: right-click a table → "Select Rows - Limit 1000" gives you a quick SELECT *.

---

## Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot `USE moviesdb;` | "Table doesn't exist" errors | always set the active DB first |
| Case-sensitivity surprises | macOS default is case-sensitive | use lowercase table names consistently |
| `LOAD DATA` permission denied | `secure_file_priv` setting | move CSV to the allowed dir, or use Workbench's import wizard |
| Different CSV separators | comma vs tab | adjust `FIELDS TERMINATED BY` |
| Forgot to `IGNORE 1 LINES` | header row imported as data | always specify |

## Self-check

- [ ] `mysql --version` works
- [ ] I can connect with my root password
- [ ] `moviesdb` schema is loaded; `SHOW TABLES;` shows 6 tables
- [ ] `SELECT * FROM movies LIMIT 5;` returns rows
- [ ] I picked a primary client (Workbench / DBeaver / CLI)
- [ ] My password is stored somewhere safe — and not committed to Git
