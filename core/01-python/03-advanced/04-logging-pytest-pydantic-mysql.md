# Section 14 — Logging, Pytest, Pydantic, Databases

## Lectures covered
- Logging · Automated Testing with Pytest · MySQL Setup (Win/Linux/Mac) · Working with MySQL in Python · Data Validation with Pydantic · Quiz · Exercise · Chapter Summary

---

## 1. Logging

### Why not `print`
- `print` goes only to stdout
- Can't filter by level (info vs error)
- Can't redirect to a file
- Can't include timestamp, module name, line number
- Hard to disable for production

### Stdlib `logging` — minimal setup
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),         # also to stdout
    ],
)

log = logging.getLogger(__name__)

log.debug("something fine-grained")
log.info("normal event")
log.warning("something off")
log.error("something failed")
log.critical("fire")

try:
    1/0
except Exception:
    log.exception("dividing failed")     # adds traceback automatically
```

### Levels (in order of severity)
DEBUG → INFO → WARNING → ERROR → CRITICAL.
Setting level to INFO means DEBUG is hidden but INFO+ shown.

### Per-module loggers
```python
# always do this at top of each module:
log = logging.getLogger(__name__)
```

### Don't use the root logger directly
Configure once at the entry point, then `getLogger(__name__)` everywhere else.

### Production tips
- Use **JSON logs** (`python-json-logger`) so log aggregators (Datadog, ELK) can parse
- Include request ID / trace ID for distributed systems
- Log at INFO normally; DEBUG temporarily when troubleshooting
- Never log secrets (tokens, passwords, PII)

---

## 2. Pytest — automated testing

### Why test
- Fearless refactoring — change code, run tests, know it still works
- Documents what the code is *supposed* to do
- Catches regressions before they reach production

### Basic shape
```python
# math_utils.py
def add(a, b): return a + b

# test_math_utils.py
from math_utils import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2
```

Run: `pytest -v`

Pytest auto-discovers files starting with `test_` and functions starting with `test_`.

### Useful assertion patterns
```python
def test_value_is_close():
    assert abs(0.1 + 0.2 - 0.3) < 1e-9

def test_in_list():
    assert "apple" in ["apple", "banana"]

import pytest

def test_raises():
    with pytest.raises(ValueError, match="bad"):
        raise ValueError("bad input")
```

### Fixtures — setup/teardown
```python
import pytest

@pytest.fixture
def sample_user():
    return {"name": "Awais", "age": 27}

def test_user_name(sample_user):
    assert sample_user["name"] == "Awais"
```

### Parametrize — same test, multiple inputs
```python
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### Useful flags
```bash
pytest -v              # verbose
pytest -x              # stop on first failure
pytest -k "user"       # only tests with "user" in name
pytest --cov=src       # coverage report (needs pytest-cov)
pytest tests/test_db.py::test_insert    # one specific test
```

### Mocking external things
```python
from unittest.mock import patch

@patch("module.requests.get")
def test_fetch(mock_get):
    mock_get.return_value.json.return_value = {"id": 1}
    result = fetch(...)
    assert result["id"] == 1
```

---

## 3. MySQL setup

### Install
- **Windows**: MySQL Installer from official site → MySQL Server + Workbench
- **Mac**: `brew install mysql && brew services start mysql`
- **Linux**: `sudo apt install mysql-server`

### Set root password (first run)
```bash
mysql_secure_installation
```

### Connect
```bash
mysql -u root -p          # CLI
```

Or use **MySQL Workbench** GUI / **DBeaver** for a visual client.

### Create a DB + user
```sql
CREATE DATABASE expenses;
CREATE USER 'app'@'localhost' IDENTIFIED BY 'secret';
GRANT ALL PRIVILEGES ON expenses.* TO 'app'@'localhost';
```

---

## 4. MySQL from Python

### Install driver
```bash
pip install mysql-connector-python
# or:  pip install pymysql
```

### Connect + query
```python
import mysql.connector

conn = mysql.connector.connect(
    host="localhost", user="app", password="secret", database="expenses"
)
cur = conn.cursor(dictionary=True)
cur.execute("SELECT * FROM expenses WHERE expense_date = %s", ("2025-01-01",))
rows = cur.fetchall()
cur.close()
conn.close()
```

### Always use parameterized queries (never f-strings)
```python
# BAD — SQL injection
cur.execute(f"SELECT * FROM users WHERE name = '{name}'")

# GOOD — driver escapes
cur.execute("SELECT * FROM users WHERE name = %s", (name,))
```

### Insert + commit
```python
cur.execute(
    "INSERT INTO expenses (expense_date, amount, category) VALUES (%s, %s, %s)",
    ("2025-01-01", 100.0, "Food"),
)
conn.commit()           # forgetting this = no save
```

### Context manager pattern
```python
from contextlib import contextmanager

@contextmanager
def get_cursor(commit=False):
    conn = mysql.connector.connect(...)
    cur = conn.cursor(dictionary=True)
    try:
        yield cur
        if commit: conn.commit()
    finally:
        cur.close(); conn.close()

with get_cursor(commit=True) as cur:
    cur.execute("INSERT ...", (...))
```

### SQLAlchemy — for richer apps
```bash
pip install sqlalchemy
```
```python
from sqlalchemy import create_engine, text

engine = create_engine("mysql+mysqlconnector://app:secret@localhost/expenses")

with engine.connect() as conn:
    rows = conn.execute(text("SELECT * FROM expenses WHERE amount > :a"), {"a": 100}).all()
```

For Project 2, raw `mysql.connector` is fine; SQLAlchemy is shown later in ML deployments.

---

## 5. Pydantic — data validation

### Why
- Catches bad inputs early
- Self-documents expected shape
- Plays seamlessly with FastAPI

### Basic model
```python
from pydantic import BaseModel, Field, EmailStr
from datetime import date

class User(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)        # constraints
    email: EmailStr                       # validates format
    signup_date: date

u = User(name="Awais", age=27, email="a@b.com", signup_date="2025-01-01")
# strings auto-coerce: "2025-01-01" → date(2025,1,1)
# Bad data: ValidationError with all the issues at once
```

### Nested models
```python
class Address(BaseModel):
    city: str
    country: str

class User(BaseModel):
    name: str
    address: Address
```

### Custom validators
```python
from pydantic import field_validator

class User(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def must_be_adult(cls, v):
        if v < 18:
            raise ValueError("must be 18+")
        return v
```

### Settings management
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_url: str
    api_key: str

    class Config:
        env_file = ".env"

settings = Settings()
```

Reads from environment vars + `.env` file. Replace ad-hoc `os.getenv` calls.

---

## 6. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| Logs go to stdout but not file | `basicConfig` already called | configure logging once at entry point |
| `print` left in production | quick debug | replace with `log.debug` |
| Forgot `conn.commit()` | INSERT but no row appears | always commit (or use context manager) |
| SQL injection vulnerability | f-string SQL | parameterize with `%s` |
| Pydantic accepting bad data | wrong type annotation | add `Field(...)` constraints / validators |
| Test passes locally, fails CI | test depends on local DB state | use fixtures + tmp data |
| Mock placed wrong | `@patch("requests.get")` instead of `@patch("module.requests.get")` | patch where it's *used*, not where it's *defined* |

## Self-check

- [ ] What's the difference between `log.error` and `log.exception`?
- [ ] Why should I never use f-strings to build SQL?
- [ ] What does `pytest.fixture` do?
- [ ] How does `@pytest.mark.parametrize` reduce test boilerplate?
- [ ] Show me a Pydantic model with a custom `age >= 18` validator.
- [ ] How do I get a coverage report?
- [ ] What's the safe way to store DB credentials for an app?
- [ ] Write a function that inserts a row and commits, with proper try/finally.
