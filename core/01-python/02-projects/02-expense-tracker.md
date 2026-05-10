# Project 2 — Expense Tracking System

## Lectures covered (Section 15)
- Problem Statement & Tech Architecture
- Database CRUD Operations
- Automated Tests Setup for CRUD
- Expense Management: Backend (FastAPI)
- Expense Management: Logging
- Streamlit Introduction
- Expense Management: Frontend (Streamlit)
- Analytics: Backend (FastAPI)
- Analytics: Frontend (Streamlit)
- README and requirements.txt
- Exercise

---

## In one sentence
You build a small **fullstack app** to log and analyze daily expenses — Streamlit on top, FastAPI in the middle, MySQL underneath — proving you can ship the whole loop, not just a notebook.

## Real-world analogy
Think of a personal expense diary. The pages where you write entries are the **frontend** (Streamlit). The clerk who validates and files your entries is the **backend** (FastAPI). The locked filing cabinet that keeps every entry safe is the **database** (MySQL). Each layer has one job; together they make a real product you could deploy and use yourself.

## The intuition (plain English)
Three layers, one HTTP call between each. The user clicks something in **Streamlit**, which sends an HTTP request to **FastAPI**, which validates with **Pydantic**, talks to **MySQL** via SQL, and returns JSON. You add **logging** so you can see what happened in production, **pytest** so you can refactor without fear, and a **README** so a recruiter can run it. This is the standard shape of every web product — once you have built it once, every future API+UI tool feels familiar.

## Mini worked example
The end-to-end flow for adding one expense:

```python
# 1. User edits a row in Streamlit and clicks Save
# frontend/app.py
import requests
requests.post(
    "http://localhost:8000/expenses/2026-05-10",
    json=[{"amount": 250.0, "category": "Food", "notes": "lunch"}],
)

# 2. FastAPI validates with Pydantic + writes to MySQL
# backend/server.py
@app.post("/expenses/{expense_date}")
def add(expense_date: date, items: list[dict]):
    for it in items:
        db_helper.insert_expense(expense_date, it["amount"], it["category"], it.get("notes"))
    logger.info(f"saved {len(items)} expenses for {expense_date}")
    return {"status": "ok"}

# 3. db_helper runs the SQL
cur.execute(
    "INSERT INTO expenses (expense_date, amount, category, notes) VALUES (%s,%s,%s,%s)",
    (expense_date, amount, category, notes),
)
conn.commit()                     # without this, no row appears
```

Three files, three layers, one round-trip. That is a fullstack app in 20 lines.

## At-a-glance

```
┌──────────────┐     HTTP POST       ┌──────────────┐    SQL INSERT    ┌──────────────┐
│  Streamlit   │  ───────────────►   │   FastAPI    │  ──────────────► │    MySQL     │
│  (UI page)   │                     │   (server)   │                  │  (storage)   │
│              │  ◄──── JSON ────    │              │  ◄── rows ────   │              │
└──────────────┘                     └──────────────┘                  └──────────────┘
                                            │
                                            └──── server.log (logging)
                                            └──── tests/ (pytest)
```

## Why this matters
- Most data-Python courses stop at notebooks. Shipping an app proves you can produce something a non-technical user can click.
- The Streamlit + FastAPI + MySQL stack is the modern shape of internal data tools at most companies.
- This is your second portfolio repo and your second LinkedIn post — recruiter gold.

---

## 1. Problem statement

Build a small fullstack app to track personal expenses:
- Log: date, category, amount, note
- View: by date range
- Analyze: monthly totals, category breakdown
- Edit / delete

Demo of the **end-to-end engineering loop**: DB → backend → frontend, with tests + logging + a real README.

---

## 2. Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐    SQL    ┌─────────────┐
│  Streamlit UI   │  ─────────> │  FastAPI server │  ──────>  │   MySQL     │
│  (frontend)     │             │  (backend)      │           │             │
│                 │  <───JSON── │                 │  <─rows── │             │
└─────────────────┘             └─────────────────┘           └─────────────┘
        ↑                              ↑
   user interacts                 logs to file
```

### Why each layer

| Layer | Why this tool |
|---|---|
| **Streamlit** | Fastest way to build a Python data UI; no JS needed |
| **FastAPI** | Modern, fast, auto-docs, Pydantic-native |
| **MySQL** | Industry-standard relational DB; covered in Section 14 |
| **pytest** | Standard Python test framework |
| **Pydantic** | Request/response validation built into FastAPI |

---

## 3. Repo structure

```
expense-tracker/
├── backend/
│   ├── server.py              # FastAPI app
│   ├── db_helper.py           # DB CRUD (MySQL)
│   ├── logging_setup.py       # logging config
│   └── models.py              # Pydantic models
├── frontend/
│   └── app.py                 # Streamlit UI
├── tests/
│   ├── test_db_helper.py
│   └── test_server.py
├── requirements.txt
├── README.md
└── server.log                 # generated at runtime
```

---

## 4. Backend (FastAPI)

### Pydantic models
```python
# backend/models.py
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

class ExpenseIn(BaseModel):
    amount: float = Field(gt=0)
    category: str
    notes: Optional[str] = None

class Expense(ExpenseIn):
    id: int
    expense_date: date
```

### Database helper (CRUD)
```python
# backend/db_helper.py
import mysql.connector
from contextlib import contextmanager

@contextmanager
def get_db_cursor(commit=False):
    conn = mysql.connector.connect(host="localhost", user="root",
                                   password="...", database="expenses")
    cur = conn.cursor(dictionary=True)
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()

def fetch_expenses_for_date(target_date):
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM expenses WHERE expense_date = %s", (target_date,))
        return cur.fetchall()

def insert_expense(expense_date, amount, category, notes):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO expenses (expense_date, amount, category, notes) VALUES (%s,%s,%s,%s)",
            (expense_date, amount, category, notes),
        )

def delete_expenses_for_date(target_date):
    with get_db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM expenses WHERE expense_date = %s", (target_date,))

def fetch_expense_summary(start_date, end_date):
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT category, SUM(amount) AS total
            FROM expenses
            WHERE expense_date BETWEEN %s AND %s
            GROUP BY category
        """, (start_date, end_date))
        return cur.fetchall()
```

### FastAPI endpoints
```python
# backend/server.py
from fastapi import FastAPI, HTTPException
from datetime import date
import db_helper, logging_setup

app = FastAPI()
logger = logging_setup.get_logger()

@app.get("/expenses/{expense_date}", response_model=list[dict])
def get_expenses(expense_date: date):
    rows = db_helper.fetch_expenses_for_date(expense_date)
    if rows is None:
        raise HTTPException(500, "DB error")
    return rows

@app.post("/expenses/{expense_date}")
def add_or_update(expense_date: date, items: list[dict]):
    db_helper.delete_expenses_for_date(expense_date)
    for it in items:
        db_helper.insert_expense(expense_date, it["amount"], it["category"], it.get("notes"))
    logger.info(f"updated {len(items)} expenses for {expense_date}")
    return {"status": "ok"}

@app.get("/analytics")
def analytics(start_date: date, end_date: date):
    return db_helper.fetch_expense_summary(start_date, end_date)
```

Run: `uvicorn backend.server:app --reload`. Auto-docs at `localhost:8000/docs`.

---

## 5. Frontend (Streamlit)

```python
# frontend/app.py
import streamlit as st
import requests
from datetime import datetime

API = "http://localhost:8000"

st.title("Expense Tracker")
tab_mgmt, tab_analytics = st.tabs(["Add / Update", "Analytics"])

with tab_mgmt:
    target = st.date_input("Date")
    if st.button("Load existing"):
        rows = requests.get(f"{API}/expenses/{target}").json()
        st.session_state["rows"] = rows or [{"amount":0, "category":"Food", "notes":""} for _ in range(5)]

    edited = st.data_editor(
        st.session_state.get("rows", [{"amount":0, "category":"Food", "notes":""} for _ in range(5)]),
        num_rows="dynamic",
    )
    if st.button("Save"):
        requests.post(f"{API}/expenses/{target}", json=edited)
        st.success("Saved")

with tab_analytics:
    c1, c2 = st.columns(2)
    start = c1.date_input("Start")
    end = c2.date_input("End")
    if st.button("Run analytics"):
        data = requests.get(f"{API}/analytics", params={"start_date": start, "end_date": end}).json()
        st.bar_chart({d["category"]: d["total"] for d in data})
```

Run: `streamlit run frontend/app.py`.

---

## 6. Logging

```python
# backend/logging_setup.py
import logging

def get_logger(name="server"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.FileHandler("server.log")
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger
```

Use throughout backend: `logger.info(...)`, `logger.warning(...)`, `logger.exception(...)`.

---

## 7. Testing

```python
# tests/test_db_helper.py
import pytest
import db_helper

def test_fetch_returns_list():
    rows = db_helper.fetch_expenses_for_date("2025-01-01")
    assert isinstance(rows, list)

def test_insert_then_fetch(tmp_date="2099-12-31"):
    db_helper.delete_expenses_for_date(tmp_date)
    db_helper.insert_expense(tmp_date, 100.0, "Food", "test")
    rows = db_helper.fetch_expenses_for_date(tmp_date)
    assert len(rows) == 1
    assert rows[0]["amount"] == 100.0
    db_helper.delete_expenses_for_date(tmp_date)   # cleanup
```

Run: `pytest -v tests/`

---

## 8. requirements.txt
```
fastapi
uvicorn
mysql-connector-python
streamlit
requests
pydantic
pytest
```

---

## 9. README — what to put in it

- Problem statement
- Architecture diagram (or ASCII)
- Tech stack with one-line "why"
- How to run (DB setup, backend, frontend)
- Screenshots (UI + Swagger docs)
- Test instructions
- Lessons learned (1 short paragraph — interviewers love this)

---

## 10. Stretch ideas (do these *after* the basic version works)
- Auth (FastAPI users + JWT)
- SQLAlchemy ORM instead of raw SQL
- Dockerize (Dockerfile + docker-compose for MySQL)
- Deploy backend on Render / Railway, frontend on Streamlit Cloud
- Add a forecasting widget (simple Prophet model on monthly totals)

These extensions become great LinkedIn posts.

## Self-check

- [ ] Can I explain why we have 3 layers (UI / API / DB)?
- [ ] Did I write at least 5 pytest tests?
- [ ] Does the FastAPI swagger UI render at `/docs`?
- [ ] Are my logs going to a file, not just stdout?
- [ ] Does my README have a screenshot?
- [ ] Have I deployed it somewhere (or written instructions)?
- [ ] Have I posted a LinkedIn writeup with a Loom/screen-recording demo?

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Fullstack** | Code spanning UI + server + storage |
| **Frontend** | The UI the user sees |
| **Backend** | The server that handles business logic + data access |
| **Database** | The persistent store — survives restarts |
| **HTTP** | The protocol browsers and servers speak |
| **REST API** | A backend that exposes resources via URLs + verbs (GET/POST/PUT/DELETE) |
| **JSON** | The data format sent between frontend and backend |
| **CRUD** | Create, Read, Update, Delete — the four basic data operations |
| **Streamlit** | A Python library for building web UIs from scripts |
| **FastAPI** | A Python web framework with auto-generated docs and Pydantic validation |
| **uvicorn** | The server program that runs FastAPI apps |
| **Swagger UI / `/docs`** | Auto-generated interactive API documentation |
| **MySQL** | A widely used open-source relational database |
| **`mysql-connector-python`** | The MySQL driver Python uses to talk to the DB |
| **SQLAlchemy** | A higher-level Python toolkit for SQL — ORM and query builder |
| **ORM** (Object-Relational Mapper) | Lets you treat DB rows as Python objects |
| **Pydantic** | Library that validates incoming data against a model |
| **Pydantic model** | A class describing required fields and types |
| **`BaseModel`** | The Pydantic base class your models inherit from |
| **HTTP status code** | Numeric result of a request — 200 OK, 404 not found, 500 server error |
| **`HTTPException`** | FastAPI's way to return an error with a status code |
| **Cursor** | Object you use to send SQL and read rows back |
| **`commit`** | Tells the DB to actually save changes |
| **Parameterized query** | SQL with `%s` placeholders — prevents SQL injection |
| **SQL injection** | Attack where user input becomes SQL — defeated by parameterized queries |
| **Context manager** | `with` block that guarantees cleanup (close cursor, close connection) |
| **Logging** | Writing structured runtime messages to a file or stream |
| **`logger.info/warning/error`** | Standard log levels |
| **pytest** | The testing framework that auto-discovers `test_*` files |
| **Fixture** | A pytest function providing setup data to tests |
| **Mocking** | Replacing real services (DB, HTTP) with stand-ins during tests |
| **`requirements.txt`** | The package list a fresh clone needs to install |
| **Deployment** | Hosting your app on a public URL — Render, Railway, Streamlit Cloud |

## Further reading
- Foundations: [../03-advanced/03-apis-fastapi.md](../03-advanced/03-apis-fastapi.md)
- Storage + tests + Pydantic: [../03-advanced/04-logging-pytest-pydantic-mysql.md](../03-advanced/04-logging-pytest-pydantic-mysql.md)
- Project 1 (the analytics counterpart): [01-hospitality-eda.md](01-hospitality-eda.md)
- ML model deployment uses the same stack: [../../06-machine-learning/05-lifecycle-mlops/README.md](../../06-machine-learning/05-lifecycle-mlops/README.md)
