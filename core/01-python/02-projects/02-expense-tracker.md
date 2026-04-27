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
