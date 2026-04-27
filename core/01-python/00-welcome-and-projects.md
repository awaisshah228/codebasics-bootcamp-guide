# Section 1+2 — Welcome & Project Description

## Lectures covered
- **Section 1**: Course Content Overview
- **Section 2**: Project 1 (Hospitality Domain Data Analysis), Project 2 (Expense Tracking System)

---

## What this section actually does

It frames the module as **project-driven**, not topic-driven. Two real builds are previewed *before* you write a single line of Python — so when you later learn `pandas.groupby`, you already know which project you'll use it in.

This is the right way to learn programming: **build first, theorize second**.

---

## Project 1 — Hospitality Domain Data Analysis

### Domain
The hotel industry — multiple properties, room categories, bookings, occupancy, revenue. The dataset reflects this with **fact tables** (bookings, ratings) and **dimension tables** (hotels, rooms, dates).

### Business questions you'll answer
- What's the average occupancy rate per city / per property type?
- Which property categories generated highest revenue last quarter?
- How does weekday vs weekend revenue compare?
- Which booking channels (direct, OTA, others) underperform on rating?
- What's the cancellation rate by property and what does it cost?

### Skills you'll practice end-to-end
- pandas: `read_csv`, `merge`, `groupby`, `pivot_table`, `apply`
- matplotlib + seaborn for plots
- Star-schema concepts (Fact + Dim)
- ETL flow: extract → transform → load (here: clean + reshape)
- OLTP vs OLAP distinction
- Insight generation (i.e., translating numbers into a sentence a stakeholder cares about)

### What "done" looks like
A Jupyter notebook that:
1. Loads 5+ CSVs
2. Joins them into a single analytical frame
3. Cleans (NA handling, type coercion, duplicate removal)
4. Produces 8–12 visualizations
5. Ends with a "Top 5 insights for the GM" markdown summary

> **The output is not the code — the output is the markdown summary.** Stakeholders never read your notebook. They read your conclusions.

---

## Project 2 — Expense Tracking System

### Why this is unusual for a "Python for data" course
Most data-Python courses end at pandas. Codebasics intentionally takes you further: a **fullstack app**. You'll build:

- **Backend**: FastAPI service with CRUD endpoints
- **Frontend**: Streamlit UI
- **Storage**: MySQL via SQLAlchemy / mysql-connector
- **Validation**: Pydantic models
- **Tests**: pytest unit + integration
- **Logging**: structured logs with the `logging` module
- **Packaging**: README + `requirements.txt`

### Functional features
- Log a new expense (date, category, amount, notes)
- Update / delete an expense
- View expenses for a date range
- Category-wise + month-wise analytics with charts
- A clean Streamlit dashboard

### Architecture preview

```
┌──────────────────┐     HTTP     ┌──────────────────┐     SQL      ┌─────────────┐
│   Streamlit UI   │ ───────────> │   FastAPI        │ ───────────> │   MySQL     │
│  (frontend)      │              │   (backend)      │              │  (storage)  │
└──────────────────┘ <─────────── └──────────────────┘ <─────────── └─────────────┘
                       JSON                              rows
```

Layers and where they live in the bootcamp:

| Layer | Section that teaches it |
|---|---|
| Pydantic models for request/response | Section 14 |
| FastAPI endpoints | Section 13 |
| SQL CRUD + connection | Section 14 |
| Pytest tests | Section 14 |
| Streamlit UI | Section 15 |
| Logging | Section 14 |
| README / requirements.txt | Section 15 |

### What "done" looks like
A Git repo I can show in interviews containing:
- `backend/` — FastAPI code
- `frontend/` — Streamlit code
- `tests/` — pytest suite
- `requirements.txt`
- `README.md` with screenshots and run instructions
- A short demo video (optional, big plus)

---

## Why these two projects in particular

The pair is chosen deliberately:

1. **Project 1** = **read-only analytics** with messy data. The "Data Analyst" skin.
2. **Project 2** = **read-write transactional system** with a UI. The "Software Engineer who knows data" skin.

Together they prove you can do both — which is exactly what entry-level data jobs require.

## My notes from this section

- Initial reaction:
- Which project excites me more, and why:
- Concerns / unknowns:

## Self-check

- [ ] Can I summarize each project in 30 seconds?
- [ ] Do I understand the *business* value of each, not just the tech?
- [ ] Do I know which sections of this module map to which project?
