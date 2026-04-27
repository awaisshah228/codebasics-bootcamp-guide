# Module 1 — Python: Beginner to Advanced for Data Professionals

> **Duration**: 17h 39m 41s · 114 lectures · 2 projects · 1 final quiz · interview question bank
> **Status**: ✅ Complete (100%)

## Why this module exists

Python is the *connective tissue* for the entire bootcamp. Every later module — pandas analytics, ML in scikit-learn, deep learning in PyTorch, LLM apps in LangChain — runs on Python. If Python is shaky, every later module is shaky.

Codebasics' Python module goes from `print("hello")` to building a full FastAPI + Streamlit + MySQL expense tracker. That arc is unusual for a "Python for data" course — most stop at pandas.

## Folder layout

```
01-python/
├── README.md                                ← you are here
├── 00-welcome-and-projects.md               ← Sections 1 & 2 (Welcome + Project descriptions)
├── 01-basics/                               ← Sections 3–9 (foundations through pandas EDA)
│   ├── README.md
│   ├── 01-installation.md
│   ├── 02-variables-numbers-strings.md
│   ├── 03-lists-conditions-loops.md
│   ├── 04-functions-dict-tuples-files.md
│   ├── 05-classes-exceptions.md
│   ├── 06-numpy.md
│   └── 07-eda-pandas-matplotlib-seaborn.md
├── 02-projects/                             ← Sections 10 & 15 (the two big builds)
│   ├── README.md
│   ├── 01-hospitality-eda.md
│   └── 02-expense-tracker.md
└── 03-advanced/                             ← Sections 11–14 (advanced Python + APIs + DBs)
    ├── README.md
    ├── 01-comprehensions-sets.md
    ├── 02-json-generators-decorators.md
    ├── 03-apis-fastapi.md
    └── 04-logging-pytest-pydantic-mysql.md
```

## Curriculum (verbatim)

### Section 1 — Welcome to the Python Experience (1 lecture)
- Course Content Overview

### Section 2 — Project Description (2 lectures)
- Project 1: Hospitality Domain Data Analysis
- Project 2: Expense Tracking System

### Section 3 — Python Basics: Getting Started (5 lectures)
- Peter Pandey's Journey and Need to Learn Python
- Python Installation — Windows / Linux / Mac
- How to Download Code and Get Help

### Section 4 — Variables, Numbers and Strings (8 lectures, 1 quiz, 1 exercise)
- Variables · Numbers · Strings · Strings exercise · Quiz · "Peter shares his coding fear with Tony" · Exercise · Chapter Summary

### Section 5 — Lists, If Condition and For Loop (8 lectures, 1 quiz, 1 exercise)
- Lists · Install PyCharm · If Condition · For Loop · Quiz · Practice · Exercise · Chapter Summary

### Section 6 — Functions, Dictionaries, Tuples and File Handling (9 lectures)
- Functions · Dictionary and Tuples · Modules and Pip · File Handling · Quiz · Peter's Request to Tony · Exercise · "Two Deadly Viruses Infecting Learners" · Chapter Summary

### Section 7 — Classes and Exception Handling (9 lectures)
- Classes and Objects · Operator Overloading · Inheritance · Exception Handling · `__main__` Function · Quiz · Real-time problem · Exercise · Chapter Summary

### Section 8 — NumPy (8 lectures)
- Introduction & Benefits · Basic Operations · Matrix Operations · Slicing & Stacking · Quiz · Real-time problem · Exercise · Chapter Summary

### Section 9 — EDA Using Pandas, Matplotlib, Seaborn (9 lectures)
- Pandas Intro & Installation · DataFrame Basics · Read/Write Excel & CSV · Handle NA values (1 & 2) · Group By · Concat & Merge · Visualization · Quiz

### Section 10 — Project 1: Exploratory Data Analytics in Hospitality Domain (7 lectures)
- Problem Statement, OLTP vs OLAP, ETL, Data Warehouse · CSV Data Understanding · Fact vs Dim, Star vs Snowflake · Exploration · Cleaning · Transformation · Insights

### Section 11 — Comprehensions and Sets (9 lectures)
- Set & Frozenset · List/Dict/Set Comprehensions · Quiz · Adhoc tasks · Exercise · PEP8 Naming · Code Debugging in PyCharm · Chapter Summary

### Section 12 — JSON, Generators and Decorators (6 lectures)
- Working with JSON · Generators & Iterators · Decorators · Quiz · Exercise · Chapter Summary

### Section 13 — APIs (7 lectures)
- What is API? · Calling APIs with `requests` · Building APIs with FastAPI · Quiz · Real-time API fetch · Exercise · Chapter Summary

### Section 14 — Logging, Pytest, Pydantic and Databases (9 lectures)
- Logging · Automated Testing with Pytest · MySQL Setup · Working with MySQL in Python · Data Validation with Pydantic · Quiz · Exercise · Chapter Summary

### Section 15 — Project 2: Expense Tracking System (11 lectures)
- Problem Statement & Tech Architecture · Database CRUD · Automated Tests · FastAPI Backend (Expense Mgmt) · Logging · Streamlit Intro · Streamlit Frontend · Analytics Backend · Analytics Frontend · README & requirements.txt · Exercise

### Section 16 — Final Quiz (1 lecture)
- Final Quiz

### Section 17 — Python Interview Question Bank (1 lecture)
- Python Interview Question Bank

### Section 18 — What's Next (1 lecture)
- Transition to Online Credibility

---

## Module-level goal

After this module I should be able to:

1. Read and write idiomatic Python (comprehensions, generators, decorators, context managers)
2. Manipulate any tabular dataset in pandas without Googling syntax constantly
3. Build a CLI/data-pipeline script that reads CSV, cleans, transforms, persists to MySQL, and logs properly
4. Build and call REST APIs with FastAPI
5. Write tests with pytest and validate inputs with Pydantic
6. Stand up a Streamlit UI for quick frontends

## Two flagship projects

| Project | Skills demonstrated | Domains | Folder |
|---|---|---|---|
| **Hospitality EDA** | pandas, matplotlib, seaborn, schema design | Hospitality / Hotels | `02-projects/01-hospitality-eda.md` |
| **Expense Tracker** | FastAPI, Streamlit, MySQL, pytest, Pydantic, logging | Personal finance / fullstack | `02-projects/02-expense-tracker.md` |

## Module-level self-check

- [ ] Can I explain mutable vs immutable in Python with concrete examples?
- [ ] Can I write a list comprehension that filters and transforms in one line?
- [ ] Can I write a decorator that logs function execution time?
- [ ] Can I read a 1M-row CSV in pandas without crashing?
- [ ] Can I expose `GET /expenses` and `POST /expenses` with FastAPI?
- [ ] Can I write a unit test with pytest fixtures?
- [ ] Can I validate an incoming JSON payload with Pydantic?
- [ ] Can I deploy a Streamlit dashboard locally and explain what each component does?
