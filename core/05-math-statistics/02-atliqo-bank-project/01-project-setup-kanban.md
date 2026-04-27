# Section 6 — AtliQo Bank: Kickoff & Project Management

## Lectures covered
- Chapter Overview
- Bashneer Grower's Bank Venture
- AtliQo Credit Card Launch: Problem Statement
- Project Kick-off Meeting
- Project Management Using Kanban
- Tasks Breakdown and JIRA Board Setup
- Meeting: Tony, Peter, Natasha

---

## 1. The story (Codebasics' cinematic setup)

**Bashneer Grower** runs **AtliQo Bank**. The business wants to launch a new **credit card** in a competitive market. Rather than a generic launch, leadership asks the data team to identify the right **target market segment** — customers most likely to convert, profitably.

The data team:
- **Tony** — senior analyst (mentor figure)
- **Peter** — junior analyst (the learner stand-in — that's *you*)
- **Natasha** — project manager / business stakeholder

The narrative is fictional but the **shape is real**. Every working analytics team has these dynamics: ambiguous brief, time pressure, stakeholder questions you didn't expect, recovering from "you're answering the wrong question."

---

## 2. The business problem statement

> **Goal**: Launch AtliQo's new credit card. Identify a **target customer segment** with the best balance of:
> - Spend-volume potential (will they use the card?)
> - Repayment reliability (creditworthiness)
> - Demographic fit (existing relationship / branch reach)

> **Deliverable (Phase 1)**: A data-backed recommendation of one or two target segments, with supporting analysis.

---

## 3. Why this is the perfect Math & Stats project

It forces you to apply *every* stats topic from this module:

| Topic | Where it's used |
|---|---|
| Visualization basics | initial EDA — distributions of age, income, spending |
| Mean / median / mode | summarizing customer demographics |
| Percentile / IQR | outlier detection on income |
| Correlation | does income drive credit-score? |
| Probability | "what fraction of segment X is high-credit?" |
| Distributions / Normal | testing if income is roughly Normal |
| Z-score | standardizing for outlier detection |
| Hypothesis testing (later in the module) | "do these segments truly differ?" |
| A/B test design (later) | the launch strategy itself |

You can see why this project sits *inside* Math & Stats — it's the integration test.

---

## 4. The dataset (typical Codebasics shape)

You'll receive 4 CSV files (or MySQL tables). Approximate:

| Table | Approx rows | Key columns |
|---|---|---|
| `customers` | 50,000 | customer_id, age, gender, location, occupation, annual_income |
| `credit_profiles` | 50,000 | customer_id, credit_score, credit_utilization, outstanding_debt, credit_inquiries_last_year |
| `transactions` | 200,000+ | tran_id, customer_id, tran_date, tran_amount, platform, product_category |
| `dim_date` | ~1,500 | date, weekday, month, quarter |

The volume is real-world-ish — large enough that pandas needs care, small enough that it fits in memory.

---

## 5. Project management — Kanban + JIRA

### Why bother with PM as an analyst
- Stakeholders ask "where are you?" mid-week. A board answers without you typing.
- Splitting work into cards forces clarity.
- Done-column visibility makes your contribution legible to reviewers.

### Kanban basics (recap from SQL module)
Three columns: **To Do · In Progress · Done**. Each task is a card. Cards flow left → right.

### Tasks breakdown for AtliQo Phase 1 (typical)
1. Get raw data + verify counts vs spec
2. Set up MySQL schema + import CSVs
3. Initial EDA in Jupyter
4. Data cleaning — annual income (NULLs, outliers)
5. Data cleaning — credit score table (NULLs, outliers)
6. Data cleaning — transactions (NULLs, IQR-based outliers)
7. Visualizations: age, gender, location distributions
8. Correlation analysis on credit-profile variables
9. Identify outlier strategy (IQR vs std-dev) — defend the choice
10. Initial target-segment hypothesis
11. Stakeholder feedback meeting (Phase 1 wrap)

Each becomes a JIRA / Trello card with:
- Title + 1-line description
- Acceptance criteria ("done" definition)
- Estimate (S / M / L hours)

### JIRA setup steps (from the lecture)
1. Atlassian → free JIRA Cloud account
2. Create project: "AtliQo Credit Card Phase 1"
3. Choose Kanban template
4. Add backlog items (use the 11 above)
5. Drag the first 1–2 to "In Progress"
6. As you finish, drag to Done

Modern alternatives if you don't want JIRA:
- **GitHub Projects** (free, integrated with the repo)
- **Trello** (lightweight)
- **Notion / Linear** (if you already use them)

> The tool matters less than the habit. Pick one and use it for *every* later project.

---

## 6. The "Tony, Peter, Natasha" meeting — what to take from it

The kickoff meeting in the lecture demonstrates:
- **Asking clarifying questions** — Peter starts with "what does target mean here?"
- **Aligning on success metrics** — credit-utilization? approval-rate? spend?
- **Negotiating scope** — Phase 1 (data prep + segment hypothesis) vs Phase 2 (test, validate)
- **Managing expectations** — what's realistic in 2 weeks vs 2 months

In your real career: every kickoff goes better if you ask these. Don't leave the room until they're answered.

---

## 7. Repo structure for this project

```
atliqo-bank/
├── data/
│   ├── customers.csv
│   ├── credit_profiles.csv
│   ├── transactions.csv
│   └── dim_date.csv
├── sql/
│   ├── schema.sql
│   └── import.sql
├── notebooks/
│   ├── 01-data-validation.ipynb
│   ├── 02-cleaning-customers.ipynb
│   ├── 03-cleaning-credit.ipynb
│   ├── 04-cleaning-transactions.ipynb
│   ├── 05-eda-segments.ipynb
│   └── 06-target-segment-hypothesis.ipynb
├── reports/
│   ├── phase-1-summary.md
│   └── stakeholder-deck.pdf
├── images/
└── README.md
```

---

## 8. Self-check

- [ ] Can I state the AtliQo problem in one sentence?
- [ ] Have I set up a Kanban board (any tool) for this project?
- [ ] Are my 11 task cards in the backlog?
- [ ] Have I imported the CSVs into MySQL or pandas?
- [ ] Do I know who the stakeholders are and what they expect at Phase 1 wrap?
