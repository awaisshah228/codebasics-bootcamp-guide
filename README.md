# Codebasics — Gen AI & Data Science Bootcamp 3.0

> Personal companion guide for the **Gen AI & Data Science Bootcamp 3.0: With Practical Job Placement Support & Virtual Internship** by Codebasics.
>
> Each module is in its own folder. Larger modules (Python, SQL, Math/Stats, ML, DL, Gen AI) are further organized into sub-folders.

---

## Source material

- **Public bootcamp page**: https://codebasics.io/bootcamps/gen-ai-data-science-bootcamp-with-virtual-internship
- **Brochure PDF**: [resources/Brochure_GenAI_DS_Bootcamp.pdf](resources/Brochure_GenAI_DS_Bootcamp.pdf)
- **Codebasics YouTube**: https://www.youtube.com/@codebasics
- **Codebasics public GitHub**: https://github.com/codebasics

The curriculum below is verbatim from those sources.

---

## Folder structure

```
data-scient-bootcamp/
├── README.md                        ← you are here
├── PROGRESS.md                      ← weekly progress tracker
├── Brochure_GenAI_DS_Bootcamp.pdf   ← official 49-page brochure
├── core/                            ← 10 core modules, each its own folder
│   ├── 00-bootcamp-introduction/
│   ├── 01-python/
│   │   ├── 00-welcome-and-projects.md
│   │   ├── 01-basics/                  ← installation through pandas EDA
│   │   ├── 02-projects/                ← Hospitality EDA, Expense Tracker
│   │   └── 03-advanced/                ← APIs, FastAPI, pytest, MySQL
│   ├── 02-online-credibility/
│   ├── 03-build-in-public/             ← Git/GitHub fundamentals
│   ├── 04-sql/
│   │   ├── 01-basics/
│   │   └── 02-advanced/
│   ├── 05-math-statistics/
│   │   ├── 01-foundations/
│   │   ├── 02-atliqo-bank-project/
│   │   └── 03-inferential/
│   ├── 06-machine-learning/
│   │   ├── 01-foundations/
│   │   ├── 02-classification/
│   │   ├── 03-ensemble/
│   │   ├── 04-unsupervised/
│   │   ├── 05-lifecycle-mlops/
│   │   └── 06-projects/
│   ├── 07-deep-learning/
│   │   ├── 01-foundations/
│   │   ├── 02-training/
│   │   ├── 03-vision/
│   │   ├── 04-sequence/
│   │   ├── 05-deployment/
│   │   └── 06-projects/
│   ├── 08-nlp/
│   └── 09-gen-ai-agentic-ai/
│       ├── 01-foundations/
│       ├── 02-rag/
│       ├── 03-orchestration/
│       ├── 04-agents/
│       └── 05-projects/
├── career/                          ← career track (LinkedIn, GitHub, ATS, etc.)
├── internship/                      ← Virtual Internship 1 & 2
├── practice/                        ← practice rooms (Python, ML, DL, NLP)
├── supplementary/                   ← live webinars, problem-solving sessions, etc.
├── resources/                       ← cheatsheets, datasets, templates
└── projects/                        ← portfolio projects index
```

---

## Curriculum map (10 core modules)

| # | Module | Status | Folder |
|---|---|---|---|
| 0 | Welcome to the Bootcamp Experience | ✅ 100% | [core/00-bootcamp-introduction](core/00-bootcamp-introduction/) |
| 1 | Python: Beginner to Advanced | ✅ 100% | [core/01-python](core/01-python/) |
| 2 | Online Credibility | ✅ 100% | [core/02-online-credibility](core/02-online-credibility/) |
| 3 | Build In Public | ✅ 100% | [core/03-build-in-public](core/03-build-in-public/) |
| 4 | SQL for Data Science | ✅ 100% | [core/04-sql](core/04-sql/) |
| 5 | Math & Statistics | ✅ 100% | [core/05-math-statistics](core/05-math-statistics/) |
| 6 | Machine Learning | ✅ 100% | [core/06-machine-learning](core/06-machine-learning/) |
| 7 | Deep Learning | 🔒 0% | [core/07-deep-learning](core/07-deep-learning/) |
| 8 | Natural Language Processing | 🔒 0% | [core/08-nlp](core/08-nlp/) |
| 9 | Gen AI & Agentic AI | 🔒 0% | [core/09-gen-ai-agentic-ai](core/09-gen-ai-agentic-ai/) |

Status legend: ✅ Complete · 🟡 In-progress · 🔒 Locked · 🔵 Available

---

## 13–15 portfolio projects (across the bootcamp)

| # | Project | Module |
|---|---|---|
| 1 | Hospitality Domain EDA | Python |
| 2 | Expense Tracking System (FastAPI + Streamlit + MySQL) | Python |
| 3 | Finance & Top-N Insights (UDFs, stored procedures) | SQL |
| 4 | Supply Chain Analytics & Optimisation | SQL |
| 5 | AtliQo Bank — Credit Card Launch | Math & Stats |
| 6 | Healthcare Premium Prediction (regression) | ML |
| 7 | Credit Risk Modeling (NBFC, classification) | ML |
| 8 | Beverage Price Range Prediction (multi-class) | ML |
| 9 | Car Damage Detection (CNN + transfer learning) | DL |
| 10 | Real Estate Assistant using RAG | Gen AI |
| 11 | E-Commerce Chatbot (intent routing + tools) | Gen AI |
| 12 | Agentic AI HR Onboarding (MCP + Claude) | Gen AI |
| 13 | Customer Care Agent (Bedrock AgentCore) | Gen AI |
| 14 | Stale Fruit Detector (CNN, agri-tech) | Virtual Internship 2 |
| 15 | RAG-based Q&A for Healthcare (PubMed + Llama 3) | Virtual Internship 2 |

---

## How to use this guide

### Daily / weekly rhythm

1. **Before a lecture** — open the relevant module / part file. Skim *Goal* and *Key concepts*.
2. **During a lecture** — pause and type along.
3. **After a lecture** — fill in the personal-notes section.
4. **For tasks / exercises** — paste the official Codebasics task description into the file and write your solution there.
5. **Before moving on** — answer the *Self-check* questions out loud. If anything is fuzzy, re-watch.
6. **Weekly** — update [PROGRESS.md](PROGRESS.md).

### File structure within each module

Each module's folder contains:
- `README.md` — overview, full curriculum (verbatim), file index, module-level goal & self-check
- `<part>.md` files — one per major section, each with:
  - Lectures covered
  - Concept explanations + code
  - Practice exercises
  - Common pitfalls
  - Self-check questions
  - Personal-notes section

### Status conventions

- ✅ **Complete** — 100% finished, all tasks submitted
- 🟡 **In-progress** — currently studying
- 🔒 **Locked** — not yet unlocked in the portal
- 🔵 **Available** — unlocked but not started

---

## My commitments (sign here)

- Hours / week: **___**
- Daily slot: **___ to ___**
- Target completion: **___ (date)**
- LinkedIn build-in-public posts / week: **___ (min 1)**
- Target role: **___**

---

## External anchors

- Codebasics learner portal: https://codebasics.io
- Codebasics YouTube: https://www.youtube.com/@codebasics
- Codebasics GitHub: https://github.com/codebasics
- My LinkedIn: _(add)_
- My GitHub: _(add)_
- My portfolio site: _(add when built)_
