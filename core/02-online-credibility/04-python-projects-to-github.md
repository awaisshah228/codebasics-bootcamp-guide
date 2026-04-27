# Section 4 — Python Projects for Your Portfolio

## Lectures covered
- Add your Python Project to GitHub
- Share Your Python Skill to the World

---

## Why GitHub for projects (not just LinkedIn)

LinkedIn says "I claim I can do this." GitHub *proves* it.

A pinned, well-documented GitHub project lets a recruiter — or interviewer — see your *actual* code. Reading 50 lines of your code tells them more than 5 LinkedIn posts.

For data/AI roles, **GitHub is the de facto portfolio**.

---

## GitHub profile setup (5-min boost)

### 1. Username
Use real name (`awaisshah228`, `john-smith`). Avoid `dark_lord_42`.

### 2. Avatar
Same headshot as LinkedIn / Discord.

### 3. Profile README — the special trick
Create a repo named exactly **your username** (`awaisshah228/awaisshah228`). Its README appears at the top of your profile page.

Template:
```markdown
### Hi, I'm Awais 👋

I'm a [current role] transitioning to data science via the Codebasics Gen AI & DS Bootcamp 3.0.

**Currently learning**: Math & Stats, Machine Learning
**Recent shipped**: [Hospitality EDA](link), [Expense Tracker](link)
**Tools**: Python · pandas · NumPy · scikit-learn · SQL · FastAPI · Streamlit
**Learning toward**: Data Scientist roles in retail / hospitality

🔗 [LinkedIn](url) · [Portfolio](url) · [Email](mailto:...)
```

### 4. Bio + location + links
- Bio: same as your LinkedIn headline (compressed)
- Location: city
- Website: portfolio URL
- Twitter/X: optional

### 5. Pinned repos
Pin 6 best repos (GitHub limit). For now, pin:
- Your Python projects from this bootcamp
- Any practice / OSS contribution you're proud of
- Your portfolio site repo (when you have one)

Don't pin empty / abandoned / "test" repos. Hide them.

---

## How to push your Codebasics projects to GitHub

### Pre-flight checklist (per project)
- [ ] Code in a clean directory
- [ ] No secrets / API keys committed
- [ ] No huge data files (>50MB) — use `.gitignore`
- [ ] No `__pycache__`, `.venv/`, `.DS_Store`
- [ ] A real README.md with screenshots
- [ ] A `requirements.txt`

### Sample `.gitignore` (Python projects)
```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.DS_Store
*.log
data/raw/
```

### Commands (first push)
```bash
cd hospitality-eda
git init
git add .
git commit -m "Initial: hospitality EDA project"
git branch -M main

# Create empty repo on github.com first, then:
git remote add origin https://github.com/<you>/hospitality-eda.git
git push -u origin main
```

### Subsequent updates
```bash
git status
git add notebook.ipynb reports/insights.md
git commit -m "Add 5 GM insights and revenue-by-city chart"
git push
```

> **Commit messages matter** — `update notebook` is bad; `add cancellation-rate analysis with heatmap` is good. Recruiters do read your commit log.

---

## README — the single most-read file in your repo

### Anatomy of a great project README
```markdown
# Hospitality Domain — Exploratory Data Analysis

> A multi-property hotel chain wanted to identify revenue leaks and underperforming
> segments. This project produced 5 GM-ready insights from 100k+ booking records.

## Demo
![dashboard](images/insight-1.png)

## Problem
[2–4 sentences]

## Dataset
| File | Rows | Description |
|---|---|---|
| fact_bookings.csv | 134,590 | each booking |
| dim_hotels.csv | 25 | property metadata |

## Tools
Python · pandas · matplotlib · seaborn · Jupyter

## Top insights
1. Mumbai Luxury drives 41% of revenue with 28% of bookings.
2. ...

## How to run
\`\`\`bash
git clone <repo>
cd hospitality-eda
pip install -r requirements.txt
jupyter lab notebook.ipynb
\`\`\`

## What I learned
- Star vs snowflake schema — and when each makes sense
- Outlier handling with IQR vs z-score
- ...

## Author
**Awais Shah** — [LinkedIn](url) · [Portfolio](url)
```

### What to include
- A 2-line problem statement
- A demo screenshot or GIF (huge ROI vs. text)
- Dataset table
- Tools list
- 3–5 results / insights
- How to run
- "What I learned" — interviewers love this

### What NOT to include
- Wall of text
- Notebook output dumps
- Personal email/phone
- Unprofessional jokes

---

## "Share your Python skill to the world"

Codebasics frames this lecture as: **once your repo is public, post about it**.

### LinkedIn post — project announcement template
```
Just shipped my first end-to-end data science project as part of the Codebasics Bootcamp.

Project: Hospitality Domain EDA
Problem: a hotel chain needed insights on which properties were leaking revenue.

What I did:
- Joined 5 fact + dim tables (~135k rows)
- Cleaned NULLs, treated outliers via IQR
- Built ~12 visualizations
- Produced 5 prioritized insights for the GM

Tech: Python · pandas · matplotlib · seaborn

Top finding: Mumbai Luxury properties drive 41% of revenue with only 28% of bookings — a segment worth protecting.

GitHub: [link]
What surprised me: the dirtiness of "real" data — half the work was cleaning, not analysis.

#datascience #buildinpublic #python
```

### Hashtag advice
Use **3–5** relevant tags. Too many looks spammy. Useful ones:
`#datascience #python #pandas #buildinpublic #100daysofdata #dataanalysis #portfolio`

### Posting cadence
- 1 post per major milestone (project complete, certification, hard concept)
- 1 commentary post every 2 weeks
- 1 thoughtful comment on others daily

---

## Repo polish checklist (do this before sharing publicly)

- [ ] Repo name is descriptive (`hospitality-eda`, not `project-1`)
- [ ] Repo description filled in (the one-line under the title)
- [ ] Topics added (`data-analysis`, `pandas`, `python`)
- [ ] License chosen (MIT or Apache 2.0 for projects)
- [ ] README has demo screenshot
- [ ] README has "How to run"
- [ ] No secrets / private data committed (check with `git log -p`)
- [ ] Repo is **Public**
- [ ] Pinned to profile

---

## Self-check

- [ ] Profile README repo created and live
- [ ] Both Python projects pushed and pinned
- [ ] Each project has a README with screenshot + insights + how-to-run
- [ ] Posted on LinkedIn for at least one project
- [ ] No `.env` or large CSV in git history
- [ ] Star count on my repo > 0 (ask peers in Discord to star)
