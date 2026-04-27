# Module 5 — Math & Statistics for AI / Data Science

> **Duration**: 12h 49m 38s · 98 lectures
> **Status**: 🟡 In-progress (20%) — **active module as of 2026-04-27**
> **Domain project**: AtliQo Bank Credit Card Launch (50k+ records)

## Why this module is the actual cornerstone

Most learners skim this module to "get to ML faster." It's the wrong move. Statistics is what makes ML / Gen AI claims **defensible** — without it, you don't know whether your 87% accuracy is real or noise, whether your A/B test winner is luck, or whether your 5% RAG-hallucination rate is statistically different from 7%.

Codebasics' module is also unusually project-driven: ~1/3 of the time is the **AtliQo Bank** business case — exactly how stats is used in a working analytics team.

## Folder layout

```
05-math-statistics/
├── README.md
├── 01-foundations/                       ← descriptive + probability + distributions
│   ├── README.md
│   ├── 01-data-visualization-basics.md
│   ├── 02-central-tendency-dispersion.md
│   ├── 03-probability-theory.md
│   └── 04-distributions.md
├── 02-atliqo-bank-project/               ← the integrated project across stats topics
│   ├── README.md
│   ├── 01-project-setup-kanban.md
│   └── 02-phase-1-find-target-market.md
└── 03-inferential/                       ← CLT + hypothesis testing + A/B
    ├── README.md
    ├── 01-central-limit-theorem.md
    └── 02-hypothesis-testing.md
```

## Curriculum (verbatim from public outline)

### Section 1 — Welcome / Course Overview
### Section 2 — Data & Visualization Basics (9 lectures)
- Types of Data · Pie/Bar · Histograms/Line · Scatter/Bubble · Univariate/Bivariate/Multivariate
### Section 3 — Central Tendency & Dispersion (18 lectures)
- Descriptive vs Inferential · Mean/Median/Mode · Percentile · Range/IQR · Box plot · Outlier treatment · Variance/Std Dev · Stock returns case · Correlation vs Causation
### Section 4 — Probability Theory (7 lectures)
- Probability basics · Addition/Multiplication rule · Conditional · Bayes Theorem
### Section 5 — Distributions (12 lectures)
- What's a distribution · Skewness · Normal · Outliers via Normal · Z score · Standard Normal Distribution
### Section 6 — AtliQo Bank Project — Kickoff (7 lectures)
- Bashneer Grower's Bank · Credit Card Launch · Project kickoff · Kanban · JIRA setup · Tony/Peter/Natasha meeting
### Section 7 — Phase 1: Find Target Market (21 lectures)
- Data validation · MySQL setup · Jupyter import · Cleaning + outlier treatment · Visualization · Hypothesis · Phase 1 stakeholder feedback
### Section 8 — Central Limit Theorem (12 lectures)
- Random sampling · LLN · CLT · Sampling distribution · Standard error · Z-table · Confidence interval
### Section 9 — Hypothesis Testing (~19 lectures)
- Null vs alternate · Z test · p-value · Statistical power · Effect size · A/B testing · t-test · Chi-squared (goodness-of-fit + independence)

## Module-level goal

After this module:
- I can read any DS paper's stats section without panicking
- I can defend any "this is significant" claim with proper test + p-value + effect size
- I can design and analyze a real A/B test
- I built the AtliQo Bank target-market analysis end-to-end and produced a stakeholder report
- I can translate "the model improved accuracy from 84% to 87%" into "is that improvement statistically meaningful?"

## Module self-check

- [ ] Explain mean vs median to a non-technical friend, and when each lies
- [ ] State Bayes' theorem and walk through the medical-test example
- [ ] Describe IQR-based outlier detection
- [ ] Explain why Normal distribution shows up everywhere (CLT)
- [ ] Walk through a hypothesis test: H₀, H₁, test, p-value, decision
- [ ] Design an A/B test for a website CTA: sample size, metric, duration, decision
- [ ] What's the difference between p-value and effect size?
- [ ] When do I use chi-squared instead of t-test?
