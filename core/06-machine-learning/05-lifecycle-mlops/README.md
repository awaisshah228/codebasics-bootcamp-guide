# ML — AI Project Lifecycle (10 stages) + MLOps

## Lectures covered
- 10 Stages of AI Project Lifecycle
- Requirements and Scope of Work (SOW)
- Data Collection · Data Cleaning · Exploratory Data Analysis
- Feature Engineering
- Model Selection & Training · Model Fine Tuning
- Model Deployment
- Monitoring and Feedback Using ML Ops

---

## The 10 stages — what an end-to-end ML project actually looks like

```
1. Requirements & SOW
2. Data Collection
3. Data Cleaning
4. EDA
5. Feature Engineering
6. Model Selection & Training
7. Model Fine-Tuning
8. Model Evaluation
9. Deployment
10. Monitoring & Feedback (MLOps)
```

This is the lifecycle that converts "we need a model" → "model running reliably in production." Most "ML failures" come from skipping or short-changing one of these stages.

---

## Stage 1 — Requirements & SOW

### What
Translate the business need into a *concrete, measurable* ML problem.

### Output
A 1-2 page **Scope of Work** answering:
- Business objective ("reduce credit defaults by 5%")
- ML translation ("predict probability of default for new applicants")
- Success metric (PR-AUC ≥ 0.7, recall ≥ 70%, etc.)
- Constraints (latency, interpretability, fairness)
- Stakeholders + sign-off
- Timeline + deliverables

### Why this matters
Skipping = building the wrong thing. The most expensive bug.

---

## Stage 2 — Data Collection

### Sources
- Internal databases (SQL)
- Logs / events
- Third-party APIs
- Public datasets / Kaggle
- Manually labeled data
- Web-scraped (with permission)

### Considerations
- **Rights / privacy** — GDPR, HIPAA, internal policy
- **Volume** — minimum sample size for the desired effect
- **Representativeness** — does the sample reflect production?
- **Freshness** — how stale is "old enough to retire"?
- **Imbalance** — is the target rare?

### Output
A frozen, versioned dataset with documentation (data dictionary).

---

## Stage 3 — Data Cleaning

### Common cleaning tasks
- Handling missing values (drop / impute / flag)
- Removing duplicates
- Fixing data types (date strings → datetime)
- Outlier treatment (IQR / std-dev / domain rules)
- Standardizing formats ("USA" / "U.S.A." / "United States")
- Removing PII you don't need

### Output
A cleaned, audit-trail-tracked dataset.

---

## Stage 4 — EDA

Descriptive + visual investigation:
- Univariate: distribution per column
- Bivariate: relationships with target
- Multivariate: correlation heatmap, pair plots
- Time effects, seasonality
- Subgroup analysis (gender, region, etc. — fairness check)

### Output
A markdown report or notebook with **insights and hypotheses**, not just plots.

---

## Stage 5 — Feature Engineering

Where most of the gains in ML competitions come from.

### Common transformations
- Encoding (one-hot, target, ordinal)
- Scaling (StandardScaler, MinMax, Robust)
- Dates → year / month / day-of-week / hour
- Cyclical encoding for hour, month
- Interactions (product of features)
- Polynomial features
- Aggregations (mean of x within group y)
- Domain-specific (BMI, RFM, IV/WOE for credit)
- Text → TF-IDF / embeddings
- Images → CNN features / pre-trained embeddings

### Use Pipelines
Always combine preprocessing steps into a `Pipeline` to prevent leakage and to make models reproducible / deployable.

---

## Stage 6 — Model Selection & Training

### Reasonable progression for tabular data
1. Logistic / Linear baseline
2. Random Forest
3. XGBoost / LightGBM
4. Neural network only if you have a *lot* of data

### Cross-validate every model
With the same CV splitter, compute the same metric. Compare.

### Pick a winner
Or stack/ensemble the top performers if the gain justifies the complexity.

---

## Stage 7 — Model Fine-Tuning

Hyperparameter search:
- RandomizedSearchCV → coarse
- Optuna → fine
- Early stopping for boosting / NN

Don't overdo this — diminishing returns past a point.

---

## Stage 8 — Model Evaluation

The honest evaluation:
- Held-out test set (untouched until now)
- Multiple metrics (don't optimize one, ignore others)
- Subgroup analysis (does the model work on minorities? regions? small accounts?)
- Calibration (if probabilities matter)
- Stress tests (rare events, edge cases)

### The model card
A 1-page summary:
- What the model does
- Training data summary
- Performance (overall + per subgroup)
- Known limitations
- Intended use vs misuse

---

## Stage 9 — Model Deployment

Three patterns:

### 1. Batch
Run model on a schedule, write predictions to a database/dashboard.
- Lowest infra complexity
- Latency = batch frequency

### 2. Real-time API
Wrap model in FastAPI / Flask, serve predictions on request.
- Latency: ms–seconds
- Need scaling, monitoring

### 3. Edge / on-device
Model runs on a phone, browser, IoT device.
- Hardest; requires model compression

### Tooling
- **Streamlit** — quick UI for stakeholders (Codebasics' default for projects)
- **FastAPI** — production-grade Python APIs
- **Docker** — containerize for deployment
- **Cloud** — AWS SageMaker, GCP Vertex AI, Azure ML
- **MLflow / DVC / W&B** — experiment + model versioning

### Codebasics' typical deploy stack
```
model.pkl  ───►  FastAPI service  ───►  Streamlit UI
                       │
                       ▼
                  hosted on Render / Streamlit Cloud
```

---

## Stage 10 — Monitoring & Feedback (MLOps)

A model that ships and is forgotten will degrade silently.

### What to monitor
- **Input drift** — has the input distribution shifted?
- **Output drift** — has the prediction distribution shifted?
- **Concept drift** — has the relationship between input and output changed?
- **Model performance** — when ground-truth labels arrive, recompute the metric
- **Operational** — latency, error rates, throughput

### Tools
- **Evidently AI** — drift detection
- **Whylogs** — data quality monitoring
- **MLflow** — model registry + tracking
- **Prometheus + Grafana** — operational
- **Airflow / Prefect / Dagster** — pipelines

### Retraining triggers
- Performance below threshold
- Significant drift
- Scheduled (e.g., monthly)

---

## A simple end-to-end stack for portfolio projects

```
Data: CSV / SQL
    ↓
Cleaning + FE: pandas + sklearn pipelines
    ↓
Training: sklearn / XGBoost
    ↓
Tracking: MLflow (locally even)
    ↓
Persisting: pickle / joblib
    ↓
Serving: FastAPI
    ↓
UI: Streamlit
    ↓
Hosting: Render / Streamlit Cloud / Hugging Face Spaces
    ↓
Monitoring: simple logging → prom/grafana later
```

For a bootcamp portfolio, **Streamlit + FastAPI + Render + a screen-recorded demo** is enough to be impressive.

---

## Self-check

- [ ] Name the 10 stages without looking.
- [ ] What goes into a Scope of Work?
- [ ] When use batch vs real-time deployment?
- [ ] What's the difference between data drift, concept drift, and operational drift?
- [ ] What is a model card?
- [ ] What's MLflow used for?
- [ ] Walk through an end-to-end pipeline for the credit-risk project: data → deploy → monitor.
- [ ] What's the cheapest meaningful monitoring you can set up for a portfolio project?
