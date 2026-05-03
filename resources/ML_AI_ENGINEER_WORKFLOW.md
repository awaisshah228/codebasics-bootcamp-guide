# The Full ML / AI Engineer Workflow — End-to-End Reference

> A practical, opinionated reference you can come back to every time you start
> a new ML or AI project. Covers the full lifecycle: from "we have a business
> problem" → data → model → deployment → monitoring → feedback → retraining.

---

## The Big Picture (One-Diagram View)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       1. PROBLEM FRAMING                                 │
│       business goal → ML problem type → success metric                   │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       2. DATA COLLECTION                                 │
│   sources → ingestion → raw storage (data lake / warehouse / files)      │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       3. DATA CLEANING & VALIDATION                      │
│   missing / dupes / outliers / types / schema checks                     │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       4. EDA & FEATURE ENGINEERING                       │
│   distributions → correlations → new features → encoding / scaling       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       5. SPLITTING & BASELINE                            │
│   train / val / test → dummy model → metric floor                        │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       6. MODEL TRAINING & TUNING                         │
│   pick algorithm → train → cross-validate → hyper-parameter search       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       7. EVALUATION                                      │
│   metrics → error analysis → bias / fairness → final test set            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       8. PACKAGING & VERSIONING                          │
│   serialize model → version data + code + model → register               │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       9. DEPLOYMENT                                      │
│   batch / REST API / streaming / edge → CI/CD → canary / A-B             │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       10. MONITORING                                     │
│   service health + data drift + concept drift + business KPI             │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       11. FEEDBACK LOOP & RETRAIN                        │
│   ground-truth collection → labels → trigger retrain → back to step 2    │
└──────────────────────────────────────────────────────────────────────────┘
```

The whole loop is what people mean by **MLOps**. Steps 1-7 are "data science",
steps 8-11 are what turns a notebook into a product.

---

## Stage 1 — Problem Framing

Before touching any data, write down four things on a single page:

| Question | Example |
|---|---|
| What is the **business problem**? | "Reduce churn in monthly subscribers." |
| What **ML problem** does that map to? | Binary classification: will user churn next 30 days? |
| What **decision** does the prediction drive? | Send a discount email to predicted churners. |
| What **metric** defines success? | Precision @ top-10% AND uplift in retained revenue. |

**Common pitfall:** picking accuracy when the classes are 95/5 imbalanced.
Pick a metric that punishes the failure mode you actually care about
(precision, recall, F1, ROC-AUC, MAE, RMSE, MAPE, NDCG, BLEU, etc.).

**Deliverable:** a one-page "ML project brief" with goal, metric, baseline,
constraints (latency, cost, privacy), and stakeholders.

---

## Stage 2 — Data Collection

### 2.1 Identify sources
- Internal: production DB, data warehouse, application logs, event streams.
- External: public datasets, third-party APIs, web scraping, partners.
- Generated: surveys, manual labelling, synthetic data, simulations.

### 2.2 Ingest into a stable place
Don't train off the live OLTP database. Land raw data in a **data lake** or
**warehouse** so you can re-process it later.

| Pattern | Tools |
|---|---|
| Batch ingestion | Airflow, Prefect, Dagster, cron + Python |
| Streaming | Kafka, Kinesis, Pub/Sub, Vercel Queues |
| Storage (raw) | S3 / GCS / Azure Blob, Vercel Blob |
| Storage (structured) | Postgres, BigQuery, Snowflake, Redshift, DuckDB |

### 2.3 Rules of the road
- **Always keep an immutable raw copy.** Cleaning is destructive.
- **Document the schema** (column, type, unit, source, timestamp).
- **Track lineage**: which pipeline produced this file/table?
- **Respect privacy**: PII must be encrypted, masked, or tokenised at rest.
- **Version the data**: DVC, LakeFS, or just date-partitioned folders.

---

## Stage 3 — Data Cleaning & Validation

You will spend **60-80%** of project time here. Accept it.

### 3.1 The cleaning checklist
1. **Schema check** — column names, dtypes, ranges match expectations.
2. **Missing values** — impute (mean/median/mode/KNN), flag, or drop.
3. **Duplicates** — exact and near-duplicates (fuzzy matching).
4. **Outliers** — investigate, don't auto-delete; cap, transform, or keep.
5. **Inconsistent categories** — `"USA"`, `"U.S.A"`, `"us"` → one canonical form.
6. **Data types** — dates as `datetime`, IDs as strings, booleans as bool.
7. **Encoding issues** — UTF-8 vs latin-1, trailing whitespace, smart quotes.
8. **Time zones** — store everything in UTC, convert at display time.
9. **Leakage** — drop columns that wouldn't exist at inference time.

### 3.2 Automate validation
Don't eyeball — codify. Tools:
- `pandera`, `great_expectations` — schema + value-range tests on dataframes.
- `pydantic` — typed contracts for records crossing service boundaries.
- Run validation **in the pipeline** so bad data fails the run, not the model.

---

## Stage 4 — EDA & Feature Engineering

### 4.1 Exploratory Data Analysis
Goal: build intuition about what signal exists.

- Univariate: histograms, value counts, summary stats.
- Bivariate: correlation matrix, scatter, group-by means, mutual information.
- Target leakage hunt: which columns are *too* predictive?
- Visual tools: matplotlib, seaborn, plotly, sweetviz, ydata-profiling.

### 4.2 Feature engineering — where you actually win
Better features beat fancier models nine times out of ten.

| Type | Examples |
|---|---|
| Numeric transforms | log, sqrt, polynomial, bucketing |
| Datetime | hour, day-of-week, days-since-event, is-weekend |
| Categorical | one-hot, target/mean encoding, embeddings, hashing |
| Text | TF-IDF, n-grams, sentence embeddings (BERT, OpenAI, etc.) |
| Aggregations | rolling mean, last-7-day count, ratio of A to B |
| Interactions | feature_a × feature_b, ratios, differences |
| Domain-specific | RFM for retail, MFCCs for audio, OHLCV for finance |

### 4.3 Avoid leakage
- Compute aggregations using only data available **before** the prediction time.
- Fit scalers / encoders on **train only**, then apply to val/test.
- For time series: split chronologically, no random shuffle.

### 4.4 Feature stores (when you scale up)
Tools like Feast, Tecton, or Vertex AI Feature Store give you the same
feature definitions in training and serving — eliminating online/offline skew.

---

## Stage 5 — Splitting & Baseline

### 5.1 Splits
| Data type | Strategy |
|---|---|
| IID tabular | Random 70/15/15 |
| Time series | Chronological — train past, validate recent, test newest |
| Grouped (e.g. multiple rows per user) | Group-aware split — no user in two splits |
| Imbalanced | Stratified by target class |

**Lock the test set.** Touch it once, at the very end. If you peek at it during tuning, you no longer have an honest estimate of generalisation.

### 5.2 Always build a dumb baseline first
- Classification: predict majority class.
- Regression: predict the mean / last value.
- Time series: predict yesterday's value (naïve forecast).

If your shiny model can't beat this, something is wrong.

---

## Stage 6 — Model Training & Tuning

### 6.1 Pick the right family

| Problem | First thing to try | Why |
|---|---|---|
| Tabular | Gradient boosting (XGBoost / LightGBM / CatBoost) | Wins most Kaggle tabular comps |
| Tabular, tiny data | Logistic / linear regression | Interpretable, low variance |
| Image | Pre-trained CNN / ViT + fine-tune | Transfer learning is cheap |
| Text classification | Transformer (DistilBERT, RoBERTa) + fine-tune | SOTA + tooling is mature |
| Generation / reasoning | LLM via API + prompt engineering / RAG / fine-tune | Train-from-scratch rarely justified |
| Time series | ARIMA / Prophet / LightGBM with lag features | Often beats deep models on small data |
| Recommendations | Two-tower / matrix factorisation / LightFM | Scales, well-understood |

### 6.2 Train responsibly
- Set **random seeds** for reproducibility.
- Track **every experiment**: params, code commit, data version, metrics.
  Tools: MLflow, Weights & Biases, Neptune, Comet, DVC.
- Use **cross-validation** for small datasets (k-fold, stratified k-fold,
  time-series split).

### 6.3 Hyperparameter tuning
- Don't tune until baseline + simple model is working.
- Search strategies: grid (small), random (medium), Bayesian (Optuna, Hyperopt).
- Tune on **validation**, never on test.
- Stop early when validation plateaus to save compute.

---

## Stage 7 — Evaluation

### 7.1 Pick metrics that match the decision

| Task | Default metrics |
|---|---|
| Binary classification | ROC-AUC, PR-AUC, F1, precision/recall @ threshold |
| Multi-class | Macro/micro F1, top-k accuracy, log-loss |
| Regression | RMSE, MAE, MAPE, R² |
| Ranking / search | NDCG, MAP, MRR, Hit-Rate@k |
| Generation | BLEU, ROUGE, BERTScore, human eval, LLM-as-judge |
| Time series | MAPE, sMAPE, RMSE per horizon |

### 7.2 Beyond a single number
- **Confusion matrix** — where does it fail?
- **Error analysis** — pull 100 worst predictions, look at them by hand.
- **Slices** — performance per segment (region, device, age group).
  A model that's 90% accurate overall but 50% on women is broken.
- **Calibration** — do predicted probabilities match real frequencies?
- **Robustness** — test on perturbed / out-of-distribution inputs.
- **Fairness & bias** — disparate impact, equal opportunity, demographic parity.

### 7.3 Final test
Touch the test set **once**. Report the number with a confidence interval
(bootstrap is fine). If it's much worse than validation, you've overfit
to the validation set.

---

## Stage 8 — Packaging & Versioning

A model is not a `.pkl` file — it's the **triple of (data, code, weights)**.
You need to be able to reproduce a prediction six months from now.

### 8.1 Version everything
| Artifact | Tool |
|---|---|
| Code | Git |
| Data | DVC, LakeFS, snapshot tables |
| Models | MLflow Model Registry, SageMaker, Vertex, S3 with versioning |
| Environment | `requirements.txt` / `pyproject.toml` + Docker image |

### 8.2 Serialization formats
- `pickle` / `joblib` — fast, Python-only, version-fragile.
- `ONNX` — cross-language, good for edge deployment.
- `TorchScript` / `SavedModel` — for PyTorch / TensorFlow respectively.
- `safetensors` — for transformer weights (no pickle CVEs).

### 8.3 Build a model card
A short markdown file shipped with every model:
- What it predicts, on what inputs, with what limits.
- Training data summary and date range.
- Evaluation metrics, including per-slice performance.
- Known biases and intended use vs. *not*-intended use.

---

## Stage 9 — Deployment

### 9.1 Pick the serving pattern

| Pattern | When | Tools |
|---|---|---|
| **Batch** | Predictions used hourly/daily (churn list, lead scores) | Airflow + warehouse, Spark |
| **REST / gRPC API** | Real-time, single request | FastAPI, BentoML, TorchServe, KServe, Vercel Functions |
| **Streaming** | Continuous events (fraud, anomaly) | Kafka + Flink / Spark Streaming |
| **Edge / on-device** | Privacy, offline (mobile, IoT) | ONNX Runtime, Core ML, TF Lite |
| **LLM-as-API** | Generative apps | OpenAI, Anthropic, Vercel AI Gateway |

### 9.2 The deployment stack
1. Wrap model in a **service** (FastAPI, Flask, BentoML).
2. Containerise with **Docker** — pin Python version + deps.
3. Push to a **registry** (ECR, GHCR, Docker Hub).
4. Deploy to a **runtime** (Kubernetes, ECS, Cloud Run, Lambda, Vercel).
5. Front it with an **API gateway** for auth, rate limiting, observability.

### 9.3 Roll out safely
- **Shadow mode** — run new model alongside old, log both, compare offline.
- **Canary** — route 1% → 10% → 50% → 100% of traffic over hours/days.
- **A/B test** — randomly assign users, measure business KPI, not just ML metric.
- **Feature flags** — instant rollback without redeploy.
- **Blue/green** — keep last good version live, swap traffic atomically.

### 9.4 CI/CD for ML
A typical pipeline (GitHub Actions, GitLab CI, etc.):

```
push → lint → unit tests → data validation → train (if data changed)
     → eval against thresholds → register model → deploy to staging
     → integration tests → manual approve → canary → prod
```

---

## Stage 10 — Monitoring

The model that's perfect on launch day will be wrong in 3 months.
You **must** monitor it.

### 10.1 Four layers of monitoring

| Layer | What it answers | Examples |
|---|---|---|
| **Service** | Is the API up? | Latency p50/p95/p99, error rate, QPS |
| **Data** | Are inputs the same shape as training? | Feature drift (PSI, KS test), null rate spike |
| **Model** | Are predictions still well-calibrated? | Score distribution drift, confidence drop |
| **Business** | Is it still making money / saving costs? | Conversion, retention, revenue per prediction |

### 10.2 Key concepts
- **Data drift**: input distribution changes (more mobile users than before).
- **Concept drift**: relationship between input and target changes
  (post-COVID consumer behaviour ≠ pre-COVID).
- **Label delay**: ground truth arrives weeks/months later (loan default).
- **Prediction drift**: outputs shift even if you can't see the truth yet —
  this is your earliest warning.

### 10.3 Tooling
- Open source: Evidently AI, NannyML, WhyLabs, Arize Phoenix.
- Hosted: Arize, Fiddler, Datadog ML Observability, Vercel Observability.
- Always export to your standard observability stack (Prometheus, Grafana,
  Datadog, OpenTelemetry) so on-call sees ML alerts alongside everything else.

### 10.4 Alerts that matter
- Input feature drift > threshold for N hours.
- Prediction distribution shift > threshold.
- Performance metric drops below SLA (when ground truth is available).
- Latency p95 > X ms.
- Error rate > Y %.
- **Don't alert on absolutes** ("predictions = 0.6 today") — alert on **change**.

---

## Stage 11 — Feedback Loop & Retraining

This is where most teams give up. Don't.

### 11.1 Capture ground truth
You need labels to know if the model is still correct. Ways to get them:

| Source | Example |
|---|---|
| **Implicit** | User clicked the recommendation → positive label |
| **Explicit** | Thumbs up/down, star rating, "was this helpful?" |
| **Delayed truth** | Did the user actually churn? Did the loan default? |
| **Human review** | Sample N% of predictions, send to labellers |
| **Active learning** | Send the model's *most uncertain* predictions for review |

Log the **prediction**, the **input**, the **prediction ID**, and later
**join with the outcome**. Without this join key, you have no feedback loop.

### 11.2 Close the loop
```
prediction logged → outcome captured → joined → fresh labelled dataset
→ scheduled re-training → evaluation gate → auto-deploy if better
                                        ↘ alert humans if worse
```

### 11.3 When to retrain
- **Cadence** (daily / weekly / monthly) — simple, predictable.
- **Performance trigger** — when monitoring shows accuracy dropped.
- **Drift trigger** — when feature distribution drifted past threshold.
- **Volume trigger** — every N new labelled examples.
- **Event trigger** — new product launch, policy change, market shock.

### 11.4 Retraining is not free
- Always **evaluate the new model against the live one** before shipping —
  newer ≠ better.
- Keep the **previous model warm** so you can roll back instantly.
- Watch for **feedback loops that bias future data** (a recommender that
  only shows what it predicts users will click → never learns what they
  *would* click on items it never showed).

---

## Bonus — The GenAI / LLM Variant

The same workflow applies, with a few twists:

| Stage | LLM-app version |
|---|---|
| Data | Documents → chunks → embeddings → vector DB |
| Feature eng. | Prompt design, system messages, few-shot examples |
| Training | Mostly *not* training — RAG, prompt tuning, light fine-tune |
| Eval | Golden Q-A sets, LLM-as-judge, human eval, hallucination rate |
| Deploy | API route → LLM provider, with caching + fallback (AI Gateway) |
| Monitor | Token cost, latency, refusal rate, hallucination, jailbreaks |
| Feedback | Thumbs up/down, conversation length, follow-up questions |
| Retrain | Update RAG index, refresh prompts, periodic fine-tune |

---

## The Engineer's Checklist (Print This)

Before you say "shipped":

- [ ] Business goal + ML metric written down and agreed.
- [ ] Raw data is immutable, documented, and version-controlled.
- [ ] Data validation runs in every pipeline run.
- [ ] Train / val / test split is appropriate and locked.
- [ ] Baseline beaten by chosen model with margin.
- [ ] Per-slice performance evaluated; no group is silently failing.
- [ ] Test set touched once; result reported with CI.
- [ ] Code, data, model, environment all versioned together.
- [ ] Model card published.
- [ ] Inference service has tests, auth, rate limits, retries, timeouts.
- [ ] Rollout uses canary / shadow / A-B — not full-fleet day one.
- [ ] Monitoring covers service + data + model + business KPI.
- [ ] Ground-truth join key logged for every prediction.
- [ ] Retraining trigger and pipeline exist (even if manual).
- [ ] Rollback procedure documented and tested.

---

## Tooling Cheat-Sheet (One-Stop)

| Job | Pick one to learn first |
|---|---|
| Notebooks | Jupyter, VS Code notebooks |
| DataFrames | pandas, Polars |
| Viz | matplotlib + seaborn |
| Validation | pandera, Great Expectations |
| Classical ML | scikit-learn |
| Boosting | XGBoost, LightGBM |
| Deep learning | PyTorch (+ Lightning) |
| NLP | Hugging Face Transformers |
| Experiment tracking | MLflow |
| Pipelines | Airflow or Prefect |
| Versioning | Git + DVC |
| Serving | FastAPI + Docker |
| Orchestration | Kubernetes (or Cloud Run / Vercel) |
| Monitoring | Evidently AI |
| Vector DB | pgvector, Qdrant, Pinecone |
| LLM apps | OpenAI / Anthropic SDK + LangChain or LlamaIndex |

---

## Further Reading
- **"Designing Machine Learning Systems"** — Chip Huyen (the canonical book).
- **"Building Machine Learning Powered Applications"** — Emmanuel Ameisen.
- **"Made With ML"** — free MLOps course (madewithml.com).
- **Google's "Rules of Machine Learning"** — every line is gold.
- **Andrew Ng's "Machine Learning Yearning"** — short, free, focuses on strategy.

---

*Keep this file open in a tab whenever you start a new ML project. Jump to
the stage you're in and run through the checklist for that stage.*
