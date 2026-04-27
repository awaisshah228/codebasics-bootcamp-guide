# Module 6 — Master Machine Learning for Data Science & AI (Beginner → Advanced)

> **Status**: 🔒 Locked (after Math/Stats)
> **Domain projects**: Healthcare Premium Prediction (regression) · Credit Risk Modeling (classification) · Beverage Price Range Prediction

## Why this module is the heart of the bootcamp

Every previous module supports this one. SQL fetches data; Python wrangles it; stats explains it; **ML predicts on it**. Most "data scientist" job descriptions reduce to: *can you take a business problem, frame it as ML, train a baseline, evaluate honestly, ship it?*

Codebasics' module covers **classical ML end-to-end** — algorithms (regression, classification, clustering), evaluation, the project lifecycle, and the MLOps tail (deployment + monitoring). Deep learning is its own module afterward.

## Folder layout

```
06-machine-learning/
├── README.md
├── 01-foundations/
│   ├── README.md
│   ├── 01-intro-and-categories.md
│   ├── 02-linear-regression.md
│   ├── 03-gradient-descent-cost.md
│   ├── 04-model-evaluation-regression.md
│   ├── 05-preprocessing-encoding.md
│   ├── 06-overfit-underfit-bias-variance.md
│   └── 07-regularization.md
├── 02-classification/
│   ├── README.md
│   ├── 01-logistic-regression.md
│   ├── 02-classification-metrics.md
│   ├── 03-svm.md
│   ├── 04-naive-bayes.md
│   ├── 05-decision-tree.md
│   ├── 06-class-imbalance.md
│   └── 07-roc-auc.md
├── 03-ensemble/
│   ├── README.md
│   ├── 01-bagging-random-forest.md
│   ├── 02-boosting-adaboost-gbm-xgb.md
│   └── 03-cross-validation-tuning.md
├── 04-unsupervised/
│   ├── README.md
│   ├── 01-clustering-kmeans.md
│   └── 02-vif.md
├── 05-lifecycle-mlops/
│   └── README.md
└── 06-projects/
    ├── README.md
    ├── 01-healthcare-premium-regression.md
    ├── 02-credit-risk-classification.md
    └── 03-beverage-price.md
```

## Curriculum (verbatim from brochure)

### Foundations
- Introduction to ML · Classification vs Regression · Supervised vs Unsupervised
- Simple Linear Regression · Multiple Linear Regression
- Cost Function · Derivatives & Partial Derivatives · Chain Rule
- Gradient Descent · Model Evaluation
- Data Preprocessing: One-Hot Encoding · Polynomial Regression
- Overfitting / Underfitting · L1 / L2 Regularization · Bias-Variance Tradeoff

### Classification
- Logistic Regression: Binary + Multiclass · Cost: Log Loss
- Accuracy, Precision, Recall, F1, Confusion Matrix
- SVM · Naive Bayes · Decision Tree
- Data Preprocessing: Scaling · sklearn Pipeline
- Class Imbalance · ROC Curve & AUC · Cost-benefit analysis

### Ensemble
- Voting (majority, average, weighted) · Bagging
- Random Forest · Boosting (AdaBoost · Gradient Boosting · XGBoost)
- K-Fold + Stratified K-Fold Cross Validation · Hyperparameter Tuning

### Unsupervised + utilities
- K-Means Clustering
- Variance Inflation Factor (VIF)

### AI Project Lifecycle (10 stages)
- Requirements / SOW · Data Collection · Data Cleaning + EDA
- Feature Engineering · Model Selection + Training · Fine Tuning
- Deployment · Monitoring & Feedback (MLOps)

### Projects
- Healthcare Premium Prediction (regression) — end-to-end deployment with Streamlit
- Credit Risk Modeling (NBFC, classification) — WOE/IV, KS Statistic, Gini, Optuna, Streamlit
- Beverage Price Range Prediction (classification on survey data)

## Module-level goal

After this module:
- Frame any tabular ML problem (regression, binary class, multiclass, clustering)
- Train, tune, evaluate using sklearn pipelines
- Pick the right metric for the problem (and explain why)
- Understand bias-variance, regularization, cross-validation
- Deploy a model to a Streamlit app + FastAPI service
- Have 3 portfolio projects across healthcare, finance, and consumer survey domains

## Module-level self-check

- [ ] Bias-variance tradeoff in 30 seconds
- [ ] When use accuracy vs precision/recall vs ROC-AUC
- [ ] Why scale features for k-NN but not random forest
- [ ] Difference between bagging and boosting
- [ ] When use L1 vs L2 regularization
- [ ] Walk through k-fold cross validation
- [ ] How does XGBoost differ from a random forest mathematically
- [ ] How would you handle 95% class imbalance
