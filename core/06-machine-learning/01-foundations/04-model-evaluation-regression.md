# Foundations 4 — Model Evaluation (Regression)

## Lectures covered
- Model Evaluation

---

## 1. The evaluation question

After training, two questions:
1. **How wrong** is the model on average?
2. **How well** does it explain the variance in y?

Different metrics answer these differently. Pick the one matching your problem.

---

## 2. The four standard regression metrics

### MAE — Mean Absolute Error
$$\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$$
- Same units as y → directly interpretable ("off by $X dollars on average")
- Robust to outliers (linear penalty)
- Not differentiable at 0 (a quirk for optimization)

### MSE — Mean Squared Error
$$\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$$
- Penalizes big errors quadratically
- Units: y² (not directly interpretable)
- Differentiable everywhere → used as the **loss** during training

### RMSE — Root Mean Squared Error
$$\text{RMSE} = \sqrt{\text{MSE}}$$
- Same units as y
- Penalizes big errors more than MAE
- Default for "report a single number" in many contexts

### R² — Coefficient of Determination
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- Range: ≤ 1 (1 = perfect; 0 = no better than mean; negative = worse than mean)
- "Fraction of variance explained"
- **Caveat**: high R² doesn't mean a useful model — it depends on baseline variance

### MAPE — Mean Absolute Percentage Error
$$\text{MAPE} = \frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$$
- Easy to communicate ("we're off by 8%")
- Breaks when y is near 0 (division by ~0)

### sklearn shape
```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              root_mean_squared_error, r2_score,
                              mean_absolute_percentage_error)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)        # sklearn ≥1.4
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
```

---

## 3. Picking the right metric

| If... | use |
|---|---|
| You need a single, interpretable number | MAE or RMSE |
| Big errors are especially bad | RMSE / MSE |
| Outliers shouldn't dominate | MAE |
| The audience speaks in % | MAPE |
| You need "how good is this *vs* a constant baseline" | R² |
| Comparing models on the same data | any — pick one and stick with it |

> **For business reports, MAE in dollars/units is almost always the right choice.** R² is for academics.

---

## 4. Adjusted R² — when adding features

R² *always* goes up (or stays the same) when you add features, even garbage ones. Adjusted R² penalizes for free parameters:

$$R^2_{adj} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$

Where $p$ = number of features.

If adjusted R² goes up after adding a feature, the feature was probably useful.

---

## 5. Residual diagnostics — beyond a single number

A single metric can hide patterns. Always plot residuals.

### Residuals vs predicted
```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals); plt.axhline(0, color="red")
```

What you want: random cloud around 0.

What you don't want:
- **Funnel shape** → heteroscedasticity (variance grows with prediction)
- **U-shape / curve** → non-linearity, model is misspecified
- **Clusters** → missing categorical feature

### Residual histogram
```python
sns.histplot(residuals, kde=True)
```
Should be roughly Normal-ish around 0. Heavy skew → model is biased on certain ranges.

### Predicted vs actual
```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")     # 45° line
```
Points should hug the diagonal.

---

## 6. Train vs test metrics — the diagnostic

| Train metric | Test metric | Diagnosis |
|---|---|---|
| Good | Good | ✅ healthy model |
| Good | Bad (gap) | overfit — too complex |
| Bad | Bad | underfit — too simple, or bad data |
| Bad | Good | unusual; check for shuffling bug |

Always look at both — a test number alone hides overfitting.

```python
print("train R²:", model.score(X_train, y_train))
print("test  R²:", model.score(X_test, y_test))
```

---

## 7. Cross-validation — beyond a single split

A single train/test split has variance — your one number could be lucky/unlucky. Cross-validation averages over multiple splits.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(scores.mean(), "±", scores.std())
```

We dive deep into CV in the **ensemble subfolder** (`03-cross-validation-tuning.md`).

---

## 8. Real example — interpreting an evaluation

You build a house-price model and report:
- MAE: $18,500
- RMSE: $32,000
- R²: 0.84
- MAPE: 9%

What this means:
- On average, prediction is off by $18.5k (MAE)
- A few large errors push RMSE to $32k (so heavy-tailed errors)
- Model captures 84% of price variance
- 9% mean percentage error → reasonable for real estate

**One-line summary**: "Our model predicts house prices with ~9% MAPE; performance degrades on luxury homes (>$2M)."

That's the kind of sentence stakeholders read. Lead with that, support with a chart.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Reporting train metric only | hides overfitting | always show train + test |
| Only R² | doesn't say the magnitude of error | pair with MAE / RMSE |
| MAPE on near-zero y | huge fluctuations | use SMAPE or absolute error |
| Comparing models on different splits | apples to oranges | use cross-val with `random_state` fixed |
| RMSE in different units across reports | can't compare | always include units |

## Self-check

- [ ] When use MAE vs RMSE?
- [ ] What does R² = 0.7 mean?
- [ ] Why does adding random features always raise R²?
- [ ] What does adjusted R² fix?
- [ ] How do you spot heteroscedasticity in a residual plot?
- [ ] How do you diagnose overfitting from train vs test scores?
- [ ] Why is MAPE risky when y can be near 0?
- [ ] Walk through how you'd present a regression model's performance to a non-technical exec.
