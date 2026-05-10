# Foundations 2 — Linear Regression (Simple & Multiple)

## Lectures covered
- Simple Linear Regression
- Multiple Linear Regression

---

## In one sentence
**Linear regression** draws the straight line (or flat plane, in higher dimensions) that comes closest to your data points, then uses that line to predict a number for new inputs.

## Real-world analogy
Imagine you're plotting houses on graph paper: x-axis is square footage, y-axis is sale price. The dots roughly form a rising trend — bigger houses cost more. Linear regression places a ruler through those dots so the total gap between the line and each dot is as small as possible. To predict a new house's price, you look up its sqft on the line.

## The intuition (plain English)
1. The model is a **straight line**: `price = intercept + slope × sqft`.
2. We pick the line that minimizes total **squared gap** between predicted and actual prices (squared so over- and under-predictions both count, and big misses are punished more).
3. With multiple inputs (sqft, bedrooms, age), the line becomes a **flat surface** through higher-dimensional space — but the math is the same.
4. The slopes (coefficients) tell you how much price changes per unit of each input — extremely useful for explanation.

## Mini worked example — fitting a line to 4 houses

```
sqft (x):    1000   1500   2000   2500
price (y):   200k   260k   330k   410k
```

Find the line `price = β₀ + β₁ · sqft`:

```
mean_x = (1000+1500+2000+2500)/4 = 1750
mean_y = (200+260+330+410)/4    = 300

β₁ = Σ(xᵢ−mean_x)(yᵢ−mean_y) / Σ(xᵢ−mean_x)²
   = [(-750)(-100) + (-250)(-40) + (250)(30) + (750)(110)] / [750² + 250² + 250² + 750²]
   = [75000 + 10000 + 7500 + 82500] / [562500 + 62500 + 62500 + 562500]
   = 175000 / 1250000
   = 0.14   (i.e. $140 per sqft)

β₀ = mean_y − β₁ · mean_x = 300 − 0.14 · 1750 = 55  (i.e. $55,000)
```

So `price ≈ 55,000 + 140 × sqft`. A new 1,800-sqft house: predicted price = $55,000 + $140 × 1,800 = **$307,000**.

## At-a-glance — the geometry

```
price │              ●     <- gap (residual) between dot and line
      │         ●  /
      │      ─/●─/         <- best-fit line
      │    /● 
      │   /  ●
      │  /●
      │ /
      └────────────► sqft

The line minimizes Σ(gap)²   ←  "ordinary least squares"
```

## Why this matters
- **The base model in every regression problem** — start here, then upgrade only if it underperforms.
- **Interpretable coefficients**: "+$140 per sqft, +$5k per bedroom" — directly explainable to stakeholders.
- **Foundation for Ridge, Lasso, logistic regression, and even neural networks** — they all share the same `wᵀx + b` skeleton.
- **Healthcare premium project** uses linear regression as the first baseline before XGBoost.

---

## 1. Simple Linear Regression — predicting y from a single x

The model:
$$y = \beta_0 + \beta_1 x + \varepsilon$$

- $\beta_0$ — intercept (y when x = 0)
- $\beta_1$ — slope (change in y per unit x)
- $\varepsilon$ — error / noise

Goal: find $\beta_0, \beta_1$ that minimize the sum of squared errors.

### Visual
```
y
│         •
│       •     <- residual (gap between point and line)
│    ───•─────  <- best-fit line
│   •
│ •
└────────────► x
```

### Closed-form solution (Ordinary Least Squares)
$$\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}$$

You can derive this by setting the gradient of MSE to zero (covered in next file).

### sklearn shape
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)                      # X must be 2D — even if 1 feature
print(model.intercept_, model.coef_)
y_pred = model.predict(X_new)
```

> If X is 1D (e.g., shape `(n,)`), reshape to `X.reshape(-1, 1)` before fitting.

---

## 2. Multiple Linear Regression — multiple features

The model:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \varepsilon$$

In matrix form:
$$y = X\beta + \varepsilon$$

OLS solution:
$$\beta = (X^TX)^{-1}X^Ty$$

(Assuming $X^TX$ is invertible — fails when features are perfectly collinear.)

### sklearn — same code, just multiple columns
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing.csv")
X = df[["sqft", "bedrooms", "age"]]
y = df["price"]

model = LinearRegression().fit(X, y)
model.coef_         # one coefficient per feature
model.intercept_
```

---

## 3. Interpreting coefficients

For a fitted model $\hat{y} = \beta_0 + \beta_1 \cdot \text{sqft} + \beta_2 \cdot \text{bedrooms}$:
- $\beta_1 = 150$: each extra sqft adds $150 to predicted price, **holding bedrooms constant**
- $\beta_2 = 5000$: each extra bedroom adds $5000, **holding sqft constant**

### Why "holding constant" matters
If `bedrooms` is correlated with `sqft` (it is — bigger houses have more bedrooms), the simple coefficient is *adjusted* to factor out the shared variation. This is the magic of multiple regression — and the source of confusing interpretations.

### Standardized coefficients (to compare on equal footing)
Coefficients depend on feature scales. To compare *importances*, standardize features (mean 0, std 1) first; then |coefficients| are comparable.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scale", StandardScaler()),
                 ("lr", LinearRegression())])
pipe.fit(X, y)
pipe.named_steps["lr"].coef_
```

---

## 4. Assumptions of linear regression (and why they matter)

1. **Linearity** — y is roughly a linear function of x's
2. **Independence** — observations are not correlated with each other
3. **Homoscedasticity** — residual variance is constant across x
4. **Normality of residuals** — residuals are approximately Normal (mainly for inference / CIs)
5. **No multicollinearity** — features aren't too correlated with each other

Codebasics covers **VIF** in the unsupervised/utilities subfolder for diagnosing multicollinearity.

### How violations look
- Curved residuals plot → linearity broken; try polynomial or non-linear model
- Funnel-shaped residuals → heteroscedasticity; consider log-transform or weighted regression
- Heavy-tailed residual histogram → non-Normal; mostly cosmetic for prediction
- VIF > 5 (or 10) → multicollinearity; drop or combine features

### Diagnostic plot
```python
import numpy as np, matplotlib.pyplot as plt
y_pred = model.predict(X)
residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, color="red")
plt.xlabel("predicted"); plt.ylabel("residual")
```

A clean residual plot looks like a *random* horizontal cloud around zero. Patterns = problems.

---

## 5. Polynomial regression — when linear isn't enough

If the relationship is curved, add polynomial features:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \dots$$

Still a linear model in the *coefficients*, just non-linear in the input.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("lr",   LinearRegression()),
])
poly_pipe.fit(X, y)
```

> Polynomial degrees > 3 often overfit. Watch the validation curve.

---

## 6. Categorical features — one-hot encoding

Linear regression can't consume strings ("red", "blue"). Convert via one-hot encoding (covered fully in `05-preprocessing-encoding.md`).

```python
df_encoded = pd.get_dummies(df, columns=["color"], drop_first=True)
```

`drop_first=True` avoids the **dummy variable trap** (perfect multicollinearity).

---

## 7. Working example — house price (one-shot)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("housing.csv")
df = pd.get_dummies(df, columns=["zone"], drop_first=True)

X = df.drop(columns=["price"])
y = df["price"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_tr, y_tr)
y_pred = model.predict(X_te)

print("RMSE:", np.sqrt(mean_squared_error(y_te, y_pred)))
print("R²:",   r2_score(y_te, y_pred))

# coefficient inspection
coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(coefs.head(10))
```

This template is reusable for *any* tabular regression problem.

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to scale features when comparing coefficients | "income" coef looks tiny, "rooms" looks huge — falsely | standardize first |
| Strong correlation between features | unstable / nonsense coefficients | check VIF; drop or combine |
| Including future features (target leakage) | unrealistically high R² | trace timing of every feature |
| Forgetting `drop_first=True` | dummy variable trap, singular X^TX | always drop one category |
| Using polynomial degree 10 | massive overfitting | use cross-validation to pick degree |

## Self-check

- [ ] Write the formula for simple linear regression.
- [ ] Write the OLS closed-form solution for multiple regression.
- [ ] Why does "holding others constant" matter for interpreting coefficients?
- [ ] What's heteroscedasticity and how do you detect it?
- [ ] When is polynomial regression appropriate? When is it dangerous?
- [ ] Why do we use `drop_first=True` in one-hot encoding?
- [ ] List 5 assumptions of linear regression.
- [ ] Compute the formula for $\beta_1$ in simple linear regression by hand on a 4-point toy example.

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Linear regression** | Fitting a straight line (or flat plane) that minimizes total squared gap to the data |
| **Intercept (β₀)** | The model's prediction when all features are zero — where the line crosses the y-axis |
| **Slope / coefficient (β_j)** | How much y changes for a 1-unit increase in feature x_j, with other features held constant |
| **Residual** | One row's gap: `actual − predicted` — what the line missed by |
| **OLS (Ordinary Least Squares)** | The math trick of choosing β to minimize the sum of squared residuals — gives a closed-form solution |
| **Closed-form solution** | An exact formula (no iteration needed) — for OLS it's `β = (XᵀX)⁻¹Xᵀy` |
| **MSE (Mean Squared Error)** | Average of residuals squared — the standard regression loss |
| **R² (R-squared)** | "Fraction of y's variance the model explains" — 1 = perfect, 0 = no better than guessing the mean |
| **Multiple regression** | Linear regression with more than one input feature |
| **Polynomial regression** | Linear regression on `x, x², x³, …` — fits curves while still using linear math |
| **One-hot encoding** | Turning a string column ("red"/"blue") into 0/1 columns, one per category |
| **Dummy variable trap** | Keeping all one-hot columns creates a perfect linear relation — breaks regression. Drop one. |
| **Multicollinearity** | Two or more features carry the same info — coefficients become unstable |
| **VIF (Variance Inflation Factor)** | Number that flags multicollinearity for each feature; >5 is concerning |
| **Standardization** | Subtract mean, divide by std dev — puts every feature on the same scale |
| **Homoscedasticity** | Residual spread is constant across the predicted range (the "good" pattern) |
| **Heteroscedasticity** | Residual spread changes with the prediction — funnel shape on residual plot |
| **Residual plot** | Scatter of (predicted, residual) — should look like a random horizontal cloud |
| **Linearity assumption** | y is roughly a linear function of the features (one of OLS's main assumptions) |
| **Normality of residuals** | Residuals are approximately bell-shaped — needed for confidence intervals on coefficients |
| **Independence** | One row's outcome doesn't depend on another's |
| **Feature scaling** | Bringing features to similar magnitudes so coefficients are comparable |
| **`coef_` / `intercept_`** | sklearn attributes that hold the learned β₁..β_p and β₀ |
| **`reshape(-1, 1)`** | NumPy idiom to convert a 1D array of shape (n,) into a 2D column (n, 1) for sklearn |
| **Standardized coefficient** | The β computed after scaling features — comparable across features regardless of original units |

## Further reading
- Previous: [01-intro-and-categories.md](01-intro-and-categories.md)
- Next: [03-gradient-descent-cost.md](03-gradient-descent-cost.md) — *how* the line is actually found numerically
- Then: [04-model-evaluation-regression.md](04-model-evaluation-regression.md) — judging how good the line is
- Multicollinearity deep-dive: [../04-unsupervised/02-vif.md](../04-unsupervised/02-vif.md)
- Math foundation: [../../05-math-statistics/01-foundations/02-central-tendency-dispersion.md](../../05-math-statistics/01-foundations/02-central-tendency-dispersion.md) — mean, variance, correlation underlie regression
