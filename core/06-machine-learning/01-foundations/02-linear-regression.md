# Foundations 2 — Linear Regression (Simple & Multiple)

## Lectures covered
- Simple Linear Regression
- Multiple Linear Regression

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
