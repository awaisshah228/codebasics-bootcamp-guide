# Section 8 — NumPy

## Lectures covered
- Introduction & Benefits · Basic Operations · Matrix Operations · Slicing & Stacking · Quiz · Real-time problem · Exercise · Chapter Summary

---

## 1. Why NumPy

Two reasons:

1. **Speed**: a NumPy array operation runs in optimized C code under the hood — often **50–100×** faster than a Python list loop.
2. **Vectorization**: you express operations on entire arrays at once, no `for` loop. Cleaner, faster, less buggy.

```python
import numpy as np
a = np.array([1, 2, 3, 4])
a * 2                 # [2 4 6 8]   no loop needed
a + a                 # [2 4 6 8]
a ** 2                # [1 4 9 16]
```

NumPy is the foundation under pandas, scikit-learn, PyTorch, TensorFlow. Every later module uses it.

---

## 2. Creating arrays

```python
np.array([1, 2, 3])              # from list → 1D
np.array([[1, 2], [3, 4]])       # from nested list → 2D

np.zeros((3, 4))                 # 3×4 of 0.0
np.ones((2, 5))                  # 2×5 of 1.0
np.full((2, 3), 7)               # fill value
np.eye(4)                        # 4×4 identity
np.arange(0, 10, 2)              # [0 2 4 6 8]
np.linspace(0, 1, 5)             # [0. 0.25 0.5 0.75 1.]
np.random.rand(3, 3)             # uniform [0,1)
np.random.randn(3, 3)            # standard normal
np.random.randint(0, 100, 10)    # 10 ints in [0,100)
```

### Important array attributes
```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape       # (2, 3)
a.ndim        # 2
a.size        # 6
a.dtype       # int64
a.itemsize    # 8 (bytes per element)
```

### dtype matters
- `int64` / `int32` — integers
- `float64` / `float32` — defaults; `float32` halves memory (DL uses it)
- `bool` — booleans
- `object` — generic; slow; avoid unless needed
- `np.float16`, `np.bfloat16` — modern DL

---

## 3. Basic operations

### Element-wise (the default)
```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b        # [11 22 33]
a * b        # [10 40 90]
a / b        # [0.1 0.1 0.1]
a ** 2       # [1 4 9]
np.sqrt(a)   # [1.0 1.41... 1.73...]
np.exp(a)    # [e^1 e^2 e^3]
np.log(a)    # natural log
```

### Aggregations
```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()              # 21
a.sum(axis=0)        # [5 7 9]    (sum over rows → column-wise)
a.sum(axis=1)        # [6 15]     (sum over cols → row-wise)
a.mean()
a.std()
a.min(), a.max()
a.argmin(), a.argmax()    # index of min/max
```

> `axis=0` collapses rows (one per column). `axis=1` collapses cols (one per row). This trips everyone — drill it.

### Broadcasting (the killer feature)
```python
a = np.array([[1, 2, 3], [4, 5, 6]])     # shape (2, 3)
v = np.array([10, 20, 30])               # shape (3,)

a + v              # broadcasts v across rows → [[11,22,33],[14,25,36]]
```

Broadcasting rules:
- Align shapes from the right
- Dimensions of size 1 (or missing) stretch to match

This eliminates explicit loops in 95% of array math.

---

## 4. Matrix operations

### The 3 ways to multiply
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B             # element-wise (Hadamard)
A @ B             # matrix multiply
np.dot(A, B)      # same as A @ B
```

`@` is the modern operator (Python 3.5+). Use it.

### Transpose
```python
A.T                  # transpose
np.transpose(A)
```

### Inverse, determinant, eigenvalues
```python
np.linalg.inv(A)
np.linalg.det(A)
eigvals, eigvecs = np.linalg.eig(A)
np.linalg.solve(A, b)         # solve Ax = b
```

You'll meet these again in Math/Stats and ML modules.

---

## 5. Slicing & stacking

### Indexing (1D)
```python
a = np.array([10, 20, 30, 40, 50])
a[0]              # 10
a[-1]             # 50
a[1:4]            # [20 30 40]
a[::-1]           # reverse
```

### Indexing (2D)
```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

A[0, 1]           # 2     (row 0, col 1)
A[:, 0]           # [1 4 7]    (column 0)
A[1, :]           # [4 5 6]    (row 1)
A[0:2, 1:3]       # [[2 3], [5 6]]
```

### Boolean indexing (extremely common in DS)
```python
a = np.array([1, 5, 8, 3, 7, 2])
mask = a > 4              # [False True True False True False]
a[mask]                   # [5 8 7]
a[a > 4]                  # same in one line
a[(a > 2) & (a < 7)]      # [5 3]   parens required around each cond
```

### Fancy indexing (with arrays of indices)
```python
a[[0, 2, 4]]             # [1 8 7]
```

### Stacking arrays
```python
np.concatenate([a, b])           # 1D
np.vstack([a, b])                # vertical (rows)
np.hstack([a, b])                # horizontal (cols)
np.stack([a, b], axis=0)         # adds new dim
```

### Reshape
```python
a = np.arange(12)
a.reshape(3, 4)                  # 3×4
a.reshape(2, 2, 3)               # 2×2×3
a.reshape(-1, 4)                 # 3×4 — `-1` means "infer this dim"
a.flatten()                      # back to 1D
```

---

## 6. Real-time problem (typical Codebasics exercise)

### Compute exam statistics
```python
import numpy as np

# scores: 5 students × 4 subjects
scores = np.array([
    [78, 82, 91, 67],
    [85, 73, 88, 90],
    [62, 75, 80, 70],
    [95, 88, 92, 84],
    [55, 60, 65, 50],
])

# overall average per student
print(scores.mean(axis=1))               # one number per row

# subject-wise average
print(scores.mean(axis=0))               # one number per column

# students who scored > 80 average
overall = scores.mean(axis=1)
print(scores[overall > 80])               # boolean indexing on rows

# top scorer index
print(scores.sum(axis=1).argmax())
```

This pattern — `array.mean(axis=...)`, then mask + index — is the bread and butter of analytical work.

---

## 7. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| `axis=0` vs `axis=1` confusion | counterintuitive | "axis=0 collapses rows; result has 1 number per col" |
| `*` vs `@` for matrix mul | `*` is element-wise | use `@` for matrix multiply |
| Modifying a slice modifies the original | slices are views, not copies | use `.copy()` if you need a separate array |
| `np.array == None` | element-wise comparison | use `np.isnan(a)` or `a is None` |
| Mixing dtypes | array gets upcast to `object` | be explicit: `np.array([...], dtype=np.float32)` |

## Self-check

- [ ] What's the difference between `axis=0` and `axis=1` in `array.sum()`?
- [ ] How does broadcasting work? Give one example.
- [ ] `*` vs `@` — which is element-wise, which is matrix multiply?
- [ ] How do I select rows where column 0 > 5?
- [ ] What does `arr.reshape(-1, 4)` do?
- [ ] Difference between `np.zeros(5)` and `np.zeros((5,))`?
- [ ] How do I generate 100 normally-distributed random numbers?
