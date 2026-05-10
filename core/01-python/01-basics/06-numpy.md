# Section 8 — NumPy

## Lectures covered
- Introduction & Benefits · Basic Operations · Matrix Operations · Slicing & Stacking · Quiz · Real-time problem · Exercise · Chapter Summary

---

## In one sentence
**NumPy** gives Python a fast, fixed-type, multi-dimensional array (`ndarray`) and lets you do math on whole arrays at once, which is 50–100× faster than Python for-loops.

## Real-world analogy
A Python list is like a row of mismatched lockers — each one a different size and content. A NumPy array is a grid of identical mailboxes glued together: same size, same type, packed tight. That is why `array * 2` doubles every value instantly, while a Python list would need a manual loop.

## The intuition (plain English)
Arrays are stored in contiguous memory with a fixed `dtype` (e.g., `float64`) — that is why they are fast. **Vectorization** means writing `a + b` instead of `for i: c[i] = a[i] + b[i]`. **Broadcasting** lets arrays of different but compatible shapes interact (e.g., adding a row vector to a matrix). `axis=0` collapses rows (one number per column); `axis=1` collapses columns (one number per row). NumPy is the foundation under pandas, scikit-learn, PyTorch, and TensorFlow.

## Mini worked example
Compute average exam score per student in 3 lines:

```python
import numpy as np

scores = np.array([
    [78, 82, 91, 67],     # student 0
    [85, 73, 88, 90],     # student 1
    [62, 75, 80, 70],     # student 2
])

per_student = scores.mean(axis=1)        # collapse columns → one avg per row
print(per_student)                       # [79.5  84.   71.75]

print(scores[per_student > 80])          # boolean mask → rows for top performers
# [[85 73 88 90]]
```

No `for` loop, no `+= 1`, just `.mean(axis=1)` and a boolean filter.

## At-a-glance

```
shape (3, 4)
                axis=1 (collapses cols → one num per row)
                ─────────────────────►
              ┌─────────────────────┐
   axis=0     │ 78  82  91  67      │
   (collapses │ 85  73  88  90      │
    rows →    │ 62  75  80  70      │
    one num   └─────────────────────┘
    per col)

   .sum(axis=0) → [225, 230, 259, 227]   (4 numbers — one per col)
   .sum(axis=1) → [318, 336, 287]        (3 numbers — one per row)
```

## Why this matters
- Every later module (pandas, scikit-learn, PyTorch) sits on NumPy — this is the substrate.
- Vectorized code is shorter, faster, and less buggy than manual loops.
- Knowing axes and broadcasting is the difference between fluent and frustrated DS work.

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

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **NumPy** | The numerical computing library that gives Python fast arrays |
| **`ndarray`** | NumPy's n-dimensional array type |
| **`shape`** | Tuple of sizes per dimension — `(3, 4)` is 3 rows × 4 columns |
| **`ndim`** | Number of dimensions — 1 for vector, 2 for matrix, 3+ for tensor |
| **`size`** | Total number of elements |
| **`dtype`** | The element type — `int64`, `float64`, `bool`, etc. |
| **Vectorization** | Doing array math without a Python loop |
| **Broadcasting** | Operations on arrays of different but compatible shapes — smaller dims stretch to fit |
| **`axis=0`** | Collapses rows — result has one number per column |
| **`axis=1`** | Collapses columns — result has one number per row |
| **Element-wise** | Operation done position-by-position (`a * b` multiplies pairs) |
| **Matrix multiplication** | Linear-algebra multiply with `@` or `np.dot` |
| **`@` operator** | Matrix multiply (Python 3.5+) |
| **Hadamard product** | Element-wise multiply — what `*` does on arrays |
| **Transpose (`.T`)** | Swap rows and columns |
| **Identity matrix** | Square matrix with 1s on diagonal, 0s elsewhere — `np.eye(n)` |
| **Determinant** | A scalar describing how a matrix scales space |
| **Inverse** | The matrix that undoes another — `A @ A_inv == I` |
| **Eigenvalue / eigenvector** | Special direction unchanged by a matrix multiply, scaled by the eigenvalue |
| **Slicing** | Selecting a sub-array with ranges — `A[0:2, 1:3]` |
| **Boolean indexing** | Selecting with a mask: `a[a > 4]` |
| **Fancy indexing** | Selecting with arrays of indices: `a[[0, 2, 4]]` |
| **View vs copy** | A slice usually points at the same memory — modify it and the original changes |
| **`reshape`** | Change shape without changing data; `-1` means "infer this dim" |
| **`flatten`** | Make a copy of the array as 1D |
| **`concatenate / vstack / hstack`** | Stack arrays end-to-end / row-wise / column-wise |
| **`linspace`** | Equally spaced values between two endpoints |
| **`arange`** | Like Python's `range` but produces an array |

## Further reading
- Next: [07-eda-pandas-matplotlib-seaborn.md](07-eda-pandas-matplotlib-seaborn.md)
- Math foundations module: [../../05-math-statistics/01-foundations/README.md](../../05-math-statistics/01-foundations/README.md)
- Why DL frameworks build on NumPy: [../../07-deep-learning/architectures-and-math.md](../../07-deep-learning/architectures-and-math.md)
