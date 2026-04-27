# Section 11 — Comprehensions, Sets, PEP 8, Debugging

## Lectures covered
- Set & Frozenset · List/Dict/Set Comprehensions · Quiz · "Peter gets ad-hoc tasks" · Exercise · PEP8 Naming Convention · Code Debugging Using PyCharm · Chapter Summary

---

## 1. Sets

### What they are
Unordered, **unique** collections. Hash-based, so membership-check is **O(1)**.

```python
s = {1, 2, 3, 2, 1}        # {1, 2, 3}  — duplicates removed
s = set([1, 2, 2, 3])      # same
empty = set()              # NOT {} — that's an empty dict
```

### Operations
```python
a = {1, 2, 3}
b = {3, 4, 5}

a | b              # union           {1,2,3,4,5}
a & b              # intersection    {3}
a - b              # difference      {1,2}
a ^ b              # symmetric diff  {1,2,4,5}

a.add(6)
a.discard(2)       # no error if missing
a.remove(2)        # KeyError if missing
2 in a             # O(1) membership
```

### When to use a set vs a list
| Use a set when | Use a list when |
|---|---|
| Need uniqueness | Order matters |
| Need fast membership tests | Need indexing |
| Doing set algebra (union, diff) | Need duplicates |

### Frozenset — immutable set
```python
fs = frozenset([1, 2, 3])
# can be used as a dict key or set element
```

---

## 2. Comprehensions

The single most-used Pythonic construction. Replaces `for` loops + `append`.

### List comprehension
```python
# imperative
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x * x)

# comprehension
squares = [x * x for x in range(10) if x % 2 == 0]
```

Pattern: `[expr for var in iterable if cond]`

### Dict comprehension
```python
{x: x*x for x in range(5)}
# {0:0, 1:1, 2:4, 3:9, 4:16}

word_lens = {w: len(w) for w in ["hi", "hello", "world"]}

# invert a dict
{v: k for k, v in original.items()}
```

### Set comprehension
```python
{w.lower() for w in words}        # unique lowercased words
```

### Nested comprehensions (use sparingly)
```python
matrix = [[1,2,3], [4,5,6], [7,8,9]]
flat = [x for row in matrix for x in row]            # [1,2,...,9]
transposed = [[row[i] for row in matrix] for i in range(3)]
```

> Three+ nested for clauses: stop, write a function or use a loop. Readability > cleverness.

### Generator expression — same syntax, different brackets
```python
gen = (x * x for x in range(1_000_000))   # () instead of []
sum(gen)                                   # 333332833333500000

# memory-efficient — never builds the full list
```

Use generator expressions inside `sum()`, `max()`, `any()`, `all()`, `min()`.

---

## 3. PEP 8 — Python's style guide

PEP 8 is the standard. Following it makes your code readable to every Python developer.

### Names
```python
snake_case_for_variables = 1
snake_case_for_functions = lambda: None
PascalCase = "for classes"
UPPER_SNAKE = "for module-level constants"
_leading_underscore = "for internal-use names"
__double_leading = "name-mangled in classes"
__dunder__ = "reserved for Python's own protocols"
```

### Whitespace
- 4 spaces per indentation level (not tabs)
- 2 blank lines between top-level definitions
- 1 blank line between methods in a class
- Spaces around binary operators: `a = 1 + 2`, not `a=1+2`
- No spaces inside parens: `f(a, b)`, not `f( a, b )`

### Line length
- Soft limit: 79 chars; modern projects use 88 (Black) or 100
- Long lines: break around operators, indent for clarity

### Imports
```python
# stdlib first, blank line, third-party, blank line, local
import os
import sys

import numpy as np
import pandas as pd

from . import utils
```

### Tooling — let the machine handle it
- `black` — opinionated auto-formatter, the modern default
- `ruff` — fast linter (replaces flake8, pylint, pyflakes, isort)
- Editor: enable "Format on Save"

```bash
pip install black ruff
black .                  # reformat all .py
ruff check .             # lint
ruff check --fix .       # auto-fix lints
```

---

## 4. Code debugging in PyCharm / VS Code

### Print-debugging is fine for small bugs
```python
print(f"{x=}, {len(items)=}")          # f-string with `=` shows name + value
```

### Real debugger — set a breakpoint, step through

In **PyCharm**:
1. Click the gutter next to a line → red dot
2. Run → Debug (instead of Run)
3. Use F8 (step over), F7 (step into), F9 (resume)
4. Watch panel for variables

In **VS Code**:
1. F9 to toggle breakpoint
2. F5 to start debugger
3. Same step controls

### `pdb` — built-in debugger
```python
breakpoint()                # drops into interactive debugger
```

Commands: `n` (next), `s` (step into), `c` (continue), `p var` (print), `q` (quit).

### Why a debugger beats print
- Inspect *any* variable, not just the ones you `print`
- Modify state mid-run to test hypotheses
- Walk backwards through frames (`u`, `d`)
- Watch values change without restarting the program

---

## 5. Practice — typical exercises

### Word frequency
```python
text = "the quick brown fox jumps over the lazy dog the fox"
freq = {}
for w in text.split():
    freq[w] = freq.get(w, 0) + 1

# Pythonic
from collections import Counter
freq = Counter(text.split())
freq.most_common(3)              # [('the', 3), ('fox', 2), ...]
```

### De-duplicate while preserving order
```python
seen = set()
result = []
for x in items:
    if x not in seen:
        seen.add(x); result.append(x)

# Pythonic (3.7+: dict preserves insertion order)
result = list(dict.fromkeys(items))
```

### Pivot list of dicts to dict of lists
```python
records = [
    {"name": "A", "age": 30},
    {"name": "B", "age": 25},
]
pivoted = {k: [d[k] for d in records] for k in records[0]}
# {'name': ['A','B'], 'age': [30,25]}
```

---

## 6. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| `{}` thinking it's an empty set | actually empty dict | `set()` |
| Comprehension with side effects | `[print(x) for x in ...]` | use a normal `for` loop |
| Nested comp unreadable | too clever | rewrite as `for` loop or function |
| Imports out of order | hard to scan | run `ruff check --fix` or `isort` |

## Self-check

- [ ] How do I create an empty set?
- [ ] Convert `[1, 2, 2, 3, 3, 3]` to a set in one line.
- [ ] Write a dict comprehension that maps each word to its length.
- [ ] What's the difference between a list comp and a generator expression?
- [ ] Name three PEP 8 conventions.
- [ ] Set a breakpoint with `breakpoint()` and explain `n`, `c`, `p`.
- [ ] Convert a list of dicts to a dict of lists in one comprehension.
