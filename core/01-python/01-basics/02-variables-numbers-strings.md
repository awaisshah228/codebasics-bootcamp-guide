# Section 4 — Variables, Numbers, Strings

## Lectures covered
- Variables · Numbers · Strings · Strings exercise · Quiz · "Peter shares his coding fear with Tony" · Exercise · Chapter Summary

---

## 1. Variables

### What's actually happening
A variable is a *name bound to an object* — not a box that holds a value. This is why two variables can point to the same object:

```python
a = [1, 2, 3]
b = a            # b doesn't copy — it points to the same list
b.append(4)
print(a)         # [1, 2, 3, 4]   ← surprising if you think of variables as boxes
```

### Naming rules + conventions
- Allowed: letters, digits, underscores; can't start with digit
- Convention: `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants
- Avoid single-letter names except for trivial loop counters (`i`, `j`)

### Types are dynamic but checked at runtime
```python
x = 42        # x is int
x = "hello"   # x is now str — Python doesn't complain
```

You'll later use type hints (`x: int = 42`) — they don't change runtime behavior, but help editors and `mypy`.

---

## 2. Numbers

### Three numeric types
```python
i = 42           # int  — arbitrary precision in Python
f = 3.14         # float — 64-bit IEEE 754
c = 2 + 3j       # complex (rare in DS)
```

### Operators
```python
3 + 4    # 7
10 / 3   # 3.333...   (true division — always float)
10 // 3  # 3          (floor division)
10 % 3   # 1          (modulo)
2 ** 10  # 1024       (exponent)
```

### Floating-point gotcha
```python
0.1 + 0.2          # 0.30000000000000004
0.1 + 0.2 == 0.3   # False
```
Use `math.isclose(0.1 + 0.2, 0.3)` or `decimal.Decimal` when exact decimal arithmetic matters (money, billing).

---

## 3. Strings

### Creation
```python
s = 'single'
s = "double"
s = """triple-quoted
spans multiple lines"""
```

### Indexing & slicing
```python
s = "Codebasics"
s[0]      # 'C'
s[-1]     # 's'
s[0:4]    # 'Code'
s[::-1]   # 'scisabedoC'  (reverse)
```

### Common methods
```python
s.lower()                       # case
s.upper()
s.strip()                       # remove leading/trailing whitespace
s.replace("Code", "Data")
s.split(",")                    # → list
",".join(["a", "b", "c"])       # → "a,b,c"
"Code" in s                     # True/False
s.startswith("Code")
s.find("basics")                # index or -1
```

### f-strings (use these, not `.format` or `%`)
```python
name = "Awais"
score = 87.34
print(f"{name} scored {score:.1f}")          # Awais scored 87.3
print(f"{name=}, {score=:.2f}")              # debug: name='Awais', score=87.34
```

### Strings are immutable
```python
s = "hello"
s[0] = "H"        # TypeError — strings can't be mutated
s = "Hello"       # OK — bind name to a new string
```

---

## 4. The "Peter shares his coding fear with Tony" lesson

The cinematic side-story makes a real point: **everyone feels stupid when they start**. Symptoms — "I can't remember the syntax", "everyone else is faster", "this should be obvious" — are universal. The fix is reps, not IQ.

If a syntax doesn't stick after re-watching, **type it 3× from memory** without looking. That's the only thing that makes it stick.

---

## 5. Exercises (typical Codebasics style)

### Variables
```python
# Given temperature in Celsius, print Fahrenheit
celsius = 25
# Your one line:
fahrenheit = celsius * 9/5 + 32
print(fahrenheit)   # 77.0
```

### Numbers
```python
# Compound interest: P=10000, r=0.07, t=10
P, r, t = 10000, 0.07, 10
amount = P * (1 + r) ** t
print(round(amount, 2))   # 19671.51
```

### Strings
```python
# Given a sentence, count the vowels
sentence = "Codebasics teaches data science."
count = sum(1 for ch in sentence.lower() if ch in "aeiou")
print(count)
```

---

## 6. Common pitfalls

| Bug | Why it happens | Fix |
|---|---|---|
| `0.1 + 0.2 == 0.3` is False | float precision | use `math.isclose` |
| Mutating one variable changes another | both point to same object | use `.copy()` or `list(x)` |
| `s.replace(...)` doesn't update `s` | strings immutable; method returns new | reassign: `s = s.replace(...)` |
| `print("Hello, " + name + "!")` is verbose and breaks if `name` is `None` | string concat with None | use f-strings |

## Self-check

- [ ] What's the difference between `==` and `is`?
- [ ] What does `10 // 3` return and why?
- [ ] How do I reverse a string in one line?
- [ ] Write an f-string that formats a float to 2 decimal places.
- [ ] Why does `s[0] = "H"` raise an error?
- [ ] Convert "abc,def,ghi" into a list of three strings.
