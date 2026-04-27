# Section 7 — Classes, OOP, Exception Handling

## Lectures covered
- Classes and Objects · Operator Overloading · Inheritance · Exception Handling · `__main__` Function · Quiz · Real-time problem · Exercise · Chapter Summary

---

## 1. Classes

### Why bother — when do I use a class?
- I have **state + behavior bound together** (a "thing" with attributes + methods)
- I'm representing a **domain entity** (Customer, Order, Model, Pipeline)
- I want **multiple instances** of the same type
- I want to use Python's protocols (`__len__`, `__iter__`, etc.) on my objects

When you don't need any of those: just use a function or a `dict`/`dataclass`.

### The basic shape
```python
class Customer:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        return f"Hello, {self.name}!"

c = Customer("Awais", 27)
print(c.greet())             # Hello, Awais!
print(c.name)                # Awais
```

### `self` is "this instance"
Methods receive the instance as their first argument. Python doesn't pass it implicitly like Java's `this` — you must declare it.

### Class vs instance attributes
```python
class Counter:
    count = 0                 # class attribute (shared across instances)

    def __init__(self):
        self.value = 0        # instance attribute

c1 = Counter()
c2 = Counter()
Counter.count = 5            # changes shared
c1.value = 10                # changes only c1
```

### `dataclasses` (modern alternative for "data holder" classes)
```python
from dataclasses import dataclass

@dataclass
class Customer:
    name: str
    age: int
    role: str = "learner"

c = Customer("Awais", 27)        # auto __init__
print(c)                          # auto __repr__: Customer(name='Awais', age=27, role='learner')
```

90% of "data" classes you write should be `@dataclass`.

---

## 2. Operator overloading (dunder methods)

Python's "magic methods" let your class respond to operators / built-ins.

```python
class Money:
    def __init__(self, amount: float, currency: str = "USD"):
        self.amount = amount
        self.currency = currency

    def __repr__(self):
        return f"Money({self.amount} {self.currency})"

    def __add__(self, other):                # enables  m1 + m2
        if self.currency != other.currency:
            raise ValueError("currency mismatch")
        return Money(self.amount + other.amount, self.currency)

    def __eq__(self, other):                 # enables  m1 == m2
        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other):                 # enables  m1 < m2  (and sorting)
        return self.amount < other.amount

m1 = Money(10)
m2 = Money(20)
print(m1 + m2)                # Money(30 USD)
print(sorted([m2, m1]))       # uses __lt__
```

### Common dunders
| Method | Enables |
|---|---|
| `__init__` | constructor |
| `__repr__` | `repr(x)` and prints in REPL |
| `__str__` | `str(x)` and `print(x)` |
| `__len__` | `len(x)` |
| `__iter__` | `for x in obj:` |
| `__eq__`, `__lt__`, ... | comparisons + sorting |
| `__add__`, `__sub__`, ... | arithmetic |
| `__getitem__`, `__setitem__` | `obj[k]` |
| `__call__` | `obj()` — instance callable like a function |
| `__enter__`, `__exit__` | `with obj:` (context manager) |

---

## 3. Inheritance

### Single inheritance
```python
class Animal:
    def __init__(self, name): self.name = name
    def speak(self):          return "..."

class Dog(Animal):
    def speak(self):          return "Woof!"

class Cat(Animal):
    def speak(self):          return "Meow"

for a in [Dog("Rex"), Cat("Mia")]:
    print(a.name, a.speak())
```

### `super()` — call the parent's version
```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

class Manager(Employee):
    def __init__(self, name, salary, reports):
        super().__init__(name, salary)        # initialize parent
        self.reports = reports
```

### When NOT to inherit
Inheritance is overused. Prefer **composition** ("has-a") over **inheritance** ("is-a"). E.g., a `Car` *has* an `Engine` rather than inheriting from `Engine`.

---

## 4. Exception handling

### The basic shape
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"divide by zero: {e}")
except (ValueError, TypeError) as e:
    print(f"bad input: {e}")
else:
    print("no exception happened")
finally:
    print("always runs (cleanup)")
```

### Common exceptions you'll see
| Exception | When |
|---|---|
| `ValueError` | right type, wrong value (`int("hello")`) |
| `TypeError` | wrong type (`"a" + 1`) |
| `KeyError` | dict key missing |
| `IndexError` | list index out of range |
| `FileNotFoundError` | self-explanatory |
| `ZeroDivisionError` | self-explanatory |
| `AttributeError` | `obj.method` doesn't exist |
| `ImportError` / `ModuleNotFoundError` | import failed |

### Raising your own
```python
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError(f"insufficient funds: {balance} < {amount}")
    return balance - amount
```

### Custom exception classes
```python
class InsufficientFundsError(Exception):
    pass

raise InsufficientFundsError("not enough money")
```

### Catch only what you can handle
```python
try:
    do_thing()
except Exception:                 # ← too broad; hides bugs
    pass
```
Better: catch the specific exception, log the rest.

### `try/finally` and the `with` statement
The `with` block (Section 6) is exception-safe by design — that's what makes it the right way to handle files, locks, DB connections.

```python
with open("file.txt") as f:
    do_work(f)
# file is closed even if do_work raises
```

---

## 5. The `__main__` function pattern

```python
def main():
    print("running")

if __name__ == "__main__":
    main()
```

### Why this matters
A Python file can be **run** (`python file.py`) or **imported** (`import file`). The `__name__ == "__main__"` block runs only when run directly — not on import. This is essential for reusable code:

```python
# utils.py
def double(x): return 2 * x

if __name__ == "__main__":
    print(double(5))         # only when running utils.py directly
```

```python
# in another file
from utils import double      # `print` does NOT run
```

---

## 6. Real-time exercise (Codebasics style)

### Bank account simulator
```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

    def withdraw(self, amount: float):
        if amount <= 0:
            raise ValueError("withdrawal must be positive")
        if amount > self.balance:
            raise ValueError(f"insufficient funds: {self.balance}")
        self.balance -= amount

    def __repr__(self):
        return f"BankAccount({self.owner!r}, {self.balance})"

acc = BankAccount("Awais", 1000)
acc.deposit(500)
acc.withdraw(200)
print(acc)              # BankAccount('Awais', 1300)

try:
    acc.withdraw(10_000)
except ValueError as e:
    print(f"error: {e}")
```

---

## 7. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| `self` not first parameter | forgot — Python won't auto-add | always `def m(self, ...)` |
| Accessing instance attribute as class attr | `Customer.name` instead of `c.name` | use the instance |
| Catching `Exception` and ignoring | hides real bugs | catch specific types |
| Re-raising lost the trace | `raise NewError("...")` wipes trace | use `raise NewError(...) from e` |
| Using `class A(object)` in Python 3 | not needed — all classes are `object` | just `class A:` |

## Self-check

- [ ] What's the difference between `__str__` and `__repr__`?
- [ ] When would I use `@dataclass` instead of writing `__init__` manually?
- [ ] How does `super().__init__(...)` work and when do I need it?
- [ ] Why is catching `Exception` usually wrong?
- [ ] What does `if __name__ == "__main__":` do?
- [ ] Write a class `Product` with `name`, `price`, comparable by price, printable.
