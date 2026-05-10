# Section 12 — JSON, Generators, Decorators

## Lectures covered
- Working with JSON · Generators & Iterators · Decorators · Quiz · Exercise · Chapter Summary

---

## In one sentence
**JSON** is the universal data format for APIs, **generators** stream values one-at-a-time without holding the whole list in memory, and **decorators** wrap a function to add behavior (logging, caching, retries) without touching the original code.

## Real-world analogy
JSON is the universal shipping container — every truck (language) understands it. A generator is a Pez dispenser — one candy pops out at a time, you never carry the whole pack. A decorator is gift wrap — same gift inside, but the wrapper adds a label, a ribbon, or a security tag without changing the gift.

## The intuition (plain English)
Use `json.dumps` to serialize a Python dict to a JSON string and `json.loads` to parse one back. **Generators** use `yield` — each call resumes where it left off, producing values lazily, which is how you process a 10 GB log file with 50 MB of memory. **Decorators** are functions that take a function and return a wrapped version — perfect for cross-cutting concerns like timing, retries, caching (`@lru_cache`), or auth checks. Always pair decorators with `@functools.wraps(fn)` so the wrapped function keeps its name and docstring.

## Mini worked example
A timing decorator + a generator reading a file lazily:

```python
import time, functools, json

def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"{fn.__name__}: {time.perf_counter()-t0:.3f}s")
        return result
    return wrapper

# generator: read JSON-Lines file one record at a time
def stream_records(path):
    with open(path) as f:
        for line in f:                       # `f` is itself a generator
            yield json.loads(line)

@timeit
def count_above(path, threshold):
    return sum(1 for rec in stream_records(path) if rec["amount"] > threshold)

# count_above("expenses.jsonl", 100)
# → count_above: 0.045s
```

`stream_records` never loads the whole file. `@timeit` adds timing without touching `count_above`'s logic.

## At-a-glance

```mermaid
flowchart LR
    PyDict[Python dict] -- json.dumps --> Str[JSON string]
    Str -- json.loads --> PyDict

    Func[def fn -- yield x] --> Gen[generator object]
    Gen -- next -- > Val1[value 1]
    Gen -- next -- > Val2[value 2]
    Gen -- exhausted --> Stop[StopIteration]

    Original[def slow_call] --> Wrap[decorator wraps it]
    Wrap --> New[new fn with logging,<br/>caching, retries]
```

## Why this matters
- Every API in the world speaks JSON — `json.loads` and `json.dumps` are bread-and-butter.
- Generators turn "out of memory" errors into one-line fixes for big data.
- Decorators are everywhere in modern Python (`@app.get`, `@dataclass`, `@pytest.fixture`) — knowing how they work demystifies frameworks.

---

## 1. JSON

JSON = JavaScript Object Notation. Universal data interchange format. Every API speaks it. Python's stdlib `json` module handles it.

### Mapping JSON ↔ Python
| JSON | Python |
|---|---|
| object `{}` | `dict` |
| array `[]` | `list` |
| string | `str` |
| number | `int` / `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

### Read / write strings
```python
import json

# Python → JSON string
data = {"name": "Awais", "scores": [90, 87, 92]}
s = json.dumps(data)                          # compact
s = json.dumps(data, indent=2, sort_keys=True)  # pretty

# JSON string → Python
obj = json.loads(s)
```

### Read / write files
```python
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("data.json") as f:
    data = json.load(f)
```

### Gotchas
```python
# datetime not directly JSON-serializable
import json
from datetime import datetime
json.dumps({"now": datetime.now()})   # TypeError

# fix:
json.dumps({"now": datetime.now().isoformat()})

# or custom encoder:
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dumps({"now": datetime.now()}, cls=DateEncoder)
```

### `pandas` reads/writes JSON too
```python
df = pd.read_json("data.json", orient="records")
df.to_json("out.json", orient="records", indent=2)
```

---

## 2. Generators & iterators

### The protocol — what makes something iterable
Anything that defines `__iter__` (returns an iterator) is iterable. Iterators define `__next__` (returns next value or raises `StopIteration`).

```python
nums = [1, 2, 3]
it = iter(nums)            # get iterator
next(it)                   # 1
next(it)                   # 2
next(it)                   # 3
next(it)                   # StopIteration
```

`for x in nums:` is sugar for the above.

### Generator function — `yield`
```python
def count_up(limit):
    n = 0
    while n < limit:
        yield n
        n += 1

for x in count_up(5):
    print(x)               # 0 1 2 3 4
```

Each `yield` pauses the function, returns a value, and resumes on next call. **No list is built in memory** — values are produced on demand.

### Why use them — memory
```python
# bad — builds entire list in memory
all_lines = open("huge.txt").readlines()
for line in all_lines:
    process(line)

# good — generator-based, constant memory
with open("huge.txt") as f:
    for line in f:                # f is a generator
        process(line)
```

### Generator expressions (covered in Section 11)
```python
total = sum(x*x for x in range(10**6))
```

### Real-world generator — paginated API
```python
def fetch_all_pages(url):
    page = 1
    while True:
        r = requests.get(url, params={"page": page}).json()
        if not r["items"]:
            return                    # stops generator
        for item in r["items"]:
            yield item
        page += 1

for item in fetch_all_pages("https://api.example.com/users"):
    process(item)
```

Caller doesn't care if there are 10 or 10 million items.

### `yield from` — delegate to another iterable
```python
def all_squares():
    yield from (x*x for x in range(10))
    yield from [100, 200, 300]
```

---

## 3. Decorators

### What they are
A decorator is a function that takes a function and returns a (usually wrapped) function.

### Basic shape
```python
def loud(fn):
    def wrapper(*args, **kwargs):
        print(f"calling {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"{fn.__name__} returned {result}")
        return result
    return wrapper

@loud
def add(a, b):
    return a + b

add(2, 3)
# calling add
# add returned 5
```

`@loud` is sugar for `add = loud(add)`.

### Why useful — cross-cutting concerns
- Logging
- Timing
- Caching
- Auth checks
- Retries
- Rate limiting

You add behavior **without modifying the original function**.

### Real example — timing
```python
import time, functools

def timeit(fn):
    @functools.wraps(fn)              # preserves name, docstring, etc.
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{fn.__name__}: {elapsed:.3f}s")
        return result
    return wrapper

@timeit
def expensive():
    sum(i*i for i in range(10**6))
```

### Decorator with arguments
```python
def retry(times=3):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    print(f"retry {attempt+1}: {e}")
        return wrapper
    return decorator

@retry(times=5)
def call_api():
    ...
```

### Built-in decorators worth knowing
```python
from functools import lru_cache, cached_property
from dataclasses import dataclass
import abc

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)
# memoized — fib(100) is now instant

@dataclass
class Point:
    x: float
    y: float

class Circle:
    @cached_property
    def area(self):
        return 3.14 * self.r ** 2
# computed once, cached on the instance

class Animal(abc.ABC):
    @abc.abstractmethod
    def speak(self): ...
```

### `@property` / `@staticmethod` / `@classmethod`
```python
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance              # acc.balance — looks like attr

    @staticmethod
    def is_valid(amount):                  # no self/cls — pure utility on the class
        return amount > 0

    @classmethod
    def from_csv(cls, line):               # alternate constructor
        balance = float(line.split(",")[1])
        return cls(balance)
```

---

## 4. Practice exercises

### Generator: read a large CSV in chunks
```python
def csv_chunks(path, chunksize=1000):
    import csv
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)                           # skip header
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunksize:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

for batch in csv_chunks("huge.csv", 5000):
    process_batch(batch)
```

### Decorator: log exceptions to file
```python
def log_exceptions(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            with open("errors.log", "a") as f:
                f.write(f"{datetime.now()} - {fn.__name__}: {e}\n")
            raise
    return wrapper
```

### Cache an expensive HTTP call
```python
@lru_cache(maxsize=128)
def fetch_user(user_id):
    return requests.get(f"https://api.example.com/users/{user_id}").json()
```

---

## 5. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| Using a generator twice | exhausts after first iteration | call the generator function again, or `list(gen)` |
| Forgot `@functools.wraps` | wrapped fn's `__name__` becomes `wrapper` | always use `@functools.wraps(fn)` |
| Decorator changes signature | TypeError on call | preserve `*args, **kwargs` |
| `lru_cache` on method with `self` | self gets cached, prevents GC | use `@cached_property` for methods |
| `json.dumps` on a `set` | not JSON-serializable | convert to list first |

## Self-check

- [ ] How do I read a 10GB JSON-lines file without exhausting memory?
- [ ] Difference between a list comprehension and a generator expression?
- [ ] Write a decorator that retries on `requests.HTTPError` up to 3 times.
- [ ] What does `@functools.wraps(fn)` preserve?
- [ ] When would I use `@lru_cache`?
- [ ] How does `yield from` differ from `yield`?
- [ ] Why doesn't `for x in gen: ...` work twice on the same generator object?

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **JSON** (JavaScript Object Notation) | Universal text data format used by nearly every web API |
| **Serialize** | Turn a Python object into JSON text — `json.dumps` |
| **Deserialize** | Turn JSON text back into a Python object — `json.loads` |
| **`json.dump` / `json.load`** | File-based versions (note: no `s`) |
| **`indent=`** | Pretty-print with N spaces of indentation |
| **JSON Lines (`.jsonl`)** | One JSON object per line — streamable |
| **Custom encoder** | A class that teaches `json` how to serialize a non-default type |
| **Iterable** | Anything you can loop over (`for x in obj:`) |
| **Iterator** | An object that produces values one-at-a-time and remembers position |
| **`__iter__`** | Returns an iterator |
| **`__next__`** | Returns the next value or raises `StopIteration` |
| **Generator** | A special iterator created with a function that uses `yield` |
| **`yield`** | Pause the function, produce a value, resume on next call |
| **`yield from`** | Delegate iteration to another iterable |
| **Generator expression** | Generator built with `()` — like a list comp but lazy |
| **Lazy evaluation** | Compute on demand instead of upfront |
| **Decorator** | A function that takes a function and returns a wrapped one |
| **`@decorator`** | Syntax sugar for `fn = decorator(fn)` |
| **`functools.wraps`** | Preserves the wrapped function's name, docstring, etc. |
| **`*args, **kwargs`** | Forward all positional + keyword args inside a wrapper |
| **`lru_cache`** | Decorator that caches recent return values — memoization |
| **Memoization** | Caching results so repeated calls return instantly |
| **`cached_property`** | Compute a property once, reuse the cached value on the same instance |
| **`@property`** | Makes a method look like an attribute |
| **`@staticmethod`** | A method that doesn't take `self` or `cls` |
| **`@classmethod`** | A method that takes the class as `cls` — useful for alternate constructors |
| **Cross-cutting concern** | Logic (logging, auth, retries) that applies across many functions |
| **`abc.abstractmethod`** | Marks a method that subclasses must implement |

## Further reading
- Comprehensions (related): [01-comprehensions-sets.md](01-comprehensions-sets.md)
- APIs that produce/consume JSON: [03-apis-fastapi.md](03-apis-fastapi.md)
- Decorator-heavy frameworks: [04-logging-pytest-pydantic-mysql.md](04-logging-pytest-pydantic-mysql.md)
