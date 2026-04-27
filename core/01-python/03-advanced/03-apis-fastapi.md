# Section 13 — APIs (consuming + building with FastAPI)

## Lectures covered
- What is API? · Calling APIs with `requests` · Building APIs with FastAPI · Quiz · "Peter fetches data using APIs" · Exercise · Chapter Summary

---

## 1. What is an API

**API** = Application Programming Interface. A contract that lets one program talk to another.

**Web API / REST API** = a server that exposes URLs (endpoints), each accepting an HTTP method (GET, POST, PUT, DELETE) and returning data (usually JSON).

```
Client (browser, Python script, mobile app)
    │
    │   HTTP request
    ▼
URL: https://api.example.com/users/42
Method: GET
Headers: Authorization: Bearer ...
Body: (empty for GET)
    │
    ▼
Server processes
    │
    │   HTTP response
    ▼
Status: 200 OK
Body: {"id": 42, "name": "Awais"}
```

### Verbs — what they should mean
| Verb | Use for |
|---|---|
| `GET` | Read (no side effects) |
| `POST` | Create |
| `PUT` | Replace whole resource |
| `PATCH` | Update part of a resource |
| `DELETE` | Delete |

### Status codes — categories
- **2xx** — success (`200 OK`, `201 Created`, `204 No Content`)
- **3xx** — redirect
- **4xx** — *client* error (`400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `429 Too Many Requests`)
- **5xx** — *server* error (`500 Internal`, `503 Unavailable`)

---

## 2. Consuming APIs — `requests`

### GET
```python
import requests

r = requests.get("https://api.github.com/users/codebasics")
r.status_code              # 200
r.headers                  # dict of headers
r.text                     # raw text
data = r.json()            # parsed JSON
print(data["public_repos"])
```

### Query parameters
```python
r = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 2},
    timeout=10,                                 # always set a timeout
)
```

### POST with JSON body
```python
r = requests.post(
    "https://api.example.com/expenses",
    json={"amount": 100, "category": "Food"},   # auto-encodes JSON + sets header
    headers={"Authorization": f"Bearer {token}"},
    timeout=10,
)
r.raise_for_status()                            # raises on 4xx/5xx
```

### File upload
```python
with open("photo.jpg", "rb") as f:
    r = requests.post("https://api.example.com/upload", files={"file": f})
```

### Error handling
```python
try:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
except requests.Timeout:
    print("server too slow")
except requests.HTTPError as e:
    print(f"bad status: {e.response.status_code}")
except requests.RequestException as e:
    print(f"network failure: {e}")
```

### Sessions — reuse connection + auth across calls
```python
s = requests.Session()
s.headers["Authorization"] = f"Bearer {token}"
r1 = s.get(url1)
r2 = s.get(url2)               # reuses TCP connection — faster
```

### Authentication shapes you'll see
- **API key** — header: `X-API-Key: ...` or query param `?api_key=...`
- **Bearer token (JWT/OAuth)** — `Authorization: Bearer <token>`
- **Basic auth** — `requests.get(url, auth=("user", "pass"))`

---

## 3. Building APIs — FastAPI

FastAPI is a modern Python web framework: fast, type-hint-driven, auto-generates OpenAPI docs.

### Install
```bash
pip install fastapi uvicorn
```

### Hello world
```python
# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "hello"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```

Run:
```bash
uvicorn app:app --reload
```

Visit: `http://localhost:8000/docs` — auto-generated Swagger UI.

### Path vs query params
```python
@app.get("/items/{item_id}")
def read(item_id: int, detailed: bool = False):
    # item_id is from URL path
    # detailed is from ?detailed=true
    ...
```

### Request bodies — Pydantic models
```python
from pydantic import BaseModel

class Expense(BaseModel):
    amount: float
    category: str
    notes: str | None = None

@app.post("/expenses")
def create(expense: Expense):
    # expense is automatically validated; bad input returns 422
    return {"received": expense, "id": 1}
```

### Response models (filter what's returned)
```python
class ExpenseOut(BaseModel):
    id: int
    amount: float
    category: str

@app.get("/expenses/{id}", response_model=ExpenseOut)
def read(id: int):
    return {"id": id, "amount": 100, "category": "Food", "secret": "ignored"}
```

The `secret` field is dropped because it's not in `ExpenseOut`.

### HTTPException — return errors cleanly
```python
from fastapi import HTTPException

@app.get("/items/{id}")
def read(id: int):
    item = db.get(id)
    if not item:
        raise HTTPException(status_code=404, detail="not found")
    return item
```

### Dependency injection
```python
from fastapi import Depends

def get_db():
    db = connect()
    try: yield db
    finally: db.close()

@app.get("/items")
def list_items(db = Depends(get_db)):
    return db.fetch_all()
```

### Async endpoints
```python
@app.get("/slow")
async def slow():
    await asyncio.sleep(1)
    return {"done": True}
```

Use `async def` when the function does I/O (HTTP calls, DB) — FastAPI runs sync functions in a thread pool, so both work, but async scales further.

---

## 4. Real-time exercise — wrap a public API

### Wrap GitHub's user endpoint
```python
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

@app.get("/gh/{username}")
def github_summary(username: str):
    r = requests.get(f"https://api.github.com/users/{username}", timeout=10)
    if r.status_code == 404:
        raise HTTPException(404, f"user {username} not found")
    r.raise_for_status()
    d = r.json()
    return {
        "username": d["login"],
        "name": d["name"],
        "bio": d["bio"],
        "public_repos": d["public_repos"],
        "followers": d["followers"],
    }
```

Now `GET /gh/codebasics` returns a clean summary.

---

## 5. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| `r.json()` raises `JSONDecodeError` | response wasn't JSON | check `r.status_code` and `r.headers["content-type"]` first |
| Hardcoded production URL | environment-specific | use env vars: `os.getenv("API_URL")` |
| No timeout on `requests` | hangs forever on slow server | always `timeout=10` |
| Returning sensitive data | leaking via response | use response_model to whitelist fields |
| API key in code | committed to git | use `.env` + `python-dotenv` + `.gitignore` |
| FastAPI returns 422 on POST | request body mismatch with Pydantic | check Swagger / look at the validation error |

## Self-check

- [ ] Difference between path param, query param, and request body?
- [ ] What status code does FastAPI return for failing Pydantic validation?
- [ ] Why should I always pass `timeout=` to `requests`?
- [ ] When use `async def` vs `def` in FastAPI?
- [ ] How do I auto-generate API docs in FastAPI?
- [ ] Write a FastAPI endpoint `POST /users` accepting `{name, email}` validated by Pydantic.
- [ ] How do I store an API key safely (not in code)?
