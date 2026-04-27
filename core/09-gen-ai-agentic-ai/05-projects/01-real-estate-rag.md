# Project 1 — Real Estate Assistant Using RAG

## Domain
A real-estate site has thousands of property listings. Users ask natural-language questions like:
- "3-bedroom apartment in Lahore under 8M PKR"
- "Pet-friendly studios near university"
- "Houses with garden and parking"

Build an **AI assistant** that retrieves relevant listings from the database and answers in natural language.

## Pattern
Classic **RAG** with structured filtering:
- Vector search over listing descriptions
- Metadata filtering on price / location / bedrooms
- LLM generates a clean, cited answer

---

## Architecture

```
user question
   │
   ▼
[Streamlit UI]
   │
   ▼
1. Extract structured filters from query (LLM call) ─► {city, max_price, bedrooms}
   │
   ▼
2. Vector search ChromaDB filtered by metadata
   │
   ▼
3. Top-k listings → context
   │
   ▼
4. LLM generates answer with cited listing IDs / URLs
   │
   ▼
display results + URLs in UI
```

---

## Step-by-step build

### 1. Data — listings dataset
Either:
- Codebasics-provided listings
- Public dataset (Kaggle: real-estate listings)
- Scrape (carefully) — practice for portfolio

Each listing should have:
```python
{
  "id": "...",
  "description": "...",      # full text — what we embed
  "city": "Lahore",
  "neighborhood": "DHA Phase 5",
  "type": "apartment",        # apartment / house / studio / villa
  "bedrooms": 3,
  "bathrooms": 2,
  "size_sqft": 1200,
  "price": 12_000_000,        # in PKR
  "url": "https://...",
}
```

### 2. Index in ChromaDB
```python
import chromadb
client = chromadb.PersistentClient(path="./re_db")
col = client.get_or_create_collection(name="listings")

col.add(
    documents=[l["description"] for l in listings],
    metadatas=[{k: l[k] for k in ["city","type","bedrooms","bathrooms","size_sqft","price","url"]}
               for l in listings],
    ids=[l["id"] for l in listings],
)
```

### 3. Filter extraction (the LLM converts NL → structured)

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_filters(query: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content":
             "Extract structured search filters from the user query. "
             "Return JSON with optional keys: city, type, max_price, min_bedrooms. "
             "Omit unspecified keys."},
            {"role": "user", "content": query},
        ],
    )
    return json.loads(resp.choices[0].message.content)
```

### 4. Build the Chroma `where` from filters
```python
def build_where(filters):
    clauses = []
    if "city" in filters:
        clauses.append({"city": filters["city"]})
    if "type" in filters:
        clauses.append({"type": filters["type"]})
    if "max_price" in filters:
        clauses.append({"price": {"$lte": filters["max_price"]}})
    if "min_bedrooms" in filters:
        clauses.append({"bedrooms": {"$gte": filters["min_bedrooms"]}})

    if not clauses:    return {}
    if len(clauses) == 1: return clauses[0]
    return {"$and": clauses}
```

### 5. Search + generate

```python
def answer(query):
    filters = extract_filters(query)
    where   = build_where(filters)

    res = col.query(query_texts=[query], n_results=5, where=where)
    docs  = res["documents"][0]
    metas = res["metadatas"][0]

    if not docs:
        return "No matching listings found."

    context = "\n\n".join(
        f"[{m['url']}] {d}" for d, m in zip(docs, metas)
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content":
             "You are a real-estate assistant. Recommend listings from the context. "
             "Always cite each recommendation with its URL in [brackets]."},
            {"role": "user", "content":
             f"Context:\n{context}\n\nUser request: {query}"},
        ],
    )
    return resp.choices[0].message.content
```

### 6. Streamlit UI

```python
import streamlit as st

st.title("🏡 Real Estate Assistant")
query = st.text_area("Describe what you're looking for:")

if st.button("Search") and query:
    with st.spinner("Finding properties..."):
        result = answer(query)
    st.markdown(result)
```

That's a complete RAG product in **~120 lines**.

---

## Stretch goals (great for portfolio depth)

### A. Reranking
Add a cross-encoder rerank between top-50 and top-5.

### B. Image-aware
Listings often have photos. Add CLIP embeddings → users can query "with a pool" and find matching images.

### C. Conversational memory
Multi-turn: "show me 3-bedroom" → "make it cheaper" — uses session memory.

### D. Maps view
Geo coordinates in metadata; overlay top-5 results on a map.

### E. Lead generation
"I'm interested" button → captures contact and notifies an agent.

### F. Eval suite
30 golden queries with expected listings; track top-5 hit rate over time.

---

## Repo deliverables

```
real-estate-rag/
├── data/
│   └── listings.csv
├── notebooks/
│   ├── 01-data-prep.ipynb
│   ├── 02-index-build.ipynb
│   └── 03-eval.ipynb
├── src/
│   ├── retrieval.py
│   ├── filter_extractor.py
│   ├── answer.py
│   └── app.py            # Streamlit
├── evals/
│   ├── golden.json
│   └── run.py
├── re_db/                # chroma persistence (gitignored)
├── requirements.txt
└── README.md             # demo screenshots, eval numbers
```

---

## Self-check

- [ ] Did I extract structured filters before vector search?
- [ ] Are all answers grounded in retrieved listings (no hallucinated URLs)?
- [ ] Eval on 30 golden queries documented in README?
- [ ] Streamlit demo deployed publicly?
- [ ] Build-in-public LinkedIn post with demo link?
- [ ] Top-5 hit rate measured?
- [ ] Repo includes a clean README + run instructions?
