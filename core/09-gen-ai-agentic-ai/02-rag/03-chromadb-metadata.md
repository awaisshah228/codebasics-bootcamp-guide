# RAG 3 — ChromaDB & Metadata Filtering

## Lectures covered
- Chromadb
- Metadata Filtering

---

## 1. Why ChromaDB for the bootcamp

- Pure-Python install
- File-based persistence (no server to run)
- Embeds documents for you (or accepts custom embeddings)
- Metadata filtering built in
- Same API works for prototype and (small) production

```bash
pip install chromadb
```

---

## 2. Two client modes

### In-memory (lost on process exit)
```python
import chromadb
client = chromadb.Client()
```
For unit tests / quick experiments.

### Persistent (saved to disk)
```python
client = chromadb.PersistentClient(path="./chroma_db")
```
For most use. Stores SQLite + parquet under the path.

### Server mode (network access)
```python
client = chromadb.HttpClient(host="localhost", port=8000)
```
For multi-process / multi-machine.

---

## 3. Collections — like tables

```python
col = client.get_or_create_collection(
    name="real_estate",
    metadata={"hnsw:space": "cosine"},      # distance metric
)
```

Distance options: `cosine` (default for text), `l2` (Euclidean), `ip` (inner product).

### List / delete collections
```python
client.list_collections()
client.delete_collection(name="real_estate")
```

---

## 4. Adding documents (embedding done for you)

```python
col.add(
    documents=[
        "3-bedroom condo in downtown, $500k.",
        "5-bedroom villa in suburbs, $1.2M.",
        "Studio apartment near university, $800/mo.",
    ],
    metadatas=[
        {"type": "condo", "city": "Lahore", "price": 500000},
        {"type": "villa", "city": "Lahore", "price": 1200000},
        {"type": "studio", "city": "Karachi", "price": 800},
    ],
    ids=["1", "2", "3"],
)
```

ChromaDB ships with a default embedder (`all-MiniLM-L6-v2`) that runs locally. No API key needed for prototypes.

### Custom embedder — OpenAI
```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

emb = OpenAIEmbeddingFunction(api_key="sk-...", model_name="text-embedding-3-small")
col = client.get_or_create_collection(name="real_estate", embedding_function=emb)
```

Or pass your own pre-computed embeddings via `embeddings=...`.

---

## 5. Querying

```python
res = col.query(
    query_texts=["affordable apartment for student"],
    n_results=3,
)
print(res["documents"][0])      # list of top-3 chunks (for the 1 query)
print(res["distances"][0])      # corresponding distances
print(res["metadatas"][0])
```

### Multiple queries at once (batched)
```python
res = col.query(query_texts=["query A", "query B"], n_results=3)
# res["documents"] is a 2-element list
```

---

## 6. Metadata filtering — the big productivity win

Plain RAG retrieves the most semantically similar chunks. **Metadata filtering** lets you constrain the search:

```python
res = col.query(
    query_texts=["affordable apartment"],
    n_results=3,
    where={"city": "Lahore"},
)
```

### Operators
```python
where={"price": {"$lt": 500000}}              # less than
where={"price": {"$lte": 500000}}              # ≤
where={"price": {"$gt": 100000}}               # >
where={"price": {"$gte": 100000}}              # ≥
where={"city": {"$ne": "Karachi"}}             # not equal
where={"city": {"$in": ["Lahore", "Islamabad"]}}
where={"city": {"$nin": ["Karachi"]}}
where={"$and": [
    {"city": "Lahore"},
    {"price": {"$lt": 600000}},
]}
where={"$or": [
    {"type": "studio"},
    {"price": {"$lt": 100000}},
]}
```

### Where-document filtering (text contains)
```python
where_document={"$contains": "downtown"}
```

### Real-world example
A user query: "affordable 3-bedroom in Lahore for ≤ ₹600k."

Pure semantic search may return a 10-bedroom mansion that *mentions* "affordable in description." Add metadata constraint → only chunks where `city=Lahore`, `price ≤ 600000`, `bedrooms ≥ 3`.

This is **how production RAG systems actually work**. Naked semantic search is a toy.

---

## 7. Updating + deleting

```python
col.update(
    ids=["1"],
    documents=["3-bedroom condo, REDUCED PRICE $450k."],
    metadatas=[{"price": 450000}],
)
col.upsert(
    ids=["4"],
    documents=["new listing"],
    metadatas=[{"city": "Lahore"}],
)
col.delete(ids=["2"])
col.delete(where={"price": {"$lt": 1000}})
```

---

## 8. Practical pattern — RAG with metadata for the real-estate project

```python
import chromadb
from openai import OpenAI

oai = OpenAI()
chroma = chromadb.PersistentClient(path="./re_db")
col = chroma.get_or_create_collection(name="listings")

# Index real estate listings (one chunk per listing or per paragraph)
col.add(
    documents=[listing_descriptions],
    metadatas=[{
        "city": l.city, "type": l.type, "bedrooms": l.bedrooms,
        "bathrooms": l.bathrooms, "price": l.price, "url": l.url,
    } for l in listings],
    ids=[l.id for l in listings],
)

def query_listings(user_query, filters=None):
    res = col.query(
        query_texts=[user_query],
        n_results=5,
        where=filters or {},
    )
    chunks = res["documents"][0]
    metas  = res["metadatas"][0]

    context = "\n\n".join(f"[{m['url']}] {c}" for c, m in zip(chunks, metas))
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content":
                "Recommend properties from the context, citing URLs."},
            {"role": "user", "content":
                f"Context:\n{context}\n\nUser request: {user_query}"},
        ],
    )
    return resp.choices[0].message.content

# Usage
print(query_listings(
    "Family-friendly home with garden",
    filters={"$and": [{"city": "Lahore"}, {"bedrooms": {"$gte": 3}}, {"price": {"$lte": 800000}}]},
))
```

This is the skeleton for the real-estate assistant project (Module 9 project 1).

---

## 9. Performance and scale

ChromaDB on a laptop:
- 100k vectors: easy
- 1M vectors: slow but works
- 10M+: switch to a real vector DB

For bootcamp projects (a few thousand listings / chunks), you'll never feel a limit.

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Mixing default embedder + custom embedder in same collection | shape mismatch | choose one and stick |
| Metadata filtering on un-indexed numeric | slow | metadata is indexed by Chroma; ensure types are correct |
| Storing long lists in metadata | metadata limits | embed list as separate docs or use proper type |
| Embedding only at index, not at query | wrong results | Chroma re-embeds query for you |
| Forgetting `PersistentClient` | data lost on restart | always use persistent in prototypes |

## Self-check

- [ ] Difference between in-memory and persistent ChromaDB?
- [ ] What does ChromaDB do with documents you `add`?
- [ ] How do I query with a metadata filter "price < 500000 AND city = Lahore"?
- [ ] Use a custom OpenAI embedder.
- [ ] Walk through indexing 1000 real-estate listings + querying with filters.
- [ ] When outgrow Chroma — what next?
- [ ] How do I update a listing's price after it changes?
- [ ] Delete all vectors with `city=Karachi`.
