# Orchestration 1 — LangChain

## Lectures covered
- LangChain Installation and Setup
- Calling LLM from LangChain
- Prompt Templates & Chains
- Streamlit UI Development (referenced — covered in DL deployment + Project files)

---

## In one sentence
LangChain is a Python library that gives you a standard set of plug-and-play building blocks — prompts, models, parsers, retrievers, memory — so you can wire up an LLM app with a single `|` operator instead of hand-rolling glue code for every project.

## Real-world analogy
LangChain is **LEGO bricks for LLM apps**. The Claude SDK gives you raw plastic; LangChain gives you bricks with standard studs (a prompt brick, a model brick, a parser brick) so you can snap together a RAG chatbot in 25 lines and rip out one brick (swap Claude for GPT) without rebuilding the whole castle.

## The intuition (plain English)
- A Gen AI app is almost never a single LLM call. You **format a prompt**, **call the model**, **parse the output**, often **retrieve context** first, sometimes **remember chat history**.
- LangChain names each of those steps and gives them a uniform interface: every step is a `Runnable` you can `.invoke(input)` on.
- The killer feature is **LCEL** (LangChain Expression Language): `prompt | llm | parser` reads left-to-right like a Unix pipe and Just Works for streaming, async, and batching.
- You stay close to vanilla Python — no DSL, no YAML config, no hidden state. Each piece is debuggable.
- Pair it with `ChatAnthropic` to keep Claude as your model and you have a production-ready scaffold in minutes.

## Mini worked example — a 5-line LCEL pipeline

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Translate to {language}: {text}")
    | ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    | StrOutputParser()
)

print(chain.invoke({"language": "French", "text": "Hello, world!"}))
# -> "Bonjour, le monde !"
```

Three components, one pipe operator, runnable end-to-end. Add a retriever in front and you have RAG; add a tool node behind and you have an agent.

## At-a-glance

```mermaid
flowchart LR
    I[Input dict<br/>{question, ...}] --> P[ChatPromptTemplate<br/>fills variables]
    P --> L[ChatAnthropic<br/>calls Claude API]
    L --> O[OutputParser<br/>str / JSON / Pydantic]
    O --> R[Result]
    RT[Retriever<br/>vector store] -.optional.-> P
    M[Memory<br/>chat history] -.optional.-> P

    classDef opt stroke-dasharray: 5 5
    class RT,M opt
```

## Why this matters
- LangChain is the most common scaffold in production Gen AI code today — knowing it is table stakes.
- LCEL chains stream, retry, and trace for free; rolling your own pipeline means re-implementing each.
- **Pick LangChain when** your app has 3+ steps (retrieve → prompt → parse) or you want provider portability.
- **Pick the plain SDK** ([../06-langchain-claude-api.md](../06-langchain-claude-api.md)) for one-shot calls; **pick LangGraph** ([02-langgraph.md](./02-langgraph.md)) when you need branches, loops, or human-in-the-loop pauses.

---

## 1. What LangChain is (and isn't)

LangChain is a **composition framework** for LLM apps. It provides:
- Standardized interfaces for LLMs, embeddings, vector stores
- Prompt templates with variable substitution
- "Chains" — pipelines of operations
- "Agents" — LLM-driven tool use
- Integrations with hundreds of services

What it's *not*: a model, a vector DB, or a magical answer to RAG. It's plumbing.

> Modern preference: use **LangGraph** (a successor) for anything beyond simple chains. LangChain is still useful for one-shot patterns and prebuilt integrations.

---

## 2. Install
```bash
pip install langchain langchain-openai langchain-community
```

For Anthropic / Google / Cohere etc., add their packages:
```bash
pip install langchain-anthropic langchain-google-genai
```

---

## 3. Calling an LLM
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# or:
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

resp = llm.invoke("What is the capital of France?")
print(resp.content)
```

`.invoke` for one-shot. `.stream` for streaming. `.batch` for parallel.

---

## 4. Prompt templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Translate this to {language}: {text}"),
])

formatted = prompt.format_messages(language="French", text="Hello, world!")
resp = llm.invoke(formatted)
```

Variables in `{braces}` are filled at runtime. Cleaner than f-strings for reusable prompts.

---

## 5. Chains via LCEL (LangChain Expression Language)

The modern way. Compose with `|` like Unix pipes:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"language": "French", "text": "Hello, world!"})
print(result)        # "Bonjour, le monde !"
```

### Why LCEL
- Composable: `chain1 | chain2 | chain3`
- Streaming-aware: `.stream(...)` works through the whole chain
- Async: `.ainvoke(...)` works through the whole chain
- Built-in tracing (via LangSmith)

### Parallel branches
```python
from langchain_core.runnables import RunnableParallel

paraprompt = ChatPromptTemplate.from_template("Paraphrase: {text}")
sumprompt  = ChatPromptTemplate.from_template("Summarize: {text}")

chain = RunnableParallel(
    paraphrase=paraprompt | llm | StrOutputParser(),
    summary=sumprompt | llm | StrOutputParser(),
)
chain.invoke({"text": "long article ..."})
# {'paraphrase': '...', 'summary': '...'}
```

---

## 6. Tools — letting the LLM call functions

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

llm_with_tools = llm.bind_tools([get_weather])
resp = llm_with_tools.invoke("What's the weather in Lahore?")
print(resp.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Lahore'}, 'id': '...'}]
```

You then execute the tool yourself and feed the result back. (This is what LangGraph automates.)

---

## 7. Quick RAG with LangChain
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load + chunk
docs = TextLoader("knowledge.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Index
vs = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma")

# Retrieve + generate
retriever = vs.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
Answer based on the context:
{context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("How do I cancel my subscription?"))
```

This is the canonical LangChain RAG. ~30 lines for an end-to-end pipeline.

---

## 8. Memory — multi-turn conversations
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_history = RunnableWithMessageHistory(chain, get_session_history)

with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user-42"}},
)
```

For production, persist the history to Redis / Postgres / Mongo.

---

## 9. Output parsers — getting structured data

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="full name")
    age: int = Field(description="age in years")

parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Extract person info as JSON.\n{format_instructions}\nText: {text}",
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 30."})
print(result)        # {'name': 'John', 'age': 30}
```

For OpenAI / Anthropic newer models, use `with_structured_output(Person)` directly:
```python
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("John is 30.")        # Person(name='John', age=30)
```

---

## 10. LangSmith — tracing + observability

LangSmith is LangChain's hosted product for inspecting every step of a chain — see prompts, responses, tool calls, latency, cost.

```bash
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
```

Then every chain run is logged to https://smith.langchain.com.

Free tier covers most projects. Essential for debugging real RAG / agent flows.

---

## 11. When to drop LangChain

- Your app is simple enough that the framework adds friction → go back to direct API calls
- You hit weird abstractions for a fairly straightforward task
- You're building something complex (multi-agent, branching) → switch to **LangGraph**

LangChain is great for **typical** RAG / chain patterns. Not always optimal for *highly* custom or *extremely* simple cases.

---

## 12. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Using LangChain when 50 lines of direct API would do | over-abstraction | reach for it only when you compose 3+ steps |
| Mixing old (`LLMChain`) with new (LCEL) | confusing code | use LCEL only |
| Hardcoded `OPENAI_API_KEY` | leaked | env vars |
| Forgetting `StrOutputParser` | get message obj instead of string | always parse |
| Calling `.invoke` in a loop | not parallel | use `.batch(list)` |

## Self-check

- [ ] What's LCEL and how do you compose with it?
- [ ] Build a prompt template with variables, pipe into an LLM, parse output.
- [ ] Build a 30-line RAG chain.
- [ ] How does `with_structured_output` simplify JSON extraction?
- [ ] What's the difference between `invoke`, `stream`, and `batch`?
- [ ] When do you drop LangChain for direct API calls?
- [ ] What's LangSmith and why use it?
- [ ] When prefer LangGraph over LangChain?

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| LangChain | Python library for composing LLM apps from standard building blocks. |
| LangChain Core | The base abstractions package: `Runnable`, prompts, parsers. |
| LangChain Community | Optional integrations (vector stores, loaders, third-party tools). |
| `langchain-anthropic` | The package that wires Claude into LangChain. |
| `ChatAnthropic` | LangChain wrapper around the Claude API. |
| LCEL | LangChain Expression Language — the `prompt \| llm \| parser` pipe syntax. |
| Runnable | Anything you can `.invoke(input)` on; chains are runnables. |
| `ChatPromptTemplate` | Reusable prompt with `{variable}` placeholders. |
| Prompt template | A string with placeholders that becomes a `messages` array at runtime. |
| Output parser | Converts the model's text into a typed Python object. |
| `StrOutputParser` | Parser that just pulls the string out of the response. |
| `JsonOutputParser` | Parser that decodes JSON output into a dict. |
| `PydanticOutputParser` | Parser that validates output against a Pydantic schema. |
| `with_structured_output` | Shortcut method for typed extraction on newer models. |
| `RunnablePassthrough` | Pass an input through unchanged (useful in parallel branches). |
| `RunnableLambda` | Wrap a plain Python function so it fits in a chain. |
| `RunnableParallel` | Run multiple chains side-by-side and collect results in a dict. |
| Retriever | Object with `.get_relevant_documents(query)` — the RAG fetch step. |
| Vector store | Wrapper around a vector DB (Chroma, FAISS, Pinecone). |
| RAG | Retrieval-Augmented Generation — fetch context, then prompt. |
| Memory | Mechanism to carry chat history or facts across calls. |
| `RunnableWithMessageHistory` | LangChain helper to attach per-session history to a chain. |
| Tool | A callable the LLM can request via tool use / function calling. |
| `bind_tools` | Attach a tool list to a model so it can emit tool calls. |
| `.invoke` / `.stream` / `.batch` | One-shot / streaming / parallel chain execution. |
| LangSmith | LangChain's hosted tracing and observability product. |
| LangGraph | LangChain's state-machine framework for agents (covered next). |
| Provider portability | Swapping the underlying model (Claude, GPT, Gemini) without rewriting app code. |

## Further reading
- Next in module: [02-langgraph.md](./02-langgraph.md) — when chains aren't enough
- Sibling: [03-crewai.md](./03-crewai.md), [04-mcp.md](./04-mcp.md), [05-amazon-bedrock-agentcore.md](./05-amazon-bedrock-agentcore.md)
- Companion: [../06-langchain-claude-api.md](../06-langchain-claude-api.md) — LangChain anchored on the Claude SDK
- Foundation: [../04-agents-tool-use.md](../04-agents-tool-use.md) — what tools and agents really are
- Project: [../05-projects/03-agentic-onboarding-mcp.md](../05-projects/03-agentic-onboarding-mcp.md), [../05-projects/04-customer-care-agentcore.md](../05-projects/04-customer-care-agentcore.md)
- LangChain — [Conceptual guide](https://python.langchain.com/docs/concepts/)
- LangChain — [LCEL docs](https://python.langchain.com/docs/expression_language/)
- LangChain — [`ChatAnthropic` reference](https://python.langchain.com/docs/integrations/chat/anthropic/)
