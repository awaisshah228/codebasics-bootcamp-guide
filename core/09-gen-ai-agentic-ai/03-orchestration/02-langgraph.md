# Orchestration 2 — LangGraph

## Lectures covered
- LangGraph Crash course

---

## 1. Why LangGraph

Plain LangChain chains are **straight-line**: input → step 1 → step 2 → output.

Real agents need:
- Loops (re-tool until satisfied)
- Branches (different paths based on conditions)
- State (memory across steps)
- Human-in-the-loop pauses
- Persistence + resumption
- Streaming intermediate state

**LangGraph** is a state-machine framework for LLM apps. Inspired by graph neural network frameworks (PyTorch's autograd graph) but for prompts + tools.

---

## 2. Install
```bash
pip install langgraph
```

---

## 3. The mental model

You define:
- **State** — a typed dict that flows through the graph
- **Nodes** — functions that read/write state
- **Edges** — control flow between nodes
- **Conditional edges** — branch based on state

Then you compile to a graph that you can `invoke`, `stream`, or `astream`.

---

## 4. A minimal LangGraph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    output: str

def step1(state: State) -> State:
    return {"output": f"Hello, {state['input']}!"}

graph = StateGraph(State)
graph.add_node("greet", step1)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

app = graph.compile()

result = app.invoke({"input": "world"})
print(result)        # {'input': 'world', 'output': 'Hello, world!'}
```

State updates are **merged** into the existing state — partial returns are fine.

---

## 5. Conditional edges (branching)

```python
def classify(state: State) -> State:
    # decide if we need a tool or can answer directly
    return {"path": "tool" if "weather" in state["input"] else "direct"}

def use_tool(state):  return {"output": "[called weather tool] sunny"}
def answer_direct(state): return {"output": "I can answer that."}

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("use_tool", use_tool)
graph.add_node("direct",   answer_direct)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    lambda s: s["path"],            # routing key
    {"tool": "use_tool", "direct": "direct"},
)

graph.add_edge("use_tool", END)
graph.add_edge("direct", END)

app = graph.compile()
```

---

## 6. The pre-built ReAct agent

LangGraph ships a **prebuilt ReAct agent** — the most common agent pattern.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"top result for '{query}'..."

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[get_weather, search_web])

result = agent.invoke({"messages": [("user", "What's the weather in Lahore right now?")]})
for m in result["messages"]:
    print(m.type, ":", m.content)
```

Under the hood: ReAct loop. Model decides when to call tools, calls them, and uses results to answer.

---

## 7. State accumulation — appending to lists

For chat history:
```python
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chat(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

`add_messages` is a reducer — instead of overwriting, it appends new messages to the list.

---

## 8. Memory + persistence (checkpointing)

For multi-turn chats / resumable workflows:
```python
from langgraph.checkpoint.memory import MemorySaver
# or for prod: from langgraph.checkpoint.postgres import PostgresSaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-42"}}

app.invoke({"messages": [("user", "Hi")]}, config)
app.invoke({"messages": [("user", "What did I just say?")]}, config)
# state persisted across calls per thread_id
```

Critical for production agents.

---

## 9. Human-in-the-loop

You can **pause** execution and resume later (after human approval):

```python
from langgraph.graph import START, END
graph.add_node("draft", draft_email)
graph.add_node("send", send_email)

# pause before "send" so a human approves
app = graph.compile(checkpointer=memory, interrupt_before=["send"])

state = app.invoke({"...": "..."}, config)
# inspect the draft
# ...if human approves:
app.invoke(None, config)         # resume from where it stopped
```

---

## 10. Streaming intermediate state

```python
async for event in app.astream({"input": "..."}, config, stream_mode="updates"):
    print(event)            # see each node's output as it happens
```

Use `stream_mode`:
- `"values"` — full state after each step
- `"updates"` — only the changes
- `"messages"` — token-level streaming

---

## 11. Multi-agent setups (preview — full in `04-agents/02-multi-agent-systems.md`)

LangGraph naturally expresses multi-agent collaboration:
- Each agent is a subgraph
- A "supervisor" agent routes between them
- Shared state coordinates

This is the **modern recommended** way to build complex agentic apps.

---

## 12. When LangGraph beats LangChain

| | LangChain (LCEL) | LangGraph |
|---|---|---|
| Linear chains | great | overkill |
| Branching / loops | clunky | natural |
| Stateful conversations | works (memory) | first-class |
| Multi-agent | possible | designed for it |
| Human-in-the-loop | manual | built-in |
| Resumable workflows | hard | first-class |

**Rule**: chain → LangChain. Anything more complex → LangGraph.

---

## 13. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgetting reducer (`Annotated[list, add_messages]`) | state overwritten | use the reducer |
| Cycles without termination condition | infinite loop | always have a "stop" branch |
| Not using checkpointer | state lost between calls | enable for multi-turn |
| Mixing LangGraph state with global vars | hard to debug | keep all flow in state |
| Ignoring `stream_mode` for UX | feels frozen during long runs | stream updates |

## Self-check

- [ ] What's a node, an edge, a state in LangGraph?
- [ ] When is LangGraph better than LangChain?
- [ ] How do you implement branching (conditional edges)?
- [ ] What does `add_messages` reducer do?
- [ ] How do you pause a graph for human approval?
- [ ] How does checkpointing work?
- [ ] Build a 2-node graph: classify → answer.
- [ ] Use `create_react_agent` to build a tool-using agent in 10 lines.
