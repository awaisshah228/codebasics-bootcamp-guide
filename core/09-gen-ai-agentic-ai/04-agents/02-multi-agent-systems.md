# Agents 2 — Multi-Agent Systems

## Lectures covered
- Multi Agent Systems

---

## 1. Why multiple agents

A single LLM call has limits:
- Context window
- Reasoning depth
- Confused by mixed objectives

**Multi-agent** = decompose the work. Each agent has a focused role. They collaborate (sometimes by message passing, sometimes via shared state).

Common patterns:
- Researcher + writer + editor for content
- Planner + workers for project management
- Critic that reviews another agent's output
- Specialists per domain (legal, technical, marketing)

---

## 2. Three classic multi-agent patterns

### Pattern A — Pipeline (linear)

```
Researcher ──► Writer ──► Editor ──► output
```

Each agent's output is the next's input. Simple, predictable.

CrewAI sequential mode is this.

### Pattern B — Supervisor (hub-and-spoke)

```
              ┌──► Specialist A
Supervisor ──┼──► Specialist B
              └──► Specialist C
```

The supervisor routes user requests to the right specialist and aggregates results.

LangGraph's recommended pattern for most multi-agent setups.

### Pattern C — Swarm / Peer-to-peer

```
   Agent A ◄──► Agent B
       ▲           ▲
       │           │
       └─►Agent C◄─┘
```

Agents talk to each other, no single coordinator. Powerful but harder to control.

---

## 3. Supervisor pattern (LangGraph)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

llm = ChatOpenAI(model="gpt-4o-mini")

# Specialist agents
researcher = create_react_agent(llm, tools=[search_tool])
analyst    = create_react_agent(llm, tools=[code_tool])
writer     = create_react_agent(llm, tools=[])

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

def supervisor(state: TeamState):
    # ask LLM who should act next
    members = ["researcher", "analyst", "writer", "FINISH"]
    decision = route_via_llm(state["messages"], members)
    return {"next": decision}

graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "FINISH": END},
)
for m in ["researcher", "analyst", "writer"]:
    graph.add_edge(m, "supervisor")     # always loop back

team = graph.compile()
```

Pattern: supervisor decides → specialist runs → back to supervisor → ... → FINISH.

---

## 4. CrewAI hierarchical multi-agent

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
)
```

The manager LLM dynamically assigns tasks to agents — similar to LangGraph supervisor.

---

## 5. When multi-agent helps

- **Different expertise required** (technical + legal + creative)
- **Long horizon planning** that benefits from explicit decomposition
- **Parallel exploration** (multiple specialists run independently)
- **Self-correction** loops (writer + critic)

---

## 6. When it hurts

- **Simple tasks** that one agent solves better
- **Cost-sensitive** workloads (more LLM calls)
- **Tight latency** (more sequential steps)
- **You haven't tried single-agent first**

> **Default**: build single-agent first. Add agents only when one isn't sufficient.

---

## 7. The reflection / critic pattern

A common 2-agent setup that improves quality dramatically:

```
Generator → produces draft
   │
   ▼
Critic → reviews, gives feedback
   │
   ▼
Generator → revises with feedback
   │
   ▼
(repeat N times or until critic approves)
```

```python
def reflection_loop(prompt, max_iters=3):
    draft = generator.invoke(prompt)
    for _ in range(max_iters):
        critique = critic.invoke(f"Review this draft:\n{draft}")
        if "looks good" in critique.lower():
            return draft
        draft = generator.invoke(f"Original: {prompt}\nDraft: {draft}\nFeedback: {critique}\nRevise:")
    return draft
```

Cheap, effective, doubles quality on writing tasks.

---

## 8. Communication protocols

### Shared state
All agents read/write the same dict. Simple, requires care to avoid conflicts.

### Message passing
Agents send each other typed messages. Clearer audit trail. More overhead.

### Blackboard
A central "blackboard" where agents post observations / claims / updates. Other agents read.

### Tools-as-API
Agents call each other as tools. Useful when one is much heavier.

---

## 9. Example architectures from real systems

### MetaGPT (research → engineer → tester)
- Product manager: requirements
- Architect: system design
- Engineer: code
- Tester: tests

### AutoGen
- "Group chat" of agents that take turns
- A "manager" decides who speaks next

### Devin / Magnetic / Operator (coding agents)
- Planner + executor + reflector
- Sandboxed code execution

These are state-of-the-art examples — peek at their architectures for ideas.

---

## 10. Real example — content factory crew

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Research Analyst",
                    goal="Find recent advancements in {topic}",
                    tools=[search_tool])

outliner   = Agent(role="Content Strategist",
                    goal="Outline a 1500-word article based on research")

writer     = Agent(role="Senior Writer",
                    goal="Write engaging long-form content from outline")

editor     = Agent(role="Editor",
                    goal="Polish content; fact-check claims; ensure flow",
                    tools=[search_tool])

seo        = Agent(role="SEO Specialist",
                    goal="Add meta tags, optimize headings, suggest keywords")

tasks = [
    Task(description="Research {topic}", agent=researcher, ...),
    Task(description="Outline based on research", agent=outliner, ...),
    Task(description="Draft article", agent=writer, ...),
    Task(description="Edit + fact-check", agent=editor, ...),
    Task(description="SEO finalize", agent=seo, ...),
]

crew = Crew(agents=[researcher, outliner, writer, editor, seo], tasks=tasks)
result = crew.kickoff(inputs={"topic": "Gen AI agents"})
```

Output: ready-to-publish article. Used by content teams to scale 10× with one human reviewer.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Too many agents for the task | confusion, slow, expensive | start small |
| Roles overlap | redundant work | sharpen role / goal |
| No supervisor | agents wander | add a router |
| Critic too lenient | loops without improvement | strengthen critic prompt with criteria |
| Critic too strict | infinite loops | cap iterations |
| No shared state | agents can't coordinate | use proper state channel |

## Self-check

- [ ] When is multi-agent better than single-agent?
- [ ] Three multi-agent patterns?
- [ ] What's the supervisor pattern?
- [ ] When use CrewAI hierarchical mode?
- [ ] Walk through a reflection loop.
- [ ] What's the cost trade-off of multi-agent?
- [ ] Build a 3-agent supervisor team in LangGraph.
- [ ] Why default to single-agent first?
