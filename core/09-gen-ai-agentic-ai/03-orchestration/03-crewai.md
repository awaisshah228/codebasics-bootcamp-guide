# Orchestration 3 — CrewAI

## Lectures covered
- CrewAI crash course

---

## In one sentence
CrewAI is a Python framework that lets you build a multi-agent app by describing it like a project team: each agent has a **role**, **goal**, and **backstory**, you hand them **tasks**, and the **crew** runs the work — sequentially or with a manager agent assigning work dynamically.

## Real-world analogy
CrewAI is **a team of specialists** at a content agency. The Researcher, Writer, Editor, and SEO Specialist each have a job description (role), a quarterly target (goal), and a resume that shapes how they think (backstory). You hand the editor-in-chief a brief; they pass it down the line. LangGraph would model the same flow as a state machine; CrewAI lets you write it the way an HR manager would.

## The intuition (plain English)
- Many real workflows are naturally **role-based**: research → outline → draft → edit → publish. Modeling each role as its own LLM with its own tools matches how humans already organize the work.
- A CrewAI **agent** is mostly a structured prompt — `role`, `goal`, `backstory` get woven into the system prompt — plus optional tools and an LLM choice (Claude works fine via the LangChain adapter).
- A **task** is a specific assignment with a `description`, `expected_output`, and an `agent` who owns it. Tasks can depend on each other via `context=[earlier_task]`.
- A **crew** runs tasks either **sequentially** (default) or **hierarchically** (a manager agent picks who runs what next).
- Compared to LangGraph, you trade fine-grained control for fast prototyping when the work decomposes cleanly into roles.

## Mini worked example — 2 agents collaborating

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

researcher = Agent(
    role="Research Analyst",
    goal="Find 5 recent advances in {topic}",
    backstory="You read primary sources and cite them precisely.",
    tools=[SerperDevTool()],
)

writer = Agent(
    role="Content Writer",
    goal="Turn research into a 200-word brief",
    backstory="A former newsroom editor who cuts jargon.",
)

research = Task(
    description="Research recent advances in {topic}",
    expected_output="5 bullets with sources.",
    agent=researcher,
)

write = Task(
    description="Write a 200-word brief from the research",
    expected_output="One polished paragraph, no headings.",
    agent=writer,
    context=[research],          # downstream dependency
)

crew = Crew(agents=[researcher, writer], tasks=[research, write], process=Process.sequential)
print(crew.kickoff(inputs={"topic": "Gen AI agents"}))
```

Two roles, two tasks, one crew — a working content pipeline in ~25 lines.

## At-a-glance

```mermaid
flowchart LR
    U[User input<br/>{topic}] --> C[Crew]
    C --> R[Researcher Agent<br/>role + goal + backstory<br/>tools: web search]
    R --> W[Writer Agent<br/>role + goal + backstory]
    W --> O[Final output]
    M[Manager Agent<br/>Process.hierarchical] -.optional.-> R
    M -.optional.-> W

    classDef opt stroke-dasharray: 5 5
    class M opt
```

## Why this matters
- **Pick CrewAI when** the workflow decomposes cleanly into roles (research / write / review / publish), or when you want a quick multi-agent prototype without learning a graph DSL.
- **Pick LangGraph ([02-langgraph.md](./02-langgraph.md))** when you need explicit branching, retries, or human-in-the-loop pauses — control flow that CrewAI hides.
- **Pick LangChain ([01-langchain.md](./01-langchain.md))** for single-LLM pipelines with no agent collaboration.
- The HR onboarding project ([../05-projects/03-agentic-onboarding-mcp.md](../05-projects/03-agentic-onboarding-mcp.md)) uses CrewAI plus MCP tools — a good fit because onboarding is a sequence of role-driven steps.

---

## 1. What CrewAI is

CrewAI lets you build **multi-agent teams** with a high-level Python API. You define:
- **Agents** — each with a role, goal, and tools
- **Tasks** — pieces of work
- **Crew** — agents + tasks together

Inspired by how human teams collaborate. Strong abstraction for "research → write → review → publish" type workflows.

---

## 2. Install
```bash
pip install crewai crewai-tools
```

---

## 3. The minimum example

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool       # web search

researcher = Agent(
    role="Research Analyst",
    goal="Find recent news on the topic",
    backstory="A meticulous analyst who reads dozens of sources.",
    tools=[SerperDevTool()],
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Write a clear summary article",
    backstory="A former newspaper editor who simplifies complex topics.",
    verbose=True,
)

research_task = Task(
    description="Research recent advancements in {topic}",
    expected_output="A bullet-point list of 5 recent advances with sources.",
    agent=researcher,
)

write_task = Task(
    description="Write a 300-word summary article based on the research",
    expected_output="A polished 300-word article.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "Gen AI agents"})
print(result)
```

CrewAI executes tasks in order; each agent uses tools as needed; outputs flow downstream.

---

## 4. Agent fields

| Field | Required | Purpose |
|---|---|---|
| `role` | yes | one-line job title |
| `goal` | yes | what they're trying to achieve |
| `backstory` | yes | context that shapes behavior |
| `tools` | optional | functions / APIs they can call |
| `llm` | optional | model to use (default OpenAI) |
| `allow_delegation` | optional | can ask other agents for help |
| `verbose` | optional | print reasoning |
| `memory` | optional | persist across runs |

The `role`, `goal`, `backstory` are essentially structured prompt engineering — they get woven into the agent's system prompt.

---

## 5. Task fields

| Field | Required | Purpose |
|---|---|---|
| `description` | yes | what to do (with `{var}` placeholders) |
| `expected_output` | yes | what success looks like |
| `agent` | yes | who runs it |
| `tools` | optional | task-specific tools |
| `context` | optional | tasks whose outputs this depends on |
| `output_file` | optional | save result to a file |

---

## 6. Sequential vs parallel execution

```python
from crewai import Process

# Sequential (default)
crew = Crew(..., process=Process.sequential)

# Hierarchical (a manager agent decides who runs what)
crew = Crew(..., process=Process.hierarchical, manager_llm=ChatOpenAI(model="gpt-4o"))
```

For simple linear pipelines: sequential.
For complex projects where dynamic delegation matters: hierarchical (the manager picks the next agent).

---

## 7. Custom tools

```python
from crewai.tools import BaseTool

class MyDBTool(BaseTool):
    name: str = "DB Query"
    description: str = "Query the company database. Input: SQL query string."

    def _run(self, sql: str) -> str:
        # actually run the query, sandbox it
        return execute_sql_safely(sql)

tools = [MyDBTool()]
```

Built-in tools (via `crewai-tools`):
- `SerperDevTool` — web search
- `ScrapeWebsiteTool` — fetch + extract pages
- `FileReadTool` / `FileWriteTool`
- `SeleniumScrapingTool`
- `DallETool`
- many more

---

## 8. Realistic crew — content factory

Imagine: marketing wants 5 blog posts/week. Crew:
- **Researcher** finds trending topics
- **Outline Writer** creates outlines
- **Content Writer** drafts articles
- **Editor** reviews and improves
- **SEO Specialist** suggests keywords + meta descriptions
- **Publisher** posts to CMS via tool

This gives you 5 articles/week from one human review point. CrewAI makes this configurable in ~100 lines.

---

## 9. CrewAI vs LangGraph — pick which?

| | CrewAI | LangGraph |
|---|---|---|
| Mental model | "team of role-played experts" | "state machine of nodes" |
| Setup | quickest for typical multi-agent | most flexible for any flow |
| Branching / loops | possible but less explicit | first-class |
| Custom control flow | harder | easy |
| Production observability | basic | LangSmith integration |
| Best for | content / research / multi-step reports | complex agent loops |

**For Codebasics' bootcamp project** (HR onboarding agent with MCP — Module 9 project 3): CrewAI is a fine fit.

For more complex, branchy agents (Module 9 project 4 — Customer Care AgentCore): LangGraph or AgentCore native is better.

---

## 10. Memory + caching

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,                    # short-term memory between tasks
    cache=True,                     # cache tool calls
    embedder={"provider": "openai"},
)
```

Memory lets agents reference prior tasks' outputs implicitly.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Vague role / goal / backstory | agents waffle | be specific (1-2 sentences each) |
| Too many agents in one crew | confusion + cost | start with 2-3, grow if needed |
| Tasks that overlap | redundant work | partition tasks cleanly |
| Letting agents loop without limit | runaway cost | use `max_iterations` |
| No error handling on tools | crash mid-crew | wrap tool `_run` in try/except |
| Output not actually used downstream | task chain broken | use `context=[earlier_task]` to link |

## Self-check

- [ ] What three fields define a CrewAI agent?
- [ ] What's the difference between `Process.sequential` and `Process.hierarchical`?
- [ ] Build a 2-agent crew (researcher + writer) for blog posts.
- [ ] How do you give an agent a custom tool?
- [ ] When prefer CrewAI over LangGraph?
- [ ] What does `memory=True` do?
- [ ] How do you cap an agent's reasoning loops?
- [ ] What's a realistic 4-5 agent crew for a content team?

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| CrewAI | Role-based multi-agent framework with `Agent`, `Task`, `Crew` primitives. |
| Agent | A single LLM persona with role, goal, backstory, and optional tools. |
| Role | One-line job title that goes into the agent's system prompt. |
| Goal | What the agent is trying to achieve, in 1–2 sentences. |
| Backstory | Context paragraph that shapes the agent's voice and priorities. |
| Task | A specific assignment with description, expected output, and an owning agent. |
| `expected_output` | A description of what success looks like — guides the agent's stopping point. |
| `context` | List of upstream tasks whose outputs are fed into this task. |
| Crew | The container that owns agents and tasks and runs them. |
| `kickoff` | The Crew method that starts execution; returns the final output. |
| Process | Execution mode: `sequential` (run tasks in order) or `hierarchical` (manager routes). |
| Manager agent | A hierarchical-mode coordinator that picks the next agent dynamically. |
| `manager_llm` | The LLM that powers the manager in hierarchical mode. |
| `allow_delegation` | Lets an agent ask peer agents for help mid-task. |
| Tool | A callable the agent can invoke; CrewAI provides built-ins via `crewai-tools`. |
| `BaseTool` | Class to subclass when writing a custom tool. |
| `SerperDevTool` | Built-in web-search tool. |
| `ScrapeWebsiteTool` | Built-in tool that fetches and extracts page content. |
| Memory | Short-term store letting agents reference earlier tasks' outputs. |
| Cache | Optional layer that reuses identical tool-call results. |
| Embedder | The embedding model used for memory retrieval. |
| `verbose` | When true, prints each agent's reasoning and tool calls. |
| `max_iterations` | Upper bound on an agent's reasoning loops per task. |
| ReAct | Reason + Act — the underlying loop CrewAI agents use. |
| LangGraph | Lower-level state-machine alternative for non-role-based flows. |
| MCP | Model Context Protocol — lets you expose tools that any agent (CrewAI included) can consume. |
| Content factory | Common 4–6 agent crew pattern: researcher, outliner, writer, editor, SEO, publisher. |

## Further reading
- Previous: [02-langgraph.md](./02-langgraph.md) — graph-based alternative
- Next: [04-mcp.md](./04-mcp.md) — standardize tools that crews consume
- Sibling: [01-langchain.md](./01-langchain.md), [05-amazon-bedrock-agentcore.md](./05-amazon-bedrock-agentcore.md)
- Foundation: [../04-agents-tool-use.md](../04-agents-tool-use.md) — multi-agent section
- Companion: [../06-langchain-claude-api.md](../06-langchain-claude-api.md) — Claude SDK behind the scenes
- Project: [../05-projects/03-agentic-onboarding-mcp.md](../05-projects/03-agentic-onboarding-mcp.md) — CrewAI + MCP for HR onboarding
- CrewAI — [Documentation](https://docs.crewai.com/)
- CrewAI — [Examples](https://github.com/crewAIInc/crewAI-examples)
- CrewAI — [Tools](https://docs.crewai.com/tools/)
