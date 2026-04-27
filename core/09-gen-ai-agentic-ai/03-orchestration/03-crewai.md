# Orchestration 3 — CrewAI

## Lectures covered
- CrewAI crash course

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
