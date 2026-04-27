# Orchestration 5 — Amazon Bedrock AgentCore

## Lectures covered
- Amazon Bedrock AgentCore

---

## 1. What AgentCore is

**Amazon Bedrock AgentCore** is AWS's managed runtime for production AI agents. It's not a framework like LangGraph or CrewAI — it's the **infrastructure layer** beneath them.

You can build your agent with **any framework** (LangGraph, CrewAI, Strands, custom code) and deploy it to AgentCore for:
- Secure execution
- Persistent memory (short-term and long-term)
- Identity management
- Observability and audit logs
- Scaling without managing servers

This is what Codebasics' **Project 4 (Customer Care AI Agent)** uses.

---

## 2. The capability stack

```
┌─────────────────────────────────────────────────────────┐
│  Your Agent (built in any framework)                     │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│  AgentCore Runtime — secure, serverless agent execution  │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│  AgentCore Services                                       │
│   - Memory (short-term + long-term)                       │
│   - Identity (per-user auth, OAuth)                       │
│   - Browser tool (sandboxed automation)                   │
│   - Code Interpreter (safe code execution)                │
│   - Gateway (MCP-compliant tool exposure)                 │
│   - Observability (traces, metrics, audit log)            │
└────────────────────────────────────────────────────────┘
```

---

## 3. Why AgentCore for production

Compare a homemade agent vs an AgentCore-deployed one:

| | Homemade | AgentCore |
|---|---|---|
| Server management | you | AWS |
| Auth + per-user identity | DIY | built-in |
| Memory storage | DIY (Redis / Postgres) | managed |
| Audit log | DIY | built-in |
| Observability | DIY (LangSmith etc.) | built-in |
| Scaling | DIY (k8s) | automatic |
| Cost | infra + dev time | per-invocation |

For the bootcamp's customer-care project: AgentCore gives you a real production-grade demo without spinning up infrastructure.

---

## 4. Quick concept — AgentCore Memory

AgentCore Memory tracks both:

### Short-term (event memory)
- Recent conversation turns
- Task / session state
- Per-user context within an interaction

### Long-term (persistent memory)
- Facts the agent has learned about the user
- Preferences, history
- Survives across sessions

```python
# pseudocode
agent = build_my_agent()
deployed = agentcore.deploy(agent, memory=AgentCoreMemory(strategies=["summarization", "user_preferences"]))

deployed.invoke(user_id="awais", message="I prefer email contact and live in Lahore")
deployed.invoke(user_id="awais", message="What's my city?")    # remembers
```

---

## 5. AgentCore Identity

Per-user auth — the agent acts on a specific user's behalf, scoped to their permissions:
- Token store
- OAuth flows
- Passes auth context to tools

For customer support: each user gets their own context; the agent can't accidentally see another user's data.

---

## 6. AgentCore Browser & Code Interpreter

Two pre-built **sandboxed tools** every production agent eventually needs:

### Browser
The agent can navigate and interact with web pages. Sandboxed (no escape to the host).
Useful for: automation, web scraping, form-filling, research.

### Code Interpreter
The agent can write and execute Python code in a sandboxed VM.
Useful for: data analysis, custom calculations, plotting.

These are notoriously dangerous to build yourself (sandbox escapes, etc.). AgentCore handles the security.

---

## 7. AgentCore Gateway — MCP-compliant tool host

AgentCore Gateway lets you expose internal APIs as MCP tools that any AgentCore agent can use. Think: "publish your HR APIs once, every internal agent can use them."

This pairs perfectly with the **HR onboarding project (Module 9 project 3)** in spirit — same problem, AWS managed solution.

---

## 8. AgentCore Observability

- Traces of every agent execution (tool calls, LLM calls, latencies)
- Metrics (cost, success rate, error rate)
- Audit log (which user did what, when)
- Integration with CloudWatch + X-Ray

For regulated workloads: this is what auditors want to see.

---

## 9. Building an agent for AgentCore

### High-level workflow

1. Write your agent locally in any framework (LangGraph, CrewAI, Strands SDK)
2. Wrap with the AgentCore runtime SDK
3. Deploy via the AWS CLI / SDK
4. Test via console / API
5. Roll out to users

### Sample code (conceptual — actual SDK names may vary)

```python
from bedrock_agentcore import AgentCoreApp, MemoryStrategy
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock

llm = ChatBedrock(model_id="anthropic.claude-sonnet-4")
my_agent = create_react_agent(llm, tools=[customer_lookup, refund_status])

app = AgentCoreApp(
    name="customer-care-agent",
    handler=my_agent,
    memory=MemoryStrategy(short_term=True, long_term=["user_preferences"]),
    identity={"oauth_provider": "cognito"},
    observability=True,
)

# Locally:
app.run()

# Deploy:
# $ agentcore deploy
```

After deploy, you get an HTTPS endpoint scoped per user, with auth, memory, and observability all wired up.

---

## 10. Customer-care AgentCore project (Module 9 project 4)

The full lifecycle Codebasics walks through:

1. **Local development** — build the agent (LangGraph / Strands SDK) calling tools (lookup customer, refund status)
2. **Memory integration** — short-term per-conversation, long-term per-customer (preferences, past complaints)
3. **Secure deployment** — AgentCore deploy command; immutable artifact
4. **Identity** — Cognito-backed auth; tools see the authenticated user
5. **Runtime execution** — invoke from a Streamlit UI hitting the AgentCore endpoint
6. **Observability** — trace each user interaction; alert on anomalies

The lecture stresses **moving beyond demos** to **deployable systems**. That's the takeaway.

---

## 11. AgentCore vs alternatives

| | AgentCore | Self-hosted (LangGraph + your infra) | OpenAI Assistants API |
|---|---|---|---|
| Cloud lock-in | AWS | none | OpenAI |
| Memory & identity | managed | DIY | built-in (limited) |
| Tool sandboxing | built-in | DIY | partial |
| Framework agnostic | yes | yes | OpenAI's runtime |
| Observability | full | DIY (LangSmith etc.) | basic |
| When | production on AWS | full control / multi-cloud | OpenAI-centric apps |

For a portfolio project that demonstrates production skills, AgentCore is gold — recruiters at AWS / cloud-heavy enterprises know it.

---

## 12. Cost & access

AgentCore is part of AWS Bedrock — pay-per-invocation pricing:
- Per LLM token (Bedrock model)
- Per minute of agent runtime
- Per memory storage

You'll need an AWS account + Bedrock access (request via console). Free tier covers small experiments.

---

## 13. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Mixing local memory with AgentCore memory | inconsistent | pick one |
| Forgetting identity scoping | security leak | always pass user context |
| Hardcoded creds in deployed agent | exposed | use IAM roles |
| No observability | can't debug | always enable |
| Trying to run un-managed sandboxes | security holes | use AgentCore Browser / Code Interpreter |

## Self-check

- [ ] What's AgentCore in one sentence?
- [ ] Five capabilities AgentCore provides?
- [ ] Difference between short-term and long-term memory?
- [ ] What's AgentCore Identity for?
- [ ] When use AgentCore vs self-hosted vs OpenAI Assistants?
- [ ] What's the connection between AgentCore Gateway and MCP?
- [ ] Walk through deploying a LangGraph agent to AgentCore.
- [ ] Why does the bootcamp pick AgentCore for the Customer Care project?
