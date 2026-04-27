# Project 4 — Customer Care AI Agent on Bedrock AgentCore

## Domain
A telecom / SaaS company has thousands of daily customer-care queries:
- "What's my bill this month?"
- "Cancel my subscription."
- "Why is my service slow?"
- "Refund the last charge."

Build a **production-grade AI agent** running on **Amazon Bedrock AgentCore** that handles 80% of these autonomously and escalates the rest.

## Pattern
- Multi-tool agent with auth-scoped data access
- Persistent memory (per-customer history)
- AgentCore for runtime, identity, observability
- Streamlit UI calling the deployed AgentCore endpoint

This is the **most enterprise-grade** project in the bootcamp.

---

## Architecture

```
[Streamlit UI]
   │ HTTPS (Cognito auth)
   ▼
[AgentCore Runtime — your agent]
   │
   ├──► AgentCore Memory  (short-term + long-term)
   ├──► AgentCore Identity (per-user OAuth context)
   ├──► AgentCore Browser (sandboxed web actions)
   ├──► AgentCore Code Interpreter (sandboxed Python)
   │
   └──► Tools (your business APIs):
         - get_customer_profile(user_id)
         - get_recent_bill(user_id)
         - check_service_status(user_id)
         - issue_refund(user_id, amount, reason)
         - escalate_to_human(user_id, summary)
```

---

## Step-by-step build

### 1. Define the agent's tools

```python
from langchain_core.tools import tool

@tool
def get_customer_profile(user_id: str) -> dict:
    """Get the authenticated customer's profile.
    USE WHEN: you need plan, contact info, or account status."""
    return crm_api.get_profile(user_id)

@tool
def get_recent_bill(user_id: str) -> dict:
    """Get the most recent bill for the authenticated customer."""
    return billing_api.get_bill(user_id, latest=True)

@tool
def check_service_status(user_id: str) -> dict:
    """Check if the customer's service has any active outages."""
    return network_api.status_for(user_id)

@tool
def issue_refund(user_id: str, amount: float, reason: str) -> dict:
    """Issue a refund up to $50 for the authenticated customer.
    REQUIRES MANAGER APPROVAL for amounts > $50.
    """
    if amount > 50:
        return {"status": "needs_approval", "approval_id": create_approval(user_id, amount, reason)}
    return billing_api.refund(user_id, amount, reason)

@tool
def escalate_to_human(user_id: str, summary: str) -> dict:
    """Hand off to a human agent with a context summary.
    USE WHEN: customer requests human, you can't resolve, or sensitive issue."""
    return support_api.escalate(user_id, summary)
```

### 2. Build the agent (LangGraph)

```python
from langchain_aws import ChatBedrock
from langgraph.prebuilt import create_react_agent

llm = ChatBedrock(model_id="anthropic.claude-sonnet-4-v1")

agent = create_react_agent(
    llm,
    tools=[get_customer_profile, get_recent_bill, check_service_status,
            issue_refund, escalate_to_human],
)
```

### 3. Wrap with AgentCore SDK

```python
# This is conceptual; actual API names may differ.
from bedrock_agentcore import AgentCoreApp, MemoryStrategy, IdentityConfig

app = AgentCoreApp(
    name="customer-care-agent",
    handler=agent,
    memory=MemoryStrategy(
        short_term=True,                                # convo turns
        long_term=["customer_preferences", "past_issues"],
    ),
    identity=IdentityConfig(provider="cognito"),
    observability=True,
)

# Run locally
app.run()
```

### 4. Deploy

```bash
agentcore deploy
```

You get an HTTPS endpoint. Auth is Cognito-backed per-user.

### 5. Streamlit UI

```python
import streamlit as st
import requests

st.title("📞 Customer Care")

# user_id from your auth flow (Cognito-backed)
user_id = st.session_state.get("user_id")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if msg := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": msg})
    with st.chat_message("user"): st.write(msg)

    resp = requests.post(
        f"{AGENTCORE_ENDPOINT}/invoke",
        headers={"Authorization": f"Bearer {get_cognito_token(user_id)}"},
        json={"input": msg, "user_id": user_id},
    )
    answer = resp.json()["output"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"): st.write(answer)
```

### 6. Memory in action

After a few interactions, the agent's long-term memory holds:
- "Awais prefers email contact"
- "Awais has had 2 service complaints in the past month"
- "Awais's plan: Premium"

When Awais asks about service quality, the agent:
- Pulls relevant memory automatically
- Tailors response based on context

---

## 7. Observability

AgentCore's built-in tracing:
- Every customer interaction → full trace
- Tool calls + LLM messages + latency + cost per turn
- Audit log of every refund issued (for compliance)
- Alerts on error rate spikes

Visible in the AgentCore console + CloudWatch.

---

## 8. Eval suite (50 customer scenarios)

```python
scenarios = [
    {"query": "What's my bill?", "expected_tools": ["get_recent_bill"]},
    {"query": "Why is my internet slow?", "expected_tools": ["check_service_status"]},
    {"query": "Refund my last charge of $25", "expected_tools": ["issue_refund"]},
    {"query": "I need to talk to a human", "expected_tools": ["escalate_to_human"]},
    # ... 50+
]
```

For each: run the agent (in dry-run / mock mode), inspect the trace, score:
- Did it call the right tools?
- Was the response helpful?
- Did it escalate when it should have?
- Cost per query?

Track regressions across deploys.

---

## 9. Safety & guardrails

- **Refund cap**: $50 hard limit; bigger needs human approval
- **Authentication required**: tools see only the authed user's data
- **PII redaction** in logs
- **Rate limit** per customer per hour
- **Refusal of harmful instructions** (jailbreaks)
- **Escalation triggers**: legal threats, repeated frustration, irreversible actions

---

## 10. Why this project lands a job

It demonstrates:
- Production AI agent architecture
- AWS / cloud experience
- Identity + auth handling
- Memory design
- Observability + audit
- Safety / guardrails

These are *exactly* what AI-engineering job descriptions list. A working AgentCore demo on your portfolio is rare and impressive.

---

## Repo deliverables

```
customer-care-agentcore/
├── src/
│   ├── tools.py                    # tool definitions
│   ├── agent.py                    # LangGraph agent
│   ├── deploy.py                   # AgentCore wrapping + deploy
│   └── ui.py                       # Streamlit UI
├── evals/
│   ├── scenarios.json
│   ├── run.py
│   └── traces/                     # captured runs
├── infra/
│   ├── cognito-setup.md
│   └── iam-policies.md
├── docs/
│   ├── architecture.md
│   ├── safety-guardrails.md
│   └── memory-design.md
├── README.md                        # demo, screenshots, eval results
└── requirements.txt
```

---

## Self-check

- [ ] Agent reliably handles 80% of test scenarios?
- [ ] Refunds capped + manager approval flow working?
- [ ] Memory persists across customer interactions?
- [ ] Observability traces visible in AgentCore console?
- [ ] Identity ensures tools see only the authed user's data?
- [ ] Streamlit UI functional + auth-protected?
- [ ] 50-scenario eval suite + results in README?
- [ ] LinkedIn post: "Built a production AI agent on AWS AgentCore — end-to-end demo + lessons"?
