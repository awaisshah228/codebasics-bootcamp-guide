# Project 3 — Agentic AI HR Onboarding (MCP)

## Domain
Onboarding a new hire usually requires 10+ steps across systems:
- Create email account
- Provision laptop / VPN access
- Add to HR systems
- Send welcome packet
- Schedule orientation sessions
- Add to project management tool
- Set up payroll
- Create calendar invites

Each step → a different system, often a different team. **Agentic AI** automates this end-to-end, using **MCP** as the standard interface to all those systems.

## Pattern
- Wrap each existing HR API as an **MCP server**
- Build an LLM-driven **agent** (Claude via the SDK) that plans + orchestrates
- The agent calls multiple MCP tools to complete onboarding
- Human-in-the-loop for critical decisions (sensitive permissions)

---

## Architecture

```
[New hire profile] ──► [LLM Agent (Claude)]
                           │
                           ├──► MCP server: email_system   (create_email)
                           ├──► MCP server: hr_system      (add_employee, set_payroll)
                           ├──► MCP server: equipment      (request_laptop, provision_vpn)
                           ├──► MCP server: calendar       (schedule_orientation)
                           ├──► MCP server: docs           (send_welcome_packet)
                           └──► MCP server: project_tools  (add_to_jira, slack_invite)

                           Each MCP server wraps existing internal APIs.
```

---

## Why this pattern wins

Without MCP: every new internal system requires bespoke agent integration.
With MCP: wrap each system as one MCP server → every existing AI tool can use it.

For Codebasics' bootcamp project: this is the **most modern way** to expose your internal APIs to AI.

---

## Step-by-step

### 1. Identify the "tools" (functions the agent will call)

```
- list_pending_onboardings()
- create_email_account(name, role)
- send_welcome_email(to, template)
- request_laptop(employee_id, model="MacBook Pro")
- provision_vpn(employee_id)
- add_to_hr_system(employee_data)
- schedule_orientation(employee_id, start_date)
- add_to_jira(employee_id)
- invite_to_slack(email, channels)
- check_status(employee_id)
```

### 2. Build MCP servers (one per system, or one combined)

```python
# hr_mcp_server.py
from mcp.server.fastmcp import FastMCP
from typing import Literal

mcp = FastMCP("HR Onboarding System")

@mcp.tool()
def list_pending_onboardings() -> list[dict]:
    """List employees pending onboarding (status='pending')."""
    return query_hr_db("SELECT id, name, role, start_date FROM employees WHERE status = 'pending'")

@mcp.tool()
def create_email_account(name: str, role: str) -> dict:
    """Create a new corporate email account.
    USE WHEN: A new employee needs an email.
    Returns: {email, temp_password}.
    """
    # call your existing email-provisioning API
    email = generate_email(name)
    pw = create_in_email_system(email, role)
    return {"email": email, "temp_password": pw}

@mcp.tool()
def send_welcome_email(to: str, template: Literal["standard", "engineering", "exec"] = "standard") -> dict:
    """Send a welcome email to the new employee."""
    return send_via_email_api(to, template)

@mcp.tool()
def request_laptop(employee_id: str, model: str = "MacBook Pro") -> dict:
    """Request a laptop from IT."""
    return create_it_ticket(employee_id, kind="laptop", model=model)

# ... more tools

if __name__ == "__main__":
    mcp.run()
```

### 3. The agent — Claude with MCP

```python
import anthropic, asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

claude = anthropic.Anthropic()

async def onboard_one():
    server = StdioServerParameters(command="python", args=["hr_mcp_server.py"])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            anthropic_tools = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            } for t in tools]

            messages = [{"role": "user", "content":
                         "Onboard the next pending employee. Complete all required steps."}]

            while True:
                resp = claude.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    tools=anthropic_tools,
                    messages=messages,
                )

                if resp.stop_reason == "end_turn":
                    print("Done.\n", resp.content[0].text if resp.content else "")
                    break

                # Process tool uses
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        result = await session.call_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result.content),
                        })

                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})

asyncio.run(onboard_one())
```

The agent will:
1. Call `list_pending_onboardings()`
2. Pick the first employee
3. Call `create_email_account` → `request_laptop` → `provision_vpn` → ... in sequence
4. Reason about which steps are needed for that role
5. Output a final summary

---

## 4. Adding human-in-the-loop

For sensitive actions (provisioning admin access, sending welcome with sensitive info):

### Option A — Pause for approval
```python
@mcp.tool()
def grant_admin_access(employee_id: str, reason: str) -> dict:
    """Grant admin access. Requires manager approval (out-of-band)."""
    request_id = create_approval_request(employee_id, reason)
    return {"request_id": request_id, "status": "pending_approval"}
```

The agent calls; the tool returns `pending_approval`. Manager approves separately. The agent moves on or waits.

### Option B — LangGraph with `interrupt_before`
Insert a checkpoint before sensitive tools. Resume after human review.

---

## 5. Streamlit dashboard

A simple UI lets HR run / monitor / approve onboardings:
- "Onboard next" button
- Live tool-call log streaming
- Pending approval queue
- Per-employee status

```python
import streamlit as st

st.title("🤖 HR Onboarding Agent")

if st.button("Onboard next pending employee"):
    placeholder = st.empty()
    for log_entry in onboard_one_streaming():
        placeholder.text_area("Log", log_entry, height=400)
```

---

## 6. Eval & safety

For agents touching real systems:
- **Dry-run mode** for testing (tools log instead of executing)
- **Per-tool approval flags** in code
- **Audit log** of every tool call to a database
- **Rate limits** on tool calls per session
- **Hard cap** on total cost per onboarding

For production: route every action through your existing IT change-management workflow before it actually executes.

---

## 7. Why this is the most "real-world" portfolio project

- Demonstrates **MCP** — the most relevant 2024–2025 standard
- Demonstrates **agent orchestration** — concrete value, not just a chatbot
- Demonstrates **system integration** — wrapping real APIs
- Recruiters see: "Awais built an actual workflow agent, not a demo"

This is the project that gets you noticed by AI-engineering hiring managers.

---

## Repo deliverables

```
hr-onboarding-mcp/
├── mcp_servers/
│   ├── email_server.py
│   ├── hr_server.py
│   ├── equipment_server.py
│   └── calendar_server.py
├── agent/
│   └── onboard.py
├── ui/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_onboarding_flow.py
├── README.md                  # setup, MCP server config, demo
└── requirements.txt
```

---

## Self-check

- [ ] Do I have ≥3 MCP servers wrapping different systems?
- [ ] Does the agent successfully complete a full onboarding (with mocked APIs)?
- [ ] Did I implement human-in-the-loop for sensitive actions?
- [ ] Audit log of every tool call?
- [ ] Cost / rate limits in place?
- [ ] README explains: what MCP is, why this design, how to extend?
- [ ] Streamlit dashboard for observability?
- [ ] LinkedIn post explaining MCP + showing the agent in action?
