# Agents 3 — Agentic AI Evaluation

## Lectures covered
- Agentic AI Evaluation

---

## 1. Why agent eval is hard

Classical ML eval is easy: you have labels, you compute accuracy. Agent eval is harder because:
- Outputs are open-ended (no single "right" answer)
- Multi-step decisions matter (good final answer with terrible reasoning is fragile)
- Tools have side effects (can't replay deterministically)
- Different runs can give different results
- Cost / latency / safety are first-class concerns

You need a **portfolio of evaluation methods**, not a single number.

---

## 2. Three evaluation layers

### Layer 1 — Component evals
Test individual building blocks:
- Does retrieval find the right docs?
- Does the tool wrapper return correct shapes?
- Does the prompt produce structured JSON?

### Layer 2 — Trajectory evals
For each task, inspect the **path** the agent took:
- Did it call the right tools in the right order?
- Did it avoid forbidden actions?
- Did it loop unnecessarily?

### Layer 3 — Outcome evals
The final answer / state:
- Did the user's request get fulfilled?
- Was the answer factually correct?
- Was the format right?

A good test suite covers all three.

---

## 3. Trajectory evals — the agent-specific bit

For each task, capture a trace:
```json
{
  "task": "Find the next pending onboarding and create their accounts",
  "steps": [
    {"tool": "list_pending_onboardings", "args": {}, "result": [{"id":1, "name":"Awais"}]},
    {"tool": "create_email_account", "args": {"name":"Awais"}, "result": "ok"},
    {"tool": "send_welcome_email", "args": {"to":"awais@..."}, "result": "ok"}
  ],
  "final_answer": "Awais has been onboarded."
}
```

### Assertions you can run on traces
- Tool X must be called before tool Y
- Tool Z must NOT be called
- Total tool calls ≤ N
- No tool is called > 3 times in a row
- Cost ≤ $X per task

### Tools
- **LangSmith** — captures traces, has eval primitives
- **Langfuse** — open-source equivalent
- Custom — log to DB, query for assertions

---

## 4. Outcome evals — LLM-as-judge

For open-ended outputs, use a strong LLM to score:
```python
def grade(question, answer, expected):
    prompt = f"""
    Question: {question}
    Expected answer: {expected}
    Actual answer: {answer}

    Score 1-5: how well does the actual answer cover what's expected?
    Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}
    """
    return judge_llm.invoke(prompt)
```

### Best practices
- Use the strongest model available as judge (Claude Opus / GPT-4)
- Provide clear rubric (1=garbage, 3=acceptable, 5=excellent)
- Few-shot the judge with examples of each score
- Average over multiple judge runs (or seeds)

### Caveats
- Judge has its own biases
- Don't have the same model self-judge unless you've validated
- Cost: judge calls add up; cheaper to LLM-judge a sample

---

## 5. RAGAS for RAG-based agents

If your agent does RAG, use RAGAS metrics:
- **Faithfulness** — does the answer rely only on retrieved context?
- **Answer relevance** — does it address the question?
- **Context precision / recall** — was the right info retrieved?

```bash
pip install ragas
```
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

dataset = ...   # questions, answers, retrieved_contexts, ground_truths
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
```

---

## 6. Public benchmarks for agentic AI

Worth knowing for context:

- **SWE-Bench** — agents solve real GitHub issues (popular for code agents)
- **WebArena** — agents complete tasks on websites
- **GAIA** — general assistant benchmark; multi-hop reasoning + tool use
- **MMLU** — multitask language understanding (knowledge eval)
- **AgentBench** — multi-domain agent eval
- **τ-Bench** (Tau-Bench) — customer-service agent eval
- **MMMU** — multimodal multitask

Track your own benchmark numbers vs SOTA. For a portfolio: cite "model X achieves 87% on SWE-Bench, my agent achieves 35% on a similar task."

---

## 7. Building a custom eval suite (the bootcamp project way)

For Codebasics' projects, build a small suite per project:

### For the real-estate RAG (project 1)
- 30 user queries (varied: budget, location, type, edge cases)
- Per query: expected matching listings (gold)
- Metrics: top-5 hit rate, faithfulness, answer relevance

### For the e-commerce chatbot (project 2)
- 20 conversation flows (FAQs, product search, refund, escalate)
- Per flow: expected outcome (correct product? correct route?)
- Metrics: outcome correctness, # of clarification turns

### For the agentic onboarding (project 3)
- 10 tasks (different employee profiles)
- Per task: expected tool sequence (gold trace)
- Metrics: trace match rate, completion rate, # of unused tools

### For the customer-care agent (project 4)
- 50 customer scenarios across topics
- Metrics: resolution rate, escalation rate, time-to-resolution, customer-satisfaction (LLM-judged)

A repo with `evals/` directory + `pytest` for assertions makes this real.

---

## 8. CI/CD for prompts and agents

Treat prompts like code:
```yaml
# .github/workflows/eval.yml
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python evals/run.py        # runs the agent on golden tasks, scores
      - run: python evals/check_thresholds.py  # fails if quality regressed
```

This prevents prompt regressions from sneaking into prod.

---

## 9. Cost & latency evals

Track per task:
- Total tokens (input + output)
- Total $ cost
- Wall-clock latency
- Number of tool calls

Set thresholds; fail builds that exceed.

A regression in prompt size or tool design can 10× your bill. Eval catches it.

---

## 10. Safety / red-team evals

For agents touching sensitive data or making consequential decisions:

- **Prompt injection probes** — inputs designed to trick the agent
- **Jailbreak attempts** — encourage refusal of harmful requests
- **PII leakage** — does the agent leak data it shouldn't?
- **Authorization bypass** — does it do things outside the user's permissions?

Frameworks: **Garak**, **PyRIT**, **Llama Guard**, custom probe sets.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Manual eval only | doesn't scale; biased | invest in automated evals early |
| Single golden answer | misses valid alternatives | use rubric / LLM judge |
| Eval only at end | regressions sneak in | run on every PR |
| LLM judge same as production model | bias | use a different / stronger judge |
| No trajectory eval | wrong reasoning hidden by lucky outputs | always evaluate the path |
| Ignoring cost / latency | bill explosion | always include them |

## Self-check

- [ ] Three layers of agent eval?
- [ ] What's a trajectory eval and what do you assert on?
- [ ] Walk through LLM-as-judge with rubric.
- [ ] What does RAGAS measure?
- [ ] Three public agentic benchmarks?
- [ ] Build an eval suite for a chatbot.
- [ ] How do you CI-test a prompt change?
- [ ] Two safety evals to add for sensitive agents.
