# Foundations 1 — Intro to Gen AI & Agentic AI

## Lectures covered
- Introduction to Gen AI & Agentic AI
- Application of Gen AI & Agentic AI
- Gen AI Application Development Steps

---

## 1. Generative AI vs Agentic AI

### Generative AI
Models that **produce** content: text, code, images, audio, video.
- ChatGPT generating an email
- DALL-E generating an image
- GitHub Copilot suggesting code
- Suno generating music

### Agentic AI
Systems that **take actions** in the world via LLMs:
- Research → search the web → summarize
- Read your email, draft replies, send them
- Analyze a database, write SQL, execute, report
- Onboard an employee end-to-end (HR project — Module 9 project 3)

The distinction: **Gen AI passively responds; Agentic AI plans and acts.**

A modern application usually combines both: an agent that talks (Gen AI) and acts (tools).

---

## 2. The Gen AI / Agentic AI application landscape (use cases)

### Customer-facing
- Chatbots / customer support
- Virtual assistants
- Personalized recommendations
- Search ("ask the docs")

### Internal productivity
- Code assistants
- Email triage / drafting
- Meeting summarization
- Knowledge management

### Vertical applications
- **Real estate** — property assistants (project 1 in this module)
- **E-commerce** — shopping bots (project 2)
- **HR** — onboarding agents (project 3)
- **Healthcare** — clinical decision support, medical Q&A (Virtual Internship 2)
- **Finance** — research assistants, compliance checkers
- **Legal** — contract review, clause extraction

### Developer / AI-eng building blocks
- RAG over private docs
- Code generation / SQL generation
- Function-calling APIs
- Multi-agent orchestration

---

## 3. The Gen AI app development steps (high-level)

1. **Frame the problem** — what does the user ask, what does the system answer, what's the success metric?
2. **Choose the LLM** — closed (Claude / OpenAI / Gemini) vs open (Llama / Mistral); per cost / latency / privacy
3. **Design the prompt** — system prompt, user template, output format
4. **Add retrieval if needed** — RAG over your private docs / DB
5. **Add tools if needed** — let the LLM call APIs / DB / web search
6. **Plan for failure** — hallucinations, prompt injection, rate limits, cost spikes
7. **Build a UI** — Streamlit demo or full frontend
8. **Evaluate** — golden questions, LLM-as-judge, human review
9. **Deploy** — FastAPI / serverless / managed (Bedrock AgentCore)
10. **Monitor + iterate** — log all calls, track cost, cache repeated queries

---

## 4. Closed vs Open-source LLMs

| | Closed (API) | Open (run locally) |
|---|---|---|
| Examples | Claude, GPT, Gemini | Llama, Mistral, Qwen, Phi |
| Quality | usually higher | catching up |
| Cost | per-token API fee | infrastructure + GPU |
| Privacy | data sent to vendor (with enterprise terms) | full control |
| Latency | 200ms–5s typical | depends on hardware |
| Customization | prompt + fine-tune (limited) | full fine-tune, distill, quantize |
| Default for | most production apps in 2025 | privacy/sovereignty-critical, very high volume |

For this bootcamp's projects: **API LLMs** for Gen AI projects (Claude / OpenAI), demonstrated on managed services (AgentCore).

---

## 5. Quick taste — call an LLM

### OpenAI
```python
from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize: ..."}],
)
print(resp.choices[0].message.content)
```

### Anthropic (Claude)
```python
from anthropic import Anthropic
client = Anthropic()
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize: ..."}],
)
print(resp.content[0].text)
```

### Local (Ollama)
```bash
ollama run llama3
```
```python
import requests
r = requests.post("http://localhost:11434/api/generate",
                   json={"model": "llama3", "prompt": "Hello"})
```

For 95% of bootcamp projects: API-based LLMs are simplest. Move to open models later for portfolio depth.

---

## 6. The "AI engineer" stack in 2025

```
┌──────────────────────────────────────────────────────────┐
│ APPLICATION (chatbot, assistant, agent)                  │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ ORCHESTRATION (LangChain / LangGraph / CrewAI / DSPy)    │
└────────────┬─────────────────────────────────────────────┘
             │
   ┌─────────┼──────────┬─────────────────┐
   ▼         ▼          ▼                 ▼
┌──────┐ ┌──────┐  ┌──────────┐  ┌──────────────┐
│ LLM  │ │ RAG  │  │  TOOLS   │  │  MEMORY      │
│ APIs │ │ vec  │  │ APIs/DB  │  │  short/long  │
└──────┘ └──────┘  └──────────┘  └──────────────┘
   │
┌──▼─────────────┐
│ EVAL + OBSERVE │
│ ragas, langsmith│
│ langfuse, etc. │
└────────────────┘
```

The bootcamp's projects walk you through each layer.

---

## 7. Free APIs / sandboxes for bootcamp

- **OpenAI** — $5 free credit on signup; cheap for prototyping
- **Anthropic Claude** — free trial credits via console
- **Google Gemini** — free tier in AI Studio
- **Groq** — free tier; very fast hosted Llama/Mixtral
- **Together AI / Fireworks** — pay-as-you-go open models
- **Hugging Face Inference API** — free tier

For local: **Ollama** runs Llama-3.1 8B on a 16GB Mac just fine.

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| API key in repo | leaked credentials | use `.env` + `.gitignore` |
| Calling LLM in a loop without limit | runaway cost | set max trials + cap on tokens |
| Treating LLM output as truth | hallucinations | always verify or constrain |
| No retry / timeout | flaky reliability | implement exponential backoff |
| Not logging prompts/responses | can't debug | log everything (with privacy) |

## Self-check

- [ ] Difference between Gen AI and Agentic AI in one sentence?
- [ ] Name 3 verticals where Agentic AI is being deployed today.
- [ ] Walk through the 10 Gen AI dev steps.
- [ ] When pick a closed API LLM vs an open model?
- [ ] What's the AI-engineer stack — list the layers.
- [ ] Free LLM APIs you can use right now?
- [ ] What's the #1 production pitfall to avoid?
- [ ] How do you store an API key safely?
