# Module 9 — Gen AI & Agentic AI

> **Status**: 🔒 Locked
> **Tools**: LangChain · LangGraph · Amazon Bedrock AgentCore · Claude MCP SDK · CrewAI · ChromaDB · Streamlit
> **Projects**: Real Estate Assistant (RAG) · E-Commerce Chatbot · Agentic AI HR Onboarding · Customer Care Agent (AgentCore)

## Why this module exists

The 2025 data role is increasingly an **AI-engineering** role: building applications that wrap LLMs with retrieval, tools, memory, and orchestration. This is where Codebasics' bootcamp deviates from older curricula — they cover the **modern AI stack** end-to-end with 4 production-style projects.

## Curriculum (verbatim from brochure)

- Introduction to Gen AI & Agentic AI
- Application of Gen AI & Agentic AI
- Large Language Models
- AI Agent with Custom tools
- Context Window · Temperature
- Model Context Protocol (MCP)
- Agentic AI Evaluation
- Gen AI Application Development Steps
- Hallucinations, Security, Cost
- Vector Database
- RAG (Retrieval Augmented Generation)
- Streamlit UI Development
- Multi Agent Systems
- Zero-Shot, One-Shot, Few-Shot Prompting
- LangChain Installation and Setup
- Calling LLM from LangChain
- Prompt Templates & Chains
- Chromadb · Metadata Filtering
- Fine Tuning an LLM
- SQLite Database Integration
- LangGraph Crash course
- CrewAI crash course
- Amazon Bedrock AgentCore

## Folder layout

```
09-gen-ai-agentic-ai/
├── README.md
├── 01-foundations/
│   ├── README.md
│   ├── 01-intro-applications.md
│   ├── 02-llm-fundamentals.md
│   ├── 03-context-temperature.md
│   ├── 04-prompt-engineering.md
│   └── 05-hallucinations-security-cost.md
├── 02-rag/
│   ├── README.md
│   ├── 01-vector-databases.md
│   ├── 02-rag-fundamentals.md
│   ├── 03-chromadb-metadata.md
│   └── 04-fine-tuning.md
├── 03-orchestration/
│   ├── README.md
│   ├── 01-langchain.md
│   ├── 02-langgraph.md
│   ├── 03-crewai.md
│   ├── 04-mcp.md
│   └── 05-amazon-bedrock-agentcore.md
├── 04-agents/
│   ├── README.md
│   ├── 01-agent-fundamentals.md
│   ├── 02-multi-agent-systems.md
│   └── 03-agentic-evaluation.md
└── 05-projects/
    ├── README.md
    ├── 01-real-estate-rag.md
    ├── 02-ecommerce-chatbot.md
    ├── 03-agentic-onboarding-mcp.md
    └── 04-customer-care-agentcore.md
```

## Module-level goal

After this module:
- Build production-quality RAG systems
- Wire up agents with custom tools, plans, and memory
- Use LangChain / LangGraph / CrewAI fluently
- Understand MCP and how it standardizes tool integration
- Deploy on Amazon Bedrock AgentCore with proper observability
- Have 4 portfolio Gen-AI projects across real estate, e-commerce, HR, and customer care

## Module self-check

- [ ] When fine-tune vs prompt vs RAG?
- [ ] Walk through a RAG query end-to-end.
- [ ] What's the difference between ReAct and Plan-and-Execute?
- [ ] What's prompt injection and how mitigate it?
- [ ] What's MCP and what problem does it solve?
- [ ] Pick a vector DB for a small startup vs an enterprise.
- [ ] How do you evaluate a RAG system without humans in the loop?
- [ ] Why is LangGraph preferred over plain LangChain for complex agents?
