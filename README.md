# Integrated MAS Implementation
Reference: Week 2 case study 
Deliverable: JUpyter Notebook (.ipynb file) -to be submitted in Github repo

Implementation: A local-memory multi-agent system built with LangGraph, CrewAI(planner, executor and reviewer agents), PydanticAI, Mem0, and Groq 

## Overview

This project demonstrates a multi-agent architecture that follows a structured workflow:

- Planner creates a plan
- Executor performs the task
- Reviewer validates the result
- LangGraph controls the state flow
- A safety guardrail stops execution when a retry budget is exceeded

It is designed to be run in a Colab/Jupyter environment and uses Groq as the LLM provider so it can operate without requiring a local Ollama installation.

## Key Features

- LangGraph-based orchestration
- CrewAI worker agents
- PydanticAI strict validation schemas
- Persistent local memory via Mem0 + SQLite
- Groq API integration for LLM access
- Safe fallback when execution exceeds the budget

## Architecture

The system uses a three-step agent loop:

1. Planner
   - breaks the task into ordered steps
2. Executor
   - carries out the task according to the plan
3. Reviewer
   - approves or rejects the output

The graph keeps track of `execution_count` and routes to a fallback state when the count exceeds the configured limit.

## Execution Guardrail

The notebook enforces the required guardrail:

- `MAX_EXECUTIONS = 3`
- If `execution_count > MAX_EXECUTIONS`, the flow switches to a safe fallback node
- This prevents infinite loops and ensures controlled termination

## Memory Layer

Memory is managed locally with:

- Mem0 OSS
- SQLite for history persistence
- Chroma as the vector store
- Hugging Face embeddings for local semantic retrieval

This allows memory to persist across sessions for the same `user_id`.

## Output Validation

The project uses Pydantic models to validate agent outputs before the workflow continues:

- `PlannerOutput`
- `ExecutorOutput`
- `ReviewerOutput`

This keeps the system deterministic and prevents malformed or unsafe JSON responses from being used.

## Project Files

- `Integrated_MAS_implementation_plan (1).ipynb` — full implementation notebook
- `document_analyzer.py` — supporting Python code
- `document_analyzer.yaml` — config file
- `doc_boarding.json` and `doc_boarding.xlsx` — related project data

## Setup

Install the required Python packages:

```bash
pip install "crewai[litellm]" "langgraph" "pydantic-ai" "mem0ai" "groq" "sentence-transformers"
```

Set the environment variable:

```bash
GROQ_API_KEY=your_key_here
```

In Colab, this is typically stored as a secret named `GROQ_API_KEY`.

## Run the Workflow

The notebook uses async graph execution because the system is designed to work naturally with Jupyter/Colab's running event loop.

The main flow is:

```python
result_state = await app.ainvoke(initial_state, config={"recursion_limit": 25})
```

This runs the Planner → Executor → Reviewer flow and returns the final state.

## Notes

- A compatibility fix is included to strip `cache_breakpoint` values before sending requests through LiteLLM, avoiding a Groq compatibility issue.
- The notebook also includes a guardrail demonstration where the reviewer always rejects to show the fallback stopping behavior.

## Summary

This project is a reusable proof-of-concept for a safe, stateful, multi-agent orchestration workflow with:

- strong validation,
- local persistence,
- model-agnostic orchestration,
- and execution budget protection.

It is intended as a practical implementation of a controlled autonomous multi-agent system.
