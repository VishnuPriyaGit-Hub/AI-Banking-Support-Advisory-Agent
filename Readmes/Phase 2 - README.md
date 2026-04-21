# Phase 2: Fresh Baseline Agent

## What Was Built

- A Python-based banking support agent that accepts user input from the CLI and the Streamlit UI.
- A simple rule-based response engine with no LLM calls and no external API dependency.
- Persona-aware behavior aligned to the Phase 1 README roles:
  - Customer
  - Branch Manager
  - Risk & Compliance Officer
  - Admin
  - Customer Support Agent
- JSONL logging for sample interactions and normal runs.

## How The Fresh Baseline Works

- The user selects or logs in with a Phase 1 persona.
- The agent reads the latest user query.
- A rule engine classifies the request into categories such as:
  - `safe`
  - `ambiguous`
  - `disallowed`
  - `high_risk`
  - `live_data`
  - `follow_up`
  - `fallback`
- The selected persona uses a fixed response template for that category.
- The result is logged with the input, output, and category.

## Key Changes From The Older Baseline

- Removed prompt-based response framing from the agent flow.
- Removed leftover LLM-oriented response structures from the shared models.
- Refreshed persona behavior so it maps directly to the Phase 1 README.
- Cleaned old log data and generated fresh sample interactions.

## Files Involved

- `app/agents/baseline_agent.py`
- `app/models/agent.py`
- `app/ui/streamlit_app.py`
- `logs/sample_interactions.jsonl`
- `logs/baseline_agent_runs.jsonl`

## How To Run

CLI:

```powershell
python -m app.agents.baseline_agent
```

Demo with fresh sample logs:

```powershell
python -m app.agents.baseline_agent --demo
```

CLI with log reset:

```powershell
python -m app.agents.baseline_agent --reset-log
```

Streamlit UI:

```powershell
python -m streamlit run streamlit_app.py
```

## Logging

- Demo runs write to `logs/sample_interactions.jsonl`
- Interactive runs write to `logs/baseline_agent_runs.jsonl`
- Each log entry includes:
  - input role and query
  - generated response
  - category
  - persona metadata

## Outcome

This Phase 2 version is now a clean, simple, rule-based banking agent that accepts user input, generates template responses, and logs sample interactions without using any LLM call.
