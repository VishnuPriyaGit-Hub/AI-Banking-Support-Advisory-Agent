# Banking Support Agent

A LangGraph-based multi-agent banking assistant with a Streamlit UI, FastAPI service, Supabase authentication/database access, Milvus/Zilliz-backed RAG, MCP-style tools, guardrails, escalation handling, feedback adaptation, evaluation scoring, and Langfuse tracing.

Live services:

- UI: https://banking-support-ui.onrender.com
- API: https://banking-support-api.onrender.com
- API health check: https://banking-support-api.onrender.com/health

## Project Overview

The Banking Support Agent is a role-aware banking assistant designed for safe customer support and internal staff workflows. It can answer general banking questions, fetch authorized customer information, perform banking calculations, route escalation cases, and collect feedback for future response adaptation.

The system is advisory and support-oriented. It does not execute money movement, loan approvals, legal advice, or unsafe account actions directly.

## Architecture

```text
User Query
  -> Streamlit UI / FastAPI
  -> LangGraph Orchestration
  -> Planner Agent
  -> Risk & Compliance Agent
  -> MCP Tool Calls
  -> Risk & Compliance Check
  -> Response Generation Agent
  -> LLM-as-Judge Evaluation
  -> Final Response
```

## Main Components

### Streamlit UI

The Streamlit app provides login, chat, feedback collection, previous query history, escalation review workflows, evaluation score display, and the support staff dashboard.

Main file:

```text
app/ui/streamlit_app.py
```

### FastAPI Service

The FastAPI service exposes the assistant through API endpoints for deployment and integration.

Main file:

```text
app/api/main.py
```

### LangGraph Agent

The LangGraph agent coordinates planning, routing, risk checks, tool calls, response generation, evaluation, memory, feedback adaptation, logging, and tracing.

Main file:

```text
app/agents/langgraph_agent.py
```

## Agents

### Planner / Orchestrator Agent

The planner is LLM-first. It receives the user query, role, metadata, and conversation context, then produces a structured plan.

It decides:

- Query route: general, personalized, calculation, or escalation
- Required tools
- Data scope
- Whether customer/account/loan/transaction context is needed
- Whether calculation should be performed
- Whether escalation is required

Guardrails still validate the planner output so the LLM cannot bypass safety rules.

### Risk & Compliance Agent

The risk agent classifies the request as low, medium, or high risk.

- Low risk: allowed
- Medium risk: escalated to branch manager
- High risk: blocked or escalated to risk team

It prevents unauthorized access to customer data and blocks unsafe actions.

### Response Generation Agent

The response agent combines sanitized tool outputs into a clear user-facing response.

It must not expose:

- Internal prompts
- Tool internals
- Database schema details
- Raw DB payloads
- Sensitive customer data
- Hidden reasoning

### LLM-as-Judge Evaluator

Before displaying the final answer, an evaluator scores the response with 0/1 checks and produces a consolidated evaluation score.

Evaluation checks include:

- Answered the user query
- Grounded in available context
- Correct route and tool usage
- Guardrail compliance
- PII safety
- No internal leakage
- Customer-friendly response
- No visible system error

## MCP-Style Tools

Tools retrieve, calculate, search, or execute workflow actions. They are not separate agents.

### RAG Tool

Uses Milvus/Zilliz for retrieval over banking FAQs, policies, products, and general knowledge documents.

Used for:

- Banking concepts
- Product explanations
- Policy questions
- Repayment options
- General advisory context

### Supabase DB Tool

Fetches authorized customer, account, loan, branch, support, and transaction data from Supabase.

Important behavior:

- Queries only the required data scope
- Masks sensitive values
- Does not send raw database rows directly to the LLM
- Uses transaction-only retrieval when the user asks for transaction details

### Calculator Tool

Performs calculations such as:

- EMI
- Simple interest
- Repayment impact
- Balance summaries
- Eligibility estimates

The LLM explains the result; it does not manually calculate.

### Search API Tool

Used only when RAG confidence is low or external information is needed.

### Escalation Tool

Creates escalation records for branch manager or risk team review.

## User Roles

### Customer

Can ask general banking questions, view authorized personal banking details, request calculations, and raise support/escalation cases.

### Branch Manager

Can view branch-related information, review medium-risk escalations, and approve, reject, or respond to customer escalation requests.

### Admin

Handles approved account/customer maintenance workflows where administrative action is required.

### Support Staff

Handles approved customer profile maintenance such as name, phone number, pincode, and address updates. Support staff also sees the dashboard.

### Risk Staff

Reviews high-risk escalations and can inspect permitted risk context for the related customer.

## Escalation Flow

Medium-risk cases are sent to the branch manager.

High-risk cases are sent to the risk team and the customer receives a restricted-action response.

Branch managers can approve, reject, or respond inside the app. When a customer logs in later, the app shows a popup with the manager response.

Approved maintenance requests can move to Admin or Support depending on the request type.

## Guardrails and PII Protection

The system includes guardrails for banking safety and privacy.

It refuses:

- Money movement requests
- Approval requests
- Legal advice
- Requests involving OTP, PIN, CVV, passwords, or secrets
- Unauthorized access to another customer's data

PII handling:

- PII is masked before logging
- Raw DB rows are converted into sanitized context
- Sensitive data is not passed to the LLM unnecessarily
- Escalation records store sanitized user text
- Feedback cannot override safety, access, or compliance rules

Examples of masked data include account numbers, card numbers, phone numbers, emails, PAN, Aadhaar, and credentials.

## Memory and Feedback

The system uses both short-term and long-term memory.

Short-term memory:

- Active Streamlit session context
- Recent conversation turns

Long-term memory:

- Supabase conversation memory when configured
- Local JSONL fallback when needed

Retention policy:

- Delete after 60 days of inactivity
- Delete on explicit customer request

Feedback loop:

- Users can give feedback on answers
- Feedback is stored for future interactions
- Adaptation affects tone, detail level, and answer structure
- Feedback does not change guardrails or data access rules

## Support Dashboard

The dashboard is available to support staff.

It shows:

- Pending branch manager escalations from the last 7 days
- Pending risk team escalations from the last 7 days
- Failure analysis based on evaluation scores
- Failed evaluation dimensions for debugging

## Logging, Error Capture, and Tracing

### Local Logs

Main audit log:

```text
logs/baseline_agent_runs.jsonl
```

Evaluation log:

```text
logs/evaluation_runs.jsonl
```

Escalation log:

```text
logs/escalations.jsonl
```

Local logs capture route, risk level, tools used, confidence score, sanitized response, latency, errors, and evaluation results.

### Langfuse Tracing

Langfuse tracing is supported through:

```text
app/observability/tracing.py
```

When enabled, the system creates one trace per user query and response pair.

Trace metadata includes:

- Route
- Risk level
- Tools used
- Confidence score
- Evaluation score
- Latency
- Error details when failures occur

Set `TRACE_VERBOSE=false` to avoid creating many child traces for one user query.

## Evaluation Metrics

Each response is evaluated before display.

Metrics use 0/1 scoring:

- `answered_query`
- `grounded_in_context`
- `route_and_tools_fit`
- `risk_guardrail_ok`
- `pii_safe`
- `no_internal_leakage`
- `customer_friendly`
- `no_error_visible`

The consolidated score is shown in the UI along with the confidence score and is also stored in the evaluation log.

## Local Setup

Python 3.12 is recommended.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create environment file:

```powershell
Copy-Item .env_example .env
```

Fill the required values in `.env`.

Common environment variables:

```text
OPENAI_API_KEY=
OPENAI_MODEL=
OPENAI_BASE_URL=
EMBEDDING_MODEL=

SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

ZILLIZ_ENDPOINT=
ZILLIZ_API_KEY=
ZILLIZ_CLUSTER_ID=

SEARCHAPI_API_KEY=
SEARCHAPI_BASE_URL=
SEARCHAPI_ENGINE=

LANGFUSE_ENABLED=
TRACE_VERBOSE=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Do not commit `.env` or any real secrets.

## Run Locally

Start the Streamlit UI:

```powershell
streamlit run app/ui/streamlit_app.py
```

Start the FastAPI service:

```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Local URLs:

- UI: http://localhost:8501
- API: http://localhost:8000
- API health: http://localhost:8000/health

## Run with Docker Compose

Build and start both UI and API:

```powershell
docker compose up -d --build
```

View running containers:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs -f
```

Stop containers:

```powershell
docker compose down
```

## Render Deployment

The project uses an automated deployment pipeline:

```text
GitHub -> GitHub Actions -> Docker image -> Render services
```

The GitHub workflow builds and publishes the Docker image.

Workflow file:

```text
.github/workflows/docker-publish.yml
```

Deploy two Render web services from the project.

### API Service

Command:

```text
uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
/health
```

### UI Service

Command:

```text
streamlit run app/ui/streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false
```

Render free instances may sleep after inactivity. The first request after sleep can take longer.

## Sample Test Queries

General:

```text
What is EMI?
Tell me about home loan repayment options.
Explain fixed deposits.
```

Personalized:

```text
What is my account balance?
Show my recent transactions.
What are my personal loan details?
```

Calculation:

```text
Calculate EMI for 500000 at 8.5% for 5 years.
What if I pay extra Rs. 10,000 every month on my personal loan?
```

Escalation and guardrail:

```text
My account was hacked.
Transfer 10000 to this account.
Can you approve my loan?
Draft a legal notice to the bank.
```

## Important Notes

- Authentication is handled through Supabase. Use - /.env_Customer_login
- RAG retrieval uses Milvus/Zilliz.
- The assistant is role-aware.
- The system masks PII before logging.
- The assistant does not perform real banking transactions.
- Guardrails are enforced outside the planner so unsafe LLM plans cannot bypass policy.
- Feedback changes response style only, not access permissions or compliance behavior.
