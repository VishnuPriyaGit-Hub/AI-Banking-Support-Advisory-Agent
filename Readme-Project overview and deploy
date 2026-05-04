# Banking Support Agent - Project Overview and Deployment

## Live Services

- Streamlit UI: https://banking-support-ui.onrender.com
- FastAPI service: https://banking-support-api.onrender.com
- API health check: https://banking-support-api.onrender.com/health

## Login Flow

1. Open the Streamlit UI.
2. Enter the Supabase-authenticated email and password.
3. Supabase Auth validates the user.
4. The app fetches the user's profile from Supabase using the authenticated user id.
5. The profile determines role, branch, and linked customer id.
6. The assistant applies role-based access rules before fetching personalized data.

Supported roles include:

- Customer
- Branch manager
- Admin
- Support staff
- Risk staff

Customers can access only their linked account/customer data. Staff roles receive broader access only where the workflow allows it.

## High-Level Architecture

```text
User Query
  -> Streamlit UI
  -> LangGraph Orchestration
  -> Planner Agent
  -> Risk and Compliance Agent
  -> MCP Tool Calls
  -> Risk and Compliance Check
  -> Response Generation Agent
  -> Final Response
```

The Streamlit UI is the main user-facing application. FastAPI exposes the same assistant flow for API-based deployment and integration.

## Agent Flow

### Planner Agent

The planner uses the LLM to understand the user's intent and produce a structured plan.

The plan includes:

- `route`: general, personalized, calculation, or escalation
- `plan_type`: data lookup, personalized guidance, calculation, or default
- `required_tools`: RAG, DB, calculator, search, or escalation
- `data_scope`: minimum DB scope needed, such as customer loans or customer transactions
- `entities`: safe semantic targets, such as loan type
- `calculation_task`: calculation operation and numeric inputs

The planner is LLM-first. Code guardrails still validate unsafe or unauthorized actions.

### Risk and Compliance Agent

The risk layer classifies requests as low, medium, or high risk.

- Low risk: allowed
- Medium risk: escalated to branch manager
- High risk: blocked/escalated to risk team

It prevents:

- money movement
- approvals
- legal advice
- OTP/PIN/password/CVV handling
- unauthorized access to other customers' data
- ambiguous banking actions that require human approval

### MCP Tools

Tools retrieve, calculate, search, or execute workflow actions. They are not separate agents.

Current tool categories:

- RAG tool: retrieves FAQ, policy, product, and general banking knowledge
- Supabase DB tool: retrieves authorized customer, loan, transaction, branch, or staff data
- Calculator tool: performs EMI, interest, eligibility, balance, and repayment impact calculations
- Search API tool: used when RAG confidence is low or external context is needed
- Escalation tool: creates and updates branch manager, admin, support, or risk workflows

### Response Generation Agent

The response agent combines tool outputs into a customer- or staff-facing answer.

It must not expose:

- prompts
- tool internals
- raw database JSON
- database implementation details
- sensitive identifiers

Personalized DB responses use sanitized structured context before LLM response generation.

## Data Flow

### Authentication

Authentication is handled through Supabase Auth. The old local SQLite auth database was removed.

## PII Protection and Guardrails

The assistant includes PII masking and banking guardrails throughout planning, tool use, response generation, logging, and memory.

### PII Handling

PII and sensitive identifiers are redacted before being stored in logs or memory where possible.

Protected fields and patterns include:

- account numbers
- card numbers
- phone numbers
- email addresses
- PAN
- Aadhaar/Aadhar
- OTP, PIN, CVV, password, and passcodes
- auth user ids and access tokens
- raw customer identifiers where not needed for execution

Personalized DB outputs are not passed to the LLM as raw database JSON. The app first creates sanitized structured context and only includes fields needed to answer the question.

Examples:

- Transaction questions receive transaction date, type, amount, merchant, category, and balance-after.
- Loan questions receive loan type, status, EMI, tenure, interest rate, and outstanding balance.
- Internal IDs, account numbers, credentials, phone, email, and address are excluded from customer-facing LLM context.

### Guardrails

The guardrail layer is deterministic and remains active even when the LLM planner is used.

The assistant refuses or blocks:

- money movement requests
- fund transfers
- withdrawals
- loan approvals
- legal advice
- OTP/PIN/CVV/password handling
- unauthorized access to another customer's data
- destructive account actions

The assistant escalates:

- medium-risk customer/account maintenance requests
- ambiguous banking action requests
- complaints or disputes that need human review
- high-risk fraud or account compromise cases

Escalation routing:

- Medium risk: branch manager review
- Approved account/customer maintenance: admin or support workflow, depending on action type
- High risk: risk team review

Guardrails cannot be changed by user feedback. Feedback can adjust tone, detail level, or answer structure only.

### Database Access

Supabase stores customer, loan, transaction, and role-related data.

The agent fetches the minimum data scope needed:

- `customer_transactions`: transaction-only questions
- `customer_loans`: loan-only questions
- `customer_snapshot`: balance/profile or mixed account context
- `branch_customers`: manager branch customer view
- `branch_loan_customers`: manager branch loan customer view
- `all_customers`: authorized admin/support/risk review

### RAG

RAG is used for banking FAQs, policies, products, repayment options, and general knowledge.

Milvus is used as the vector database for RAG retrieval. Banking documents are embedded and stored in Milvus, then retrieved at query time to provide relevant FAQ, policy, product, and general banking context.

If RAG confidence is low, the planner can request Search API fallback.

### Calculations

The LLM does not manually calculate. It plans the calculation and extracts structured inputs. The calculator tool performs the numeric work, and the response agent explains the result.

## Memory

Short-term memory:

- Active conversation context in the current session
- Used for follow-up questions and contextual rewriting

Long-term memory:

- Supabase conversation memory table where configured
- Local fallback memory file for development/demo runs

Retention policy:

- Delete after 60 days of inactivity
- Delete on explicit customer request

Feedback preferences are stored as memory signals and can influence tone, detail level, or answer structure. Feedback does not override safety, access control, or compliance rules.

## Local Logging

Local audit logs are stored in:

```text
logs/baseline_agent_runs.jsonl
```

Each run can include:

- timestamp
- route
- risk level
- tools used
- confidence score
- trace id
- total latency
- redacted final response
- step-level logs

Step-level logs may include:

- planner decision
- risk result
- tool used
- latency per step/tool
- confidence score
- error type
- redacted error message

Escalation workflow records are stored in:

```text
logs/escalations.jsonl
```

These records are redacted before storage and are used by manager/admin/support/risk screens.

## Error Logging

Local error logging is captured inside the JSONL audit logs.

Typical error fields:

```json
{
  "step": "mcp_tool_call",
  "status": "error",
  "tool_used": "search_api",
  "latency_ms": 500,
  "error_type": "TimeoutError",
  "error": "search_api failed: timeout"
}
```

Render service errors can also be viewed from Render service logs:

- `banking-support-ui`
- `banking-support-api`

## Latency Tracking

Local latency is stored in:

```text
logs/baseline_agent_runs.jsonl
```

Tracked latency fields include:

- `total_latency_ms`
- planner latency
- risk latency
- DB/tool latency
- response generation latency

## Langfuse Tracing

Langfuse is configured for one trace per user query-response pair.

Recommended environment:

```env
LANGFUSE_ENABLED=true
TRACE_VERBOSE=false
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

With `TRACE_VERBOSE=false`, Langfuse stores one top-level trace for each user turn.

The trace metadata includes:

- route
- risk level
- tools used
- total latency
- step latency summary
- tool latency summary
- error count
- redacted error details

Set `TRACE_VERBOSE=true` only when debugging detailed internal spans. For normal demos, keep it false to avoid multiple Langfuse rows per user query.

## Deployment Pipeline

The project has an automated deployment pipeline:

```text
GitHub push to main
  -> GitHub Actions workflow
  -> Docker image build
  -> Publish image to GitHub Container Registry
  -> Render pulls and deploys service
```

Workflow file:

```text
.github/workflows/docker-publish.yml
```

The workflow builds the Docker image and publishes it to GitHub Container Registry.

## Render Services

Two Render web services are used:

### API Service

```text
Service: banking-support-api
URL: https://banking-support-api.onrender.com
Health: /health
Command: uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
```

### UI Service

```text
Service: banking-support-ui
URL: https://banking-support-ui.onrender.com
Command: streamlit run app/ui/streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false
```

Both services require the same core environment variables for OpenAI, Supabase, Langfuse, and Search API.

## Required Environment Variables

Do not commit secrets to GitHub. Add them in Render environment settings.

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...

LANGFUSE_ENABLED=true
TRACE_VERBOSE=false
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com

SEARCH_API_KEY=...
```

## Testing Checklist

1. Open the UI service.
2. Login with Supabase credentials.
3. Ask a general question:

```text
What is EMI?
```

4. Ask a personalized data question:

```text
Show my recent transactions
```

5. Ask a calculation question:

```text
What if I pay extra Rs. 10,000 every month on my personal loan?
```

6. Ask an escalation question:

```text
My account was hacked.
```

7. Check Render logs for runtime errors.
8. Check local/Render audit logs if running locally.
9. Check Langfuse for one trace per query-response pair.
