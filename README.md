# Phase 1: Problem Understanding and Definition

## Project

AI Banking Support and Advisory Agent (Non-Transactional)
### Persona 1. Customer (End User)

Who: Retail banking user interacting with chatbot

Needs:

Deposits, loans, EMI, credit cards
Product info & FAQs
Fraud awareness

AI Behavior:

Simple, clear, non-technical responses
Must refuse sensitive actions (money transfer, account access)
Guide + educate
### Persona 2. Branch Manager (Escalation Persona)

Who: Handles complex or unresolved cases

Needs:

Summarized user issue
Risk level (low/medium/high)
Suggested next steps

AI Behavior:

Provide concise summaries
Recommend escalation (not execute actions)
### Persona 3. Risk & Compliance Officer

Who: Ensures fraud prevention and policy adherence

Needs:

Fraud-related queries
Suspicious behavior classification
Policy explanations

AI Behavior:

Strict, rule-based responses
No guessing / no hallucination
Always prioritize safety
### Persona 4. Admin (System Oversight)

Who: Monitors system behavior

Needs:

How queries are classified
Safety rule enforcement
Audit-level explanations

AI Behavior:

Transparent, structured responses
No exposure of sensitive user data
### Persona 5. Customer Support Agent (Human Assist)

Who: Bank employee using AI as co-pilot

Needs:

Suggested replies to customers
Context summaries
Next best action

AI Behavior:

Assist human (not replace)
Provide ready-to-use responses
🧩 Simple Persona Routing (Use This in Design)
Customer query → Customer Persona
Fraud / suspicious → Risk Persona
Complex / unclear → Branch Manager
Internal / audit → Admin
Human-assisted flow → Support Agent

## Daily Workflow

- A customer or bank staff member has a banking-related question.
- They open the chat interface and log in with their assigned role (customer, support agent, admin, etc.).
- The system assigns the corresponding persona based on the role.
- The user asks a question in natural language.
- The system analyzes the query for:
     - Intent (loan, EMI, fraud, general info)
     - Risk level (safe, ambiguous, high-risk, disallowed)
- The system routes the query to the appropriate persona:
   - Customer → general queries
   - Risk → fraud/security
   - Branch Manager → escalations
   - Admin → internal/audit
- The system retrieves relevant banking knowledge (RAG) as context.
- The system applies safety rules (no transactions, no PII exposure, no financial/legal advice).
### The system responds with one of the following:
- Direct informational answer
- Clarification question (if query is unclear)
- Refusal for restricted or unsafe requests
- Escalation guidance for fraud, security, or high-risk cases

## 2. Problem Statement

Build an AI assistant that provides safe, accurate, and non-transactional banking support and guidance. The system must enforce compliance boundaries by refusing transactional actions, avoiding hallucinated customer data, and escalating fraud or security-related situations.

## 3. Inputs, Outputs, Constraints, and Assumptions

### Inputs

- User role
- User query in natural language
- Optional conversation history in the active session

### Outputs

Structured response containing:

- Direct answer or guidance
- Clarification when the request is incomplete or ambiguous
- Refusal when the request is disallowed
- Escalation when the request is high-risk

### Constraints

- No money movement or transaction execution
- No balance lookup or customer-specific account access
- No product approval or decision-making
- No storage of sensitive personal data
- No hallucinated personal or banking data
- Must escalate fraud, account compromise, or security threats

### Assumptions

- The assistant is not connected to core banking systems
- The assistant has no access to live account or transaction data
- Banking policies and examples are used as reference context
- The assistant is used for support and advisory guidance only

## 4. Example User Questions

### Safe

- What is the interest rate for a fixed deposit?
- How does a home loan work?
- What is EMI?

### Ambiguous

- I want to send money
- Tell me about loans
- Open an FD

### Disallowed

- Transfer 10,000 to this account
- What is my account balance?
- Approve this loan for me

### High-Risk

- Money got deducted but I did not do it
- I think my account is hacked
- Someone asked for my OTP

## 5. Success Criteria

### Functional

- Answers common informational banking questions accurately
- Uses the role and query context correctly
- Produces useful clarification when the intent is ambiguous

### Safety

- Refuses transactional and account-specific requests
- Avoids hallucinating personal or financial data
- Escalates fraud and security issues consistently

### User Experience

- Responds in clear and simple language
- Gives actionable next steps
- Maintains a clean role-based support flow

## 6. Failure Cases and Edge Scenarios

### Failure Cases to Avoid

- Hallucinating account details such as balance or transaction history
- Acting as if a transfer, approval, or account change was completed
- Missing escalation for fraud, hacking, or unauthorized access
- Giving direct financial decisions instead of general guidance

### Edge Scenarios

- Mixed intent: Explain FD and open one
- Role-based query differences between User and Admin
- Repeated attempts to bypass safety rules
- Panic situations such as all my money is gone
- Incomplete questions such as What is EMI

## Final Insight

This system is not just a chatbot. It is a policy-aware banking support assistant that classifies intent, respects safety boundaries, and uses an LLM to generate non-transactional guidance.

## Phase 4 RAG Loading

This project now includes a simple ingestion script that:

- reads selected documents from `Docs/`
- chunks the text
- creates embeddings
- loads the vectors into Zilliz Cloud / Milvus

Only these files are considered:

- `Accounts.docx`
- `Bank FAQ's.docx`
- `Deposits.docx`
- `Loan.docx`

### Required `.env` values

Add these values before running the loader:

```env
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_MODEL=text-embedding-3-small

ZILLIZ_ENDPOINT=your_zilliz_public_endpoint
ZILLIZ_API_KEY=your_zilliz_api_key
ZILLIZ_CLUSTER_ID=your_zilliz_cluster_id
ZILLIZ_COLLECTION_NAME=banking_rag_chunks
```

Notes:

- `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` can point to any OpenAI-compatible embeddings endpoint.
- `ZILLIZ_CLUSTER_ID` is stored in the ingest summary for reference. The actual Milvus connection uses `ZILLIZ_ENDPOINT` and `ZILLIZ_API_KEY`.

### Install dependencies

```powershell
pip install pymilvus
```

### Run the loader

```powershell
python -m app.scripts.load_rag_to_milvus --drop-existing
```

Optional flags:

- `--chunk-size 800`
- `--chunk-overlap 120`
- `--batch-size 16`

### Output

After loading:

- vectors are inserted into the configured Zilliz / Milvus collection
- a small summary file is written to `data/rag_ingest_summary.json`

## Phase 5 Tools

Minimal Phase 5 scaffolding has been added for:

- Calculator tool
- Supabase customer/loan/transaction tool
- SearchAPI interest-rate lookup tool
- RAG retrieval tool
- MCP-style local server/client registry
- LangGraph baseline agent scaffold

Key files:

- `app/tools/calculator.py`
- `app/tools/supabase_tool.py`
- `app/tools/searchapi_tool.py`
- `app/tools/rag_tool.py`
- `app/tools/langgraph_tools.py`
- `app/mcp/server.py`
- `app/mcp/client.py`
- `app/agents/langgraph_agent.py`
- `data/phase5_supabase_schema.sql`

Suggested `.env` additions:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

SEARCHAPI_BASE_URL=your_searchapi_endpoint
SEARCHAPI_API_KEY=your_searchapi_key
```

Run the LangGraph baseline agent:

```powershell
python -m app.agents.langgraph_agent --role customer --customer-id C001 --query "Show my latest 5 transactions"
```

### Phase 5 Prompt

The Phase 5 system prompt is now stored in:

- `prompts/phase5/system_prompt.txt`

#### Before Change

This was the original hardcoded prompt inside `app/agents/langgraph_agent.py`:

```text
You are the Phase 5 banking baseline agent. Choose tools automatically when needed. Use calculator for arithmetic. Use Supabase tools for customer, loan, branch, and transaction data. Use search_api for external rate lookup. Use rag_retrieval for internal banking document context. Respect role boundaries: customer sees only own data, manager sees only branch data, admin/support/risk can view all data, support can update only CustomerName, Address, City, and State, admin can add or delete users. If a tool is not needed, answer directly and briefly.
```

#### After Change

This is the current file-based Phase 5 prompt:

```text
You are the Phase 5 banking baseline agent.

Choose tools automatically when needed.
Use calculator for arithmetic and amount calculations.
Use Supabase tools for customer, loan, branch, and transaction data.
Use search_api for external rate lookup when internal context is weak or unavailable.
Use rag_retrieval for internal banking document context first.

Respect role boundaries:
- customer sees only own data
- manager sees only branch data
- admin, support, and risk can view all customer data
- support can update only CustomerName, Address, City, and State
- admin can add or delete users

When internal RAG results are weak, incomplete, or do not answer the exact user query, prefer search_api as fallback.
For comparison questions, combine internal RAG context and external search results before answering.
If a tool is not needed, answer directly and briefly.
Always produce a customer-friendly final response instead of raw tool output.
```

### Phase 5 Failures and Corrections

#### 1. Customer and Manager Authorization Path

Failure:

- Protected data requests were sometimes denied before the correct Supabase read path ran.

Correction:

- Protected customer and manager queries now use direct Supabase calls with the logged-in user JWT so that Supabase RLS can evaluate the real authenticated user.

#### 2. Branch Manager Returned Empty Rows

Failure:

- Branch-manager queries returned empty lists even though data existed.

Correction:

- The app now loads `role` and `branch` during login more reliably.
- Manager reads now rely on Supabase RLS with the manager JWT instead of depending on an extra app-side branch filter.
- Branch and role matching in Supabase should use trimmed values to avoid whitespace mismatch.

#### 3. RAG Did Not Fall Back to Internet Search

Failure:

- Search fallback was triggered only when RAG returned zero matches.
- Weak but irrelevant RAG matches prevented `search_api` from running.

Correction:

- Search fallback now runs when RAG has no matches, when confidence is low, or when the retrieved text does not contain the key query terms.

#### 4. Prompt Was Hardcoded

Failure:

- The Phase 5 prompt was embedded directly in the agent code.

Correction:

- The prompt is now stored under `prompts/phase5/system_prompt.txt` and loaded through `app/core/prompts.py`.
