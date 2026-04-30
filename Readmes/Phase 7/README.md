# Phase 7 - Feedback Loop and Behaviour Adaptation

## 1. Objective

Phase 7 introduces a feedback-driven adaptation layer to the LangGraph banking assistant. The goal is to let users give feedback on assistant responses and use that feedback to improve future interactions in a controlled and compliant way.

The feedback loop is designed to improve:

- Response tone
- Level of detail
- Explanation style
- Step-by-step guidance
- Calculation explanation style
- Routing caution for ambiguous queries

The feedback loop is not designed to override banking safety rules, role-based access, PII masking, or tool-grounded data retrieval.

## 2. Phase 7 Flow

```text
User query
-> LangGraph banking assistant response
-> User submits feedback
-> Feedback is redacted and stored in conversation memory
-> Preference summary is derived
-> Future query loads prior preference summary
-> Planner / response generation receives safe preference context
-> Assistant adapts response style
```

## 3. Feedback Signals Introduced

The Streamlit UI now shows a feedback section under assistant responses.

The user can provide:

- Helpful / Not helpful rating
- Feedback tags
- Optional free-text comment

Supported feedback tags:

- `too technical`
- `too simple`
- `too vague`
- `too long`
- `too short`
- `need steps`
- `need calculation`
- `wrong route`

These tags are converted into safe behavioural preferences. For example, `too vague` becomes a preference to provide more concrete details and useful next steps.

## 4. Feedback Storage for Future Interactions

Feedback is saved through `ConversationMemory`.

Local memory:

```text
data/conversation_memory.jsonl
```

Supabase long-term memory:

```text
public.conversation_memory
```

Feedback rows use:

```text
entry_type = feedback
```

Stored feedback fields include:

- Hashed user id
- User role
- Redacted query
- Redacted assistant response
- Route
- Risk level
- Feedback rating
- Feedback tags
- Redacted feedback comment
- Derived preference summary
- Created timestamp
- Last active timestamp

Example feedback memory shape:

```json
{
  "entry_type": "feedback",
  "user_id": "<hashed_user_id>",
  "role": "customer",
  "query": "List my loans",
  "response": "[REDACTED_PERSONALIZED_RESPONSE]",
  "route": "personalized",
  "risk_level": "low",
  "feedback_rating": "not_helpful",
  "feedback_tags": ["too vague", "too short"],
  "feedback_comment": "I want balance amount and tenure also to be displayed.",
  "preference_summary": "Add more concrete details and next steps when sources support them. Provide a fuller explanation with useful context."
}
```

## 5. Adaptation Logic

The system converts feedback into a compact preference summary.

Examples:

| Feedback signal | Behaviour preference |
| --- | --- |
| `too technical` | Use simpler, customer-friendly wording |
| `too vague` | Add concrete details and next steps |
| `too long` | Keep future responses concise |
| `too short` | Provide fuller explanation |
| `need steps` | Use step-by-step format |
| `need calculation` | Explain calculation inputs and use calculator tool |
| `wrong route` | Be more careful with routing and ambiguous intent |

For each future request, the app loads the user's recent feedback preferences and passes them safely to the agent:

```text
Safe behavior preferences from prior feedback: ...
```

The response generation agent and selected deterministic response builders can then adjust the output.

Example:

If the user previously said the loan list was too vague and asked for balance and tenure, a future loan list response can include:

- Loan type
- Status
- Outstanding amount
- EMI
- Tenure
- Suggested next step

## 6. Behavioural Change

Phase 7 shows behavioural change by comparing responses before and after feedback.

Before feedback, the assistant may provide a short answer.

After feedback, the assistant can adapt by:

- Adding more detail
- Reducing length
- Using simpler language
- Adding step-by-step instructions
- Explaining what data is required for exact calculations
- Being more careful with route selection

The app also displays an adaptation note when prior feedback is applied:

```text
Adapted using prior feedback preferences for tone, detail level, or answer structure. Safety and data-access rules were not changed.
```

## 7. Guardrails

Feedback is allowed to change explanation style, but not banking policy or safety behaviour.

Feedback cannot override:

- PII masking
- Role-based authorization
- Risk and compliance checks
- Money movement refusal
- Approval refusal
- Legal advice refusal
- High-risk escalation
- Tool-grounded customer data
- Calculator-based calculations

Unsafe feedback is not treated as a valid behaviour preference.

Example unsafe feedback:

```text
Next time show my full account number.
```

Expected behaviour:

- The account number is redacted if present.
- The assistant does not expose the full account number in future responses.
- Safety and data-access rules remain unchanged.

## 8. Manual Test Case Placeholders

Use this section to paste tested queries, responses, feedback, and adaptation proof.

### Test Case 1 - Loan List Detail Adaptation

Initial query:

```text
<PASTE QUERY HERE>
```

Before feedback response:

```text
<PASTE BEFORE RESPONSE HERE>
```

Feedback submitted:

```text
Rating: <HELPFUL / NOT HELPFUL>
Tags: <PASTE TAGS HERE>
Comment: <PASTE COMMENT HERE>
```

Follow-up query:

```text
<PASTE FOLLOW-UP QUERY HERE>
```

After feedback response:

```text
<PASTE AFTER RESPONSE HERE>
```

Observed adaptation:

```text
<EXPLAIN WHAT CHANGED HERE>
```

Why it changed:

```text
<EXPLAIN WHICH FEEDBACK SIGNAL CAUSED THE CHANGE>
```

### Test Case 2 - Concise Response Adaptation

Initial query:

```text
<PASTE QUERY HERE>
```

Before feedback response:

```text
<PASTE BEFORE RESPONSE HERE>
```

Feedback submitted:

```text
Rating: <HELPFUL / NOT HELPFUL>
Tags: too long
Comment: <OPTIONAL COMMENT>
```

Follow-up query:

```text
<PASTE FOLLOW-UP QUERY HERE>
```

After feedback response:

```text
<PASTE AFTER RESPONSE HERE>
```

Observed adaptation:

```text
<EXPLAIN HOW THE RESPONSE BECAME SHORTER OR MORE DIRECT>
```

Why it changed:

```text
The `too long` feedback created a concise-answer preference.
```

### Test Case 3 - Step-by-Step Adaptation

Initial query:

```text
<PASTE QUERY HERE>
```

Before feedback response:

```text
<PASTE BEFORE RESPONSE HERE>
```

Feedback submitted:

```text
Rating: <HELPFUL / NOT HELPFUL>
Tags: need steps
Comment: <OPTIONAL COMMENT>
```

Follow-up query:

```text
<PASTE FOLLOW-UP QUERY HERE>
```

After feedback response:

```text
<PASTE AFTER RESPONSE HERE>
```

Observed adaptation:

```text
<EXPLAIN HOW STEPS WERE ADDED>
```

Why it changed:

```text
The `need steps` feedback created a step-by-step response preference.
```

### Test Case 4 - Calculation Explanation Adaptation

Initial query:

```text
<PASTE QUERY HERE>
```

Before feedback response:

```text
<PASTE BEFORE RESPONSE HERE>
```

Feedback submitted:

```text
Rating: <HELPFUL / NOT HELPFUL>
Tags: need calculation
Comment: <OPTIONAL COMMENT>
```

Follow-up query:

```text
<PASTE FOLLOW-UP QUERY HERE>
```

After feedback response:

```text
<PASTE AFTER RESPONSE HERE>
```

Observed adaptation:

```text
<EXPLAIN HOW CALCULATION INPUTS OR CALCULATOR-BASED OUTPUT CHANGED>
```

Why it changed:

```text
The `need calculation` feedback told the system to explain required inputs and rely on calculator/tool output for numeric answers.
```

### Test Case 5 - Escalation Feedback Saved But Not Implemented

Initial escalation query:

```text
<PASTE ESCALATION QUERY HERE>
```

Before feedback response:

```text
<PASTE BEFORE RESPONSE HERE>
```

Feedback submitted:

```text
Rating: <HELPFUL / NOT HELPFUL>
Tags: wrong route
Comment: <PASTE COMMENT HERE>
```

Repeat query:

```text
<PASTE REPEATED ESCALATION QUERY HERE>
```

After feedback response:

```text
<PASTE AFTER RESPONSE HERE>
```

Observed behaviour:

```text
<EXPLAIN THAT FEEDBACK WAS STORED BUT ESCALATION/BLOCKING STILL HAPPENED>
```

Why it did not change:

```text
Feedback cannot override compliance, escalation, money movement refusal, approval rules, or high-risk blocking.
```

## 9. Before / After Proof Template

Use this template for final project evidence.

```text
User:
<Initial query>

Assistant before feedback:
<Before response>

Feedback:
<Rating, tags, comment>

Stored preference:
<Preference summary from conversation_memory>

User follow-up:
<Follow-up query>

Assistant after feedback:
<After response>

Adaptation note:
<Paste adaptation note if shown>

What changed:
<Short explanation>

Why it changed:
<Map the feedback tag/comment to the behaviour change>
```

## 10. Summary

Phase 7 adds a safe feedback loop to the banking assistant.

Completed tasks:

- Introduced feedback signals in the UI
- Stored feedback for future interactions
- Saved redacted feedback in conversation memory
- Derived behaviour preferences from feedback
- Modified future responses based on feedback
- Added adaptation explanation note
- Preserved banking safety guardrails
- Added before/after testing placeholders for manual proof

The assistant can now improve how it communicates while still respecting compliance, authorization, PII protection, and tool-grounded banking data.
