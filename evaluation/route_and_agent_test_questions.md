# Agent + Route Test Question Bank

This checklist is designed to exercise **all intent routes, specialist agents, fallback routing behavior, and HITL pause path** in the Life Insurance AI Copilot.

## How to Use
- Send each question to `/chat` first (functional validation).
- Repeat a subset through `/chat/stream` (SSE validation).
- Confirm `intent` and `node_path` in `/state/{session_id}`.
- For high-risk underwriting cases, verify pause behavior and then resume using `/approve`.

---

## 1) Intent Router Coverage (All 7 Intents)

### A. `policy_qa` → `policy_qa_agent`
1. What is the difference between term life and whole life insurance?
2. Explain the free-look period and refund rules in VitaLife policies.
3. What are the key exclusions in a standard term policy?
4. Is ULIP better than endowment for long-term wealth accumulation?

### B. `underwriting` → `underwriting_agent`
1. I am 38, non-smoker, want 50 lakh cover for 25 years. What is my estimated premium?
2. I have hypertension and Type 2 diabetes; how will this affect underwriting?
3. What medical disclosures are mandatory during underwriting?
4. I am a commercial pilot with occasional smoking history—what risk tier might I get?

### C. `beneficiary` → `beneficiary_agent`
1. Can I nominate two beneficiaries with 60/40 allocation?
2. How do I nominate a minor beneficiary and who receives payout before adulthood?
3. Can I change my nominee after policy issuance, and what documents are needed?
4. Can a non-family member be my nominee?

### D. `issuance` → `issuance_agent`
1. What documents are required for policy issuance?
2. How long does issuance usually take after medicals are complete?
3. My application is pending; what are common reasons for delay?
4. When is the policy bond dispatched after approval?

### E. `lapse_revival` → `lapse_revival_agent`
1. I missed two premium payments. Is my policy lapsed yet?
2. What is the grace period for monthly premium mode?
3. How do I revive a lapsed policy after 14 months?
4. Do I need fresh medicals for reinstatement?

### F. `policy_comparison` → `policy_comparison_agent`
1. Compare term, whole life, and endowment in a table with pros and cons.
2. Term plan vs ULIP: which is better for pure protection vs investment?
3. Compare maturity benefits and premium levels for endowment vs whole life.

### G. `lapse_prediction` → `lapse_prediction_agent`
1. Based on frequent delayed payments in the last 6 months, what is my lapse risk?
2. Predict lapse likelihood if I already missed two consecutive premiums.
3. I pay irregularly every alternate month—am I likely to lapse soon?

---

## 2) Human-in-the-Loop (HITL) Path Validation

Use these to trigger elevated risk and verify:
- `requires_human_review = true`
- node path includes `human_review`
- response contains pause instruction

1. I am 57, smoker, chronic kidney disease stage 3, and had a recent heart attack. I need 1 crore cover.
2. I have uncontrolled diabetes (HbA1c 10+) and severe obesity (BMI 43). Can I get immediate coverage?
3. I work in underground mining and have coronary artery disease history; estimate premium for 75 lakh.

Then test `/approve`:
- Approve scenario: “approved” decision should resume and finalize.
- Reject scenario: “rejected” decision should resume with rejection-safe messaging.

---

## 3) Fallback / Ambiguous Router Behavior

These check defaulting and robustness:

1. I need help choosing a policy.  
   - Expected: likely `policy_qa` default.
2. Compare and suggest the safest option for me (no profile provided).  
   - Expected: `policy_comparison` or `policy_qa` depending on classifier.
3. I missed payments and also want to change nominee. What should I do first?  
   - Expected: one primary intent selected; verify consistency.
4. history  
   - Expected: may route to `lapse_prediction` due to keyword fallback rule.

---

## 4) Multi-Turn State Carryover Tests

Run each as a sequential conversation in the same session:

### Sequence A: Underwriting memory
1. I am 34 years old.
2. I want 80 lakh cover for 30 years.
3. I am a smoker with mild hypertension.
4. Now estimate my premium.

Validate that extracted fields accumulate in `applicant_data` and are used in the final premium response.

### Sequence B: Short follow-up retrieval optimization
1. What riders are available?
2. What about waiver?

Validate short follow-up still retrieves relevant policy context.

---

## 5) API Route Validation Prompts

Use this minimal set to test endpoints and expected behavior:

- `GET /health` (no question; expect service up).
- `POST /chat` with:
  - “Compare term vs whole life in a table.”
- `POST /chat/stream` with:
  - “I missed 3 premiums; how can I revive my policy?”
- `GET /state/{session_id}` after each call; inspect `intent`, `node_path`, `node_outputs`.
- `POST /approve` after a HITL-triggering underwriting query.
- `GET /sessions` to verify session list shows updated conversation metadata.
- `DELETE /sessions/{session_id}` to verify cleanup.

---

## 6) Quick Smoke Set (Fast Full Coverage)

If you need a compact suite, run these 10 only:
1. Difference between term and whole life?
2. I am 29 non-smoker; estimate premium for 1 crore, 25 years.
3. I have diabetes and hypertension—risk category?
4. Can I nominate a minor and a spouse jointly?
5. Required documents for issuance?
6. I missed premium deadline; grace period and revival steps?
7. Compare ULIP vs endowment in table format.
8. Predict lapse risk if I missed 2 consecutive premiums.
9. High-risk profile: CKD + smoker + heart attack history, 1 crore cover. *(HITL expected)*
10. Approve the paused case and resume.
