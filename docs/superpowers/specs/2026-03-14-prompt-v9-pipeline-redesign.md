# Prompt v9: Field-Decomposed Pipeline Redesign

**Date:** 2026-03-14
**Validation script:** `Scripts/validate_prompt_engineering.py`
**Revision:** v2 — simplified after architectural review (removed extraction and LLM assembly stages)

## Problem Statement

After 8 iterations of prompt engineering, accuracy has plateaued:

| Field | Current Best | Target | Gap |
|-------|:---:|:---:|:---:|
| appointment_booked | 85.0% (v5) | 90% | -5pp |
| client_type | 89.5% (v6 n=100) | 90% | -0.5pp |
| treatment_type | 67.3% (v4/v7) | 80% | -12.7pp |
| reason_not_booked | 38.7% (v6 n=100) | 85% | -46.3pp |

Root causes:
1. **Cascading errors:** reason_not_booked depends on appointment_booked — wrong appointment prediction contaminates reason prediction
2. **Cognitive overload:** The reasoning model classifies all 4 fields in one pass (~3000 token prompt), leading to cross-field contamination
3. **No/Inconclusive oscillation:** Each fix swings the pendulum (v2 too permissive → v3 too strict → v4 balanced → v8 overcorrected again)
4. **Gold label noise:** ~12 labels derived from CRM data, not transcript content (8 fixed pre-v9)

## Solution: 4 Parallel Field Classifiers + Python Assembly

Replace the current 2-stage pipeline (Reasoning → Classification) with 4 parallel Pro calls — one per field — each reading the raw transcript directly. A Python assembly step applies deterministic hard rules and builds the final JSON.

**Why no extraction stage:** Each Pro call now handles only ONE field, giving it ample cognitive capacity to read the raw transcript. An extraction stage would add a failure point and risk hiding information from downstream models (if Flash misses a subtle signal, Pro never sees it). Field decomposition already solves the cognitive overload problem.

**Why Python assembly, not LLM:** The assembly step is purely deterministic (parse outputs, apply hard rules like "if Yes then reason=null", build JSON). No judgment involved — an LLM adds latency, cost, and failure risk for zero benefit.

### Architecture

```
                 ┌─ Pro: appointment_booked ─────┐
                 │         │                      │
Raw transcript ──┼─ Pro: treatment_type ──────────┼─► Python assembly ──► Final JSON
                 │         │                      │
                 ├─ Pro: client_type ─────────────┤
                 │                                │
                 └─ Pro: reason_not_booked ────────┘
                      (waits for appointment_booked)
```

### Parallelism

- appointment_booked, treatment_type, and client_type all start immediately (no dependencies)
- reason_not_booked waits for appointment_booked only (cascade break)
- Within each field, all 50 transcripts run concurrently via thread pool
- Python assembly runs after all 4 fields complete

### Models & Batch Sizes

| Field | Model | Batch Size | Rationale |
|-------|-------|-----------|-----------|
| appointment_booked | Gemini 2.5 Pro | 2 | Start at 2, measure accuracy, adjust |
| reason_not_booked | Gemini 2.5 Pro | 2 | Start at 2, measure accuracy, adjust |
| treatment_type | Gemini 2.5 Pro | 2 | Start at 2, measure accuracy, adjust |
| client_type | Gemini 2.5 Pro | 2 | Start at 2, measure accuracy, adjust |
| Assembly | Python (no LLM) | N/A | Deterministic rules |

**Cost estimate:**
- 4 fields × 25 batches (50 calls / batch_size 2) × Pro pricing (~$0.002/batch for shorter single-field prompts) = ~$0.20
- **Total: ~$0.20 per 50-call run (~$0.004 per call)**
- Comparable to current 2-stage pipeline (~$0.21). Shorter per-field prompts offset the per-call overhead of smaller batches.

### Priority Order

Fields are prioritized by impact and dependency:
1. **appointment_booked** — highest impact, reason_not_booked depends on it
2. **reason_not_booked** — largest gap to target, cascading from appointment_booked
3. **treatment_type** — second largest gap, independent
4. **client_type** — nearly at target, independent

## Field Prompt Specifications

Each Pro call receives the raw transcript and a focused prompt for one field only. Output is reasoning text + a final answer. The reasoning is preserved for debugging/auditing.

### appointment_booked (Pro)

**Input:** Raw transcript
**Output:** `{ "reasoning": "...", "answer": "Yes" | "No" | "Inconclusive" }`

**Prompt rules (carried forward from v1-v8):**
- Evaluation order: Yes → No → Inconclusive (decision tree from v8)
- Emergency walk-in where caller agrees to come in = Yes
- Existing appointment being rescheduled = Yes
- "I'll think about it" / "I'll call back" = No
- Caller got info and ended call without scheduling = No
- Voicemail / clinic callback pending = Inconclusive
- Admin/records calls with no appointment intent = Inconclusive
- **Soften Inconclusive language** — v8's "sparingly" and "rare" overcorrected (6x Inconclusive→No). Use neutral framing: "Inconclusive means the outcome depends on a future event that hasn't happened yet"
- Must cite specific transcript quotes to justify answer

### reason_not_booked (Pro)

**Input:** Raw transcript + appointment_booked result from previous step
**Output:** `{ "reasoning": "...", "answer": "<category>" | null }`

**Hard rules:**
- If appointment_booked = Yes → output null immediately, no reasoning needed
- If appointment_booked = Inconclusive → output null (unless explicit barrier stated in transcript)
- Only reason when appointment_booked = No

**Decision rules for confused pairs (carried forward):**
- Price Objection (1a) vs Procrastination (1): pricing discussed at ANY point → 1a
- Walk-ins not available (2a) vs Full schedule (2b): same-day = 2a, multiple days = 2b
- Getting info for someone else (1c) vs Procrastination (1): explicitly calling on behalf of someone = 1c
- Meant to call competitor (4): caller dialed the wrong clinic entirely

**Cascade break:** This prompt treats appointment_booked as ground truth. It does not second-guess the previous step's decision.

### treatment_type (Pro)

**Input:** Raw transcript
**Output:** `{ "reasoning": "...", "answer": "<exact enum value>" }`

**Prompt rules (carried forward):**
- Parent vs sub-category: when in doubt, use parent (biggest win from v1→v7)
- Must name the specific intervention to use a sub-category, or explicitly state "no specific intervention mentioned — using parent"
- NOT specific interventions: "we'll take a look", "bring them in for an exam", "physical examination"
- Specific interventions: "we'll run bloodwork", "dental cleaning scheduled", "allergy testing"
- Emergency vs Urgent Care: classify by actual service, not hospital name
- Preventive Care parent for general wellness / multi-service calls
- Wellness Screening vs Diagnostic Lab: routine (annual, pre-op) = Wellness Screening; symptom-driven = Diagnostic Lab
- Dermatology gate: only when PRIMARY reason is skin/ear/allergy (softened from v8)
- "Other" gate: 5-point checklist before using Other
- Dental cleanings/extractions = Surgical Services - Dental Care
- Full treatment_type enum provided in prompt for exact value matching

### client_type (Pro)

**Input:** Raw transcript
**Output:** `{ "reasoning": "...", "answer": "New" | "Existing" | "Inconclusive" }`

**Prompt rules (carried forward):**
- "Existing" means the CALLER has been to THIS specific clinic before (v6 fix)
- Existing signals: agent finds file, pet already in system, references past visit at this clinic
- New signals: asks about new patients, asks about location/hours as if unfamiliar, agent creates new file
- Edge cases: new owner of existing patient = New, vet at different clinic calling here = New
- Inconclusive should be extremely rare for client_type
- Simplest prompt of the four — client_type is already near target

### Python Assembly

**Input:** All four field outputs + raw transcript (for name extraction fallback)
**Output:** Final JSON matching existing `CLASSIFICATION_RESPONSE_SCHEMA`

**Deterministic rules (no LLM judgment):**
- Parse the `answer` field from each Pro output
- Map answers to exact enum values from `CLASSIFICATION_RESPONSE_SCHEMA`
- Cross-field consistency (hard rules):
  - appointment_booked=Yes + reason_not_booked populated → set reason_not_booked to null, log warning
  - appointment_booked=No + reason_not_booked=null → log warning for human review (do NOT generate a reason)
- Name extraction: parse reasoning outputs for hospital/pet/agent names, or use regex on raw transcript as fallback
- If any field's Pro call failed → set that field to null in output

The existing `CLASSIFICATION_RESPONSE_SCHEMA` (treatment_type enum, reason_not_booked enum, etc.) remains unchanged.

## Error Handling

Each field processes calls independently. When a call fails:

1. **appointment_booked fails for a call:** reason_not_booked skips that call (cannot run without it). treatment_type and client_type are unaffected. Assembly outputs partial result with appointment_booked=null and reason_not_booked=null.
2. **reason_not_booked, treatment_type, or client_type fails for a call:** Assembly outputs partial result with that field set to null. Other fields unaffected.
3. **Assembly fails for a call:** Log the error. That call has no final output.

**Retry strategy:** Each API call gets up to 2 retries with exponential backoff (1s, 2s) before being marked as failed. This matches the current `max_retries=3` on the OpenAI client.

**Reporting:** After each validation run, print a summary of failed calls (if any). Partial results are excluded from accuracy scoring.

## Concurrency & API Strategy

**API client:** Continue using the OpenAI-compatible shim (`generativelanguage.googleapis.com/v1beta/openai/`). Battle-tested in current codebase, avoids SDK migration.

**Concurrency model:** Single shared `concurrent.futures.ThreadPoolExecutor` with 15 workers and a global rate limiter. This avoids 4 independent pools that don't coordinate (which could collectively exceed API RPM limits).

**Rate limiting:**
- Global semaphore: max 15 concurrent Pro requests across all fields
- If 429 (rate limit) responses occur, back off and retry with exponential delay
- appointment_booked, treatment_type, and client_type share the pool concurrently
- reason_not_booked joins the pool after appointment_booked completes

**Execution flow:**
```
SharedThreadPool(15):
  1. appointment_booked (50 calls)  ──┐
     treatment_type (50 calls)        ├─ concurrent, sharing 15 workers
     client_type (50 calls)           │
  2. reason_not_booked (50 calls)  ◄──┘ (starts after appointment_booked done)
  3. Python assembly (instant, no API calls)
```

## Validation Script Changes

### Pipeline Orchestration

Refactor `validate_prompt_engineering.py` to support the new pipeline:

1. Run appointment_booked, treatment_type, client_type in parallel (shared thread pool)
2. Run reason_not_booked after appointment_booked completes
3. Run Python assembly
4. Score per-field accuracy against gold labels

### Gold Label Audit Integration

After each validation run, for every mismatch:
- Output: call_id, transcript excerpt, gold label, model prediction, model reasoning
- This enables quick human review to determine if the error is a prompt issue or a gold label issue
- Proposed gold label fixes are documented in `tasks/gold_label_audit.md`
- **CSV is never modified without explicit human sign-off**

### Backward Compatibility

Keep existing `--single-model` and two-model modes for comparison. Add `--pipeline v9` flag (or similar) to select the new field-decomposed pipeline.

## Testing Execution Strategy

### Baseline

Before any prompt changes, run current v8 prompts against corrected gold labels to establish a clean post-fix baseline.

### Per-Field Subagent Iteration

Each field gets its own iteration subagent:

```
Subagent A: appointment_booked prompt → test → audit mismatches → iterate
Subagent B: treatment_type prompt → test → audit mismatches → iterate
Subagent C: client_type prompt → test → audit mismatches → iterate
Subagent D: reason_not_booked → starts after Subagent A stabilizes appointment_booked
```

Subagents A, B, C run in parallel. Subagent D has a sequential dependency on A.

### Fixed Test Set

50 samples, first-50 sorted by call_id (deterministic, already in use for v1-v8). No `--random` flag.

### Batch Size Optimization

Start with batch_size=2. After initial results:
1. If accuracy is below v8 baseline → try batch_size=1, measure delta
2. If accuracy meets targets → try batch_size=4, measure delta
3. Find the sweet spot between throughput and accuracy
4. Apply to production pipeline

## Success Criteria

| Field | Current Best | Post-v9 Target |
|-------|:---:|:---:|
| appointment_booked | 85.0% | 90% |
| client_type | 89.5% | 90% |
| treatment_type | 67.3% | 80% |
| reason_not_booked | 38.7% | 70%+ (stretch: 85%) |

Note: reason_not_booked target is adjusted to 70%+ as a realistic first milestone given the 46pp gap. Many remaining errors are genuine category ambiguity rather than prompt issues.

## Scope: Validation Only (v9)

This redesign applies to the **validation pipeline** (`validate_prompt_engineering.py`) only. The production script (`03_CallRail_Transcripts_Analyze_Buckets.py`) retains its current 2-stage pipeline until v9 prompts are validated and stable. Production migration is a separate follow-up task.

All new prompt constants are defined in `validate_prompt_engineering.py` until production migration. This avoids polluting the production script with unused prompts.

## Files Modified

| File | Change |
|------|--------|
| `Scripts/validate_prompt_engineering.py` | New field-decomposed pipeline, shared thread pool, per-field prompts, Python assembly, audit output |
| `CallData/VetCare_CallInsight_Labels - labels.csv` | Gold label fixes (8 fields across 5 calls, already applied) |
| `tasks/gold_label_audit.md` | Ongoing gold label audit trail |
| `tasks/test_results.md` | v9 results appended |

## Revision History

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-03-14 | Initial spec (6-stage pipeline) | Field decomposition to break cascading errors |
| 2026-03-14 | v2: Removed extraction stage, LLM assembly → Python | Extraction adds hidden classification risk; assembly is deterministic; simpler architecture preserves all core benefits |
