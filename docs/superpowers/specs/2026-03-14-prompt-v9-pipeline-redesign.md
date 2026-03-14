# Prompt v9: 6-Stage Pipeline Redesign

**Date:** 2026-03-14
**Script:** `Scripts/03_CallRail_Transcripts_Analyze_Buckets.py`
**Validation:** `Scripts/validate_prompt_engineering.py`

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

## Solution: 6-Stage Pipeline

Replace the current 2-stage pipeline (Reasoning → Classification) with a 6-stage pipeline that decomposes the problem by field.

### Architecture

```
                                         ┌─ Stage 4: treatment_type (Pro)  ──┐
                                         │                                    │
Stage 1: Extract (Flash) ──► Stage 2: appointment_booked (Pro) ──────────────┼──► Stage 6: Assemble (Flash)
                                         │                                    │
                                         ├─ Stage 5: client_type (Pro)    ──┤
                                         │                                    │
                                         └─ Stage 3: reason_not_booked (Pro)─┘
                                                (waits for Stage 2)
```

### Parallelism

- Stages 2, 4, 5 start immediately after Stage 1 completes (no dependencies between them)
- Stage 3 waits for Stage 2 only (needs appointment_booked result)
- Stage 6 waits for all reasoning stages
- Within each stage, all 50 transcripts fire concurrently (async)

### Models & Batch Sizes

| Stage | Model | Batch Size | Rationale |
|-------|-------|-----------|-----------|
| 1. Extract | Gemini 2.5 Flash | 8-10 | Mechanical extraction, cheap/fast |
| 2. appointment_booked | Gemini 2.5 Pro | 1 | Accuracy-critical, no batching penalty |
| 3. reason_not_booked | Gemini 2.5 Pro | 1 | Accuracy-critical, no batching penalty |
| 4. treatment_type | Gemini 2.5 Pro | 1 | Accuracy-critical, no batching penalty |
| 5. client_type | Gemini 2.5 Pro | 1 | Accuracy-critical, no batching penalty |
| 6. Assemble | Gemini 2.5 Flash | 8-10 | Schema enforcement, cheap/fast |

**Cost estimate:**
- Stage 1: 50 calls in ~6 batches × Flash pricing = ~$0.01
- Stages 2-5: 50 calls × 4 fields × Pro pricing (~$0.004/call from v1-v8, but shorter prompts → ~$0.002/call) = ~$0.40
- Stage 6: 50 calls in ~6 batches × Flash pricing = ~$0.01
- **Total: ~$0.40-0.50 per 50-call run (~$0.008-0.010 per call)**
- Current 2-stage pipeline: ~$0.21 per run. The ~2x increase is from batch_size=1 on Pro (more API overhead per call). This can be reduced later by increasing batch sizes once prompts stabilize.

### Priority Order

Fields are prioritized by impact and dependency:
1. **appointment_booked** — highest impact, other fields depend on it
2. **reason_not_booked** — largest gap to target, cascading from appointment_booked
3. **treatment_type** — second largest gap, independent
4. **client_type** — nearly at target, independent

## Stage Specifications

### Stage 1: Extraction (Flash)

Pure fact extraction — no judgment or classification. Extracts what was said, not what it means.

**Input:** Raw transcript
**Output:** Structured JSON per call

**Exact JSON schema:**

```json
{
  "call_id": "CAL123",
  "caller_identity_signals": [
    "Agent looked up file under caller's name and found it",
    "Caller referenced a past visit last month"
  ],
  "medical_content": "Dog has been vomiting for 2 days. Agent discussed bringing dog in for examination.",
  "specific_interventions": ["bloodwork", "X-rays"],
  "appointment_outcome_signals": [
    "Agent said 'we'll see you at 2pm Tuesday'",
    "Caller confirmed 'yes, that works'"
  ],
  "pricing_discussed": {
    "discussed": false,
    "context": null
  },
  "call_purpose": "Caller wants to bring sick dog in for examination",
  "names": {
    "hospital": ["Welland Animal Hospital"],
    "pet": ["Sophie"],
    "agent": ["Linda"],
    "doctor": ["Dr. Rauch"]
  },
  "transcript_quality": "normal"
}
```

**Field specifications:**

| Field | Type | Description |
|-------|------|-------------|
| caller_identity_signals | `string[]` | List of concrete evidence quotes/observations. Empty array if no signals. |
| medical_content | `string` | Free-text summary of medical services/conditions discussed. Empty string if none. |
| specific_interventions | `string[]` | Named procedures/tests/treatments only (not "we'll take a look"). Empty array if none. |
| appointment_outcome_signals | `string[]` | Quotes/actions indicating booking outcome. Empty array if none. |
| pricing_discussed | `{discussed: bool, context: string\|null}` | Whether cost came up, with quote if yes. |
| call_purpose | `string` | Why the caller called, in their words. |
| names | `{hospital: string[], pet: string[], agent: string[], doctor: string[]}` | All names mentioned. Arrays to support multiple names per role. Empty arrays if none. |
| transcript_quality | `enum` | One of: `"normal"`, `"voicemail"`, `"very_short"` (<3 exchanges), `"garbled"` (mostly redacted/unintelligible) |

### Stage 2: appointment_booked Reasoning (Pro)

**Input:** Extraction JSON
**Output:** Reasoning text + preliminary answer (Yes / No / Inconclusive)

**Prompt rules (carried forward from v1-v8):**
- Evaluation order: Yes → No → Inconclusive (decision tree from v8)
- Emergency walk-in where caller agrees to come in = Yes
- Existing appointment being rescheduled = Yes
- "I'll think about it" / "I'll call back" = No
- Caller got info and ended call without scheduling = No
- Voicemail / clinic callback pending = Inconclusive
- Admin/records calls with no appointment intent = Inconclusive
- **Soften Inconclusive language** — v8's "sparingly" and "rare" overcorrected (6x Inconclusive→No). Use neutral framing: "Inconclusive means the outcome depends on a future event that hasn't happened yet"
- Must cite specific signals from extraction to justify answer

### Stage 3: reason_not_booked Reasoning (Pro)

**Input:** Extraction JSON + appointment_booked result from Stage 2
**Output:** Reasoning text + preliminary answer (category or null)

**Hard rules:**
- If appointment_booked = Yes → output null immediately, no reasoning needed
- If appointment_booked = Inconclusive → output null (unless explicit barrier stated in extraction)
- Only reason when appointment_booked = No

**Decision rules for confused pairs (carried forward):**
- Price Objection (1a) vs Procrastination (1): pricing discussed at ANY point → 1a
- Walk-ins not available (2a) vs Full schedule (2b): same-day = 2a, multiple days = 2b
- Getting info for someone else (1c) vs Procrastination (1): explicitly calling on behalf of someone = 1c
- Meant to call competitor (4): caller dialed the wrong clinic entirely

**Cascade break:** This stage treats appointment_booked as ground truth. It does not second-guess Stage 2's decision.

### Stage 4: treatment_type Reasoning (Pro)

**Input:** Extraction JSON
**Output:** Reasoning text + preliminary answer (exact category from enum)

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

### Stage 5: client_type Reasoning (Pro)

**Input:** Extraction JSON
**Output:** Reasoning text + preliminary answer (New / Existing / Inconclusive)

**Prompt rules (carried forward):**
- "Existing" means the CALLER has been to THIS specific clinic before (v6 fix)
- Existing signals: agent finds file, pet already in system, references past visit at this clinic
- New signals: asks about new patients, asks about location/hours as if unfamiliar, agent creates new file
- Edge cases: new owner of existing patient = New, vet at different clinic calling here = New
- Inconclusive should be extremely rare for client_type
- Simplest prompt of the four — client_type is already near target

### Stage 6: Assembly (Flash)

**Input:** All four reasoning outputs + name extraction from Stage 1
**Output:** Final validated JSON matching current schema

**Rules:**
- Extract preliminary answers from each reasoning stage into structured JSON
- Cross-field consistency check (log-only warnings — Flash never overrides Pro's reasoning output):
  - appointment_booked=Yes + reason_not_booked populated → log warning, set reason_not_booked to null (this is a hard rule, not a judgment call)
  - appointment_booked=No + reason_not_booked=null → log warning for human review (do NOT attempt to generate a reason)
- Pass through names from Stage 1 extraction
- The existing `CLASSIFICATION_RESPONSE_SCHEMA` JSON schema (treatment_type enum, reason_not_booked enum, etc.) remains unchanged — Stage 6 maps preliminary answers to the exact enum values

## Error Handling

Each stage processes calls independently. When a call fails at any stage:

1. **Stage 1 (Extract) fails for a call:** Skip that call for all downstream stages. Log the call_id and error. Stage 6 produces no output for that call.
2. **Stage 2 (appointment_booked) fails for a call:** Stage 3 (reason_not_booked) skips that call (cannot run without appointment_booked). Stages 4, 5 are unaffected. Stage 6 produces a partial result with appointment_booked=null and reason_not_booked=null.
3. **Stage 3, 4, or 5 fails for a call:** Stage 6 produces a partial result with that field set to null. Other fields are unaffected.
4. **Stage 6 (Assembly) fails for a call:** Log the error. That call has no final output.

**Retry strategy:** Each API call gets up to 2 retries with exponential backoff (1s, 2s) before being marked as failed. This matches the current `max_retries=3` on the OpenAI client.

**Reporting:** After each validation run, print a summary of failed calls (if any) so they can be investigated. Partial results are excluded from accuracy scoring.

## Concurrency & API Strategy

**API client:** Continue using the OpenAI-compatible shim (`generativelanguage.googleapis.com/v1beta/openai/`). This is battle-tested in the current codebase and avoids a SDK migration.

**Concurrency model:** Use `concurrent.futures.ThreadPoolExecutor` (matches current codebase). Not asyncio — simpler, already working, and the bottleneck is API latency not CPU.

**Rate limiting:** Gemini Pro has documented RPM limits. To avoid hitting them with batch_size=1:
- Use a semaphore-based throttle: max 10 concurrent requests per stage
- If 429 (rate limit) responses occur, back off and retry with exponential delay
- Stages 2, 4, 5 each get their own thread pool (they run in parallel but each is independently throttled)

**Execution flow:**
```
1. ThreadPool(10): Stage 1 — extract all 50 calls
2. ThreadPool(10): Stage 2 — appointment_booked for all 50 calls  ──┐
   ThreadPool(10): Stage 4 — treatment_type for all 50 calls        ├─ concurrent
   ThreadPool(10): Stage 5 — client_type for all 50 calls           │
3. ThreadPool(10): Stage 3 — reason_not_booked for all 50 calls  ◄──┘ (waits for Stage 2)
4. ThreadPool(10): Stage 6 — assemble all 50 calls
```

## Validation Script Changes

### Pipeline Orchestration

Refactor `validate_prompt_engineering.py` to support the 6-stage pipeline:

1. Run Stage 1 (Extract) for all calls, cache results
2. Run Stages 2, 4, 5 in parallel (async), each processing all calls with batch_size=1
3. Run Stage 3 after Stage 2 completes, using appointment_booked results
4. Run Stage 6 to assemble final JSON
5. Score per-field accuracy against gold labels

### Extraction Caching

Stage 1 extractions are cached between prompt iterations. When iterating on a specific field's prompt, only that field's reasoning stage needs to re-run. This enables fast iteration cycles (~30 seconds per field test vs full pipeline re-run).

**Cache implementation:**
- Stored as a JSON file: `CallData/extraction_cache.json`
- Cache key: hash of the Stage 1 extraction prompt + call_id
- If the Stage 1 prompt changes, the prompt hash changes, and all cached extractions are automatically invalidated
- CLI flag: `--no-cache` to force re-extraction

### Gold Label Audit Integration

After each validation run, for every mismatch:
- Output: call_id, transcript excerpt, gold label, model prediction, model reasoning
- This enables quick human review to determine if the error is a prompt issue or a gold label issue
- Proposed gold label fixes are documented in `tasks/gold_label_audit.md`
- **CSV is never modified without explicit human sign-off**

### Backward Compatibility

Keep existing `--single-model` and two-model modes for comparison. Add `--pipeline v9` flag (or similar) to select the new 6-stage pipeline.

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

### Batch Size Optimization (Future)

After prompts stabilize at batch_size=1:
1. Test batch_size=2, measure accuracy delta
2. Test batch_size=4, measure accuracy delta
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

The new stage prompts will be defined in the production script (as constants, like the current `REASONING_SYSTEM_PROMPT` and `CLASSIFICATION_SYSTEM_PROMPT`), but the 6-stage orchestration logic lives in the validation script for now.

## Files Modified

| File | Change |
|------|--------|
| `Scripts/03_CallRail_Transcripts_Analyze_Buckets.py` | New stage prompts, pipeline orchestration |
| `Scripts/validate_prompt_engineering.py` | 6-stage pipeline support, extraction caching, async execution, audit output |
| `CallData/VetCare_CallInsight_Labels - labels.csv` | Gold label fixes (8 fields across 5 calls, already applied) |
| `tasks/gold_label_audit.md` | Ongoing gold label audit trail |
| `tasks/test_results.md` | v9 results appended |
