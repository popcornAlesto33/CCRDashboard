# Prompt Engineering Test Results

## Script: `Scripts/03_CallRail_Transcripts_Analyze_Buckets.py`
## Prompts Modified: `REASONING_SYSTEM_PROMPT`, `CLASSIFICATION_SYSTEM_PROMPT`
## Models: gemini-2.5-pro (reasoning) -> gemini-2.5-flash (classification)
## Gold Label Set: 510 labeled examples

---

## Accuracy Targets

| Field | Target |
|-------|--------|
| appointment_booked | 90% |
| client_type | 90% |
| treatment_type | 80% |
| reason_not_booked | 85% |

---

## Results Summary (all n=50 unless noted)

| Field | v1 (baseline) | v2 | v3 | v4 | v5* | v6 (n=50) | v6 (n=100) | v7 | v8 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| appointment_booked | 77.6% | 72.0% | 76.1% | 83.7% | 85.0% | 84.0% | 76.8% | 81.6% | 79.2% |
| client_type | 83.7% | 82.0% | 80.4% | 83.7% | 82.5% | 86.0% | **89.5%** | 87.8% | 85.4% |
| treatment_type | 40.8% | 46.0% | 52.2% | **67.3%** | 60.0% | 62.0% | 48.4% | **67.3%** | 66.7% |
| reason_not_booked | 25.0% | 17.9% | 27.6% | 27.6% | 30.0% | **35.3%** | **38.7%** | 31.6% | 27.3% |

*v5 had 8 dropped calls from API 503 errors (40/50 evaluated)

---

## v1 — Baseline (original prompt, no changes)

**Results:** appt 77.6% | client 83.7% | treatment 40.8% | reason 25.0%

**Root cause analysis identified 5 prompt weaknesses:**
1. Parent vs sub-category over-specification (treatment_type — 7 of 29 errors)
2. Inconclusive definition too restrictive (appointment_booked — 11 errors)
3. Client type "seems familiar" too vague (client_type — 8 errors)
4. Reason not booked decision rules missing (reason_not_booked — 14 errors)
5. Emergency vs Urgent Care boundary unclear (treatment_type — 2 errors)

---

## v2 — Initial 5 Fixes

**Key tweaks:**
- **Fix 1**: Rewrote parent vs sub-category rule — sub-category requires THE SPECIFIC INTERVENTION, not just symptoms. Added "over-specification is the #1 error pattern" guidance.
- **Fix 2**: Removed "extremely rarely" for appointment_booked Inconclusive. Added concrete criteria (voicemail, callback, "okay" without confirming, inter-clinic, gathers info).
- **Fix 3**: Replaced vague "seems familiar" client_type default with operational signals (file lookup, pet on record, past visits = Existing; asks about new patients, pricing = New).
- **Fix 4**: Added reason_not_booked decision rules for confused pairs (Procrastination vs Price Objection, Walk-ins vs Full Schedule, Getting Info vs Procrastination).
- **Fix 5**: Clarified Emergency vs Urgent Care — hospital context matters more than symptom severity. Named Kingston Regional Pet Hospital as emergency hospital example.
- Rewrote Example 2 rationale to justify sub-category by intervention, not symptoms.
- Added Example 2b showing symptoms-only = parent category.
- Updated CLASSIFICATION_SYSTEM_PROMPT with Inconclusive and client_type guidance.

**Results:** appt 72.0% | client 82.0% | treatment 46.0% | reason 17.9%

**Key findings:**
- treatment_type improved (+5.2pp) — parent/sub guidance starting to work
- appointment_booked REGRESSED (-5.6pp) — Inconclusive over-prediction. Fix 2 was too permissive: "gathers info but does not commit" caused 6 gold=No calls to be predicted as Inconclusive
- reason_not_booked REGRESSED (-7.1pp) — cascading from wrong appointment_booked

---

## v3 — Tighten Inconclusive, Fix Emergency, Reason Not Booked Null Default

**Key tweaks:**
- **Fix A**: Tightened Inconclusive criteria — added explicit "these are No, NOT Inconclusive" rules (pricing inquiry + hang up = No, schedule full + ends call = No, "I'll think about it" = No).
- **Fix B**: Fixed Emergency vs Urgent Care — removed hospital name as sole trigger. "Classify based on ACTUAL SERVICE discussed, not the hospital name." Advice calls and stable-patient triage at emergency hospitals = classify by actual service.
- **Fix C**: Tightened reason_not_booked generation — "default to null for Inconclusive calls" unless barrier is explicitly stated.
- **Fix D**: Added Preventive Care vs Diagnostic vs Retail clarifications (routine bloodwork = Wellness Screening, symptom-driven = Diagnostic, flea/tick prevention = Preventive Care not Retail).
- **Fix E**: Added explicit reasoning step requirement — model MUST state the specific intervention or say "no specific intervention mentioned."

**Results:** appt 76.1% | client 80.4% | treatment 52.2% | reason 27.6%

**Key findings:**
- treatment_type continued improving (+6.2pp from v2)
- appointment_booked: "these are No" rules overcorrected — now 7 gold=Inconclusive predicted as No (pendulum swung other way)
- Emergency misclassification eliminated from top confusions — Fix B worked
- reason_not_booked improved from v2 dip

---

## v4 — Balance No/Inconclusive, Strengthen Preventive Care Parent

**Key tweaks:**
- **Fix F**: Softened No vs Inconclusive boundary — removed absolute "IMPORTANT" rules. Added admin calls and pending-action calls back to Inconclusive list. Made No criteria less aggressive (only when caller "clearly tried to book but did NOT succeed").
- **Fix G**: Added Preventive Care parent vs sub-category section — parent for general wellness visits, new pet checkups, or multi-service calls. Sub-category only when SOLE AND EXPLICIT purpose.

**Results:** appt 83.7% | client 83.7% | treatment 67.3% | reason 27.6%

**Key findings:**
- **Best overall run** — appointment_booked +6.1pp from baseline
- treatment_type 67.3% — massive improvement (+26.5pp from baseline). Parent/sub confusion dropped from 7 to 1 case.
- Emergency misclassification mostly resolved
- No/Inconclusive balance much better (only 1 gold=Inconclusive→pred=No)
- reason_not_booked still lagging — model still generates reasons for Inconclusive calls

---

## v5 — Classification Prompt Reason Fix (partial run)

**Key tweaks:**
- Updated CLASSIFICATION_SYSTEM_PROMPT reason_not_booked rule: "When Inconclusive, default to null — only populate if reasoning explicitly names a clear barrier."

**Results:** appt 85.0% | client 82.5% | treatment 60.0% | reason 30.0%

**Key findings:**
- Only 40/50 calls evaluated (8 dropped from API 503 errors) — results not fully comparable
- appointment_booked best at 85.0% but small sample
- Vaccinations→parent regression (2x) — parent-over-sub guidance overcorrecting for Preventive Care

---

## v6 — Client Type "THIS Clinic", Reason Null Fix, New Examples

**Key tweaks:**
- **Fix H**: client_type — added "Existing means the CALLER has been a client at THIS specific clinic before." Edge cases: new owner of existing patient = New, vet at different clinic = New.
- **Fix I**: reason_not_booked — added "9. Client/appt query is ONLY for appointment_booked=No" rule. Added "4. Meant to call competitor" decision rule.
- **Fix J**: Added Example 3b (admin call — Inconclusive with null reason, showing Preventive Care for bloodwork follow-up).
- Updated Example 5 (inter-clinic) to show null reason instead of "1c. Getting info for someone else."
- Strengthened Price Objection guidance — use 1a when pricing discussed AT ANY POINT and caller doesn't book.

**Results (n=50):** appt 84.0% | client 86.0% | treatment 62.0% | reason 35.3%

**Key findings:**
- **client_type 86.0%** — new best at n=50, "THIS clinic" fix worked well
- **reason_not_booked 35.3%** — new best! Type A errors (gold=null, model generates reason) dropped from ~6/run to just 1. Over-generation problem essentially solved.
- Remaining reason errors are mostly cascading from wrong appointment_booked (4) and genuine category confusion (6)

**Results (n=100):** appt 76.8% | client 89.5% | treatment 48.4% | reason 38.7%

**Key findings at scale:**
- **client_type 89.5%** — 0.5pp from 90% target! Fix H highly effective at scale.
- treatment_type 48.4% — much lower than n=50 runs, revealing the 50-call samples were optimistically biased
- Urgent Care parent→Diagnosis sub: 6x at n=100 (biggest single error pattern)
- Preventive Care→Other: 5x (admin/voicemail/wrong-number calls — mostly gold label issues)
- ~12 gold label issues identified (model is actually correct, gold labels based on CRM data not transcript content)

---

## v7 — Strengthen Intervention Definition, Emergency Walk-in, Appointment Rules

**Key tweaks:**
- **Fix K**: Strengthened Urgent Care parent rule — explicit lists of what IS and ISN'T a specific intervention. "We'll take a look" / "physical examination" = NOT interventions (use parent). "We'll run bloodwork" / "start antibiotics" = specific interventions (use sub-category).
- **Fix L**: Rewrote appointment_booked as Yes/No/Inconclusive with structured criteria. Added: emergency walk-in where caller agrees to come in = Yes. Existing appointment + cancellation list = Yes. Advice-only calls where caller never intended to book = No.
- **Fix M**: Added emphasis on Example 8 (Vaccinations sub-category) — model was ignoring its own few-shot example due to over-aggressive parent-default.
- Updated CLASSIFICATION_SYSTEM_PROMPT appointment_booked with Yes/No/Inconclusive summary.

**Results:** appt 81.6% | client 87.8% | treatment 67.3% | reason 31.6%

**Key findings:**
- **Urgent Care parent→sub: 0 errors!** Fix K eliminated the biggest single error pattern (was 6x at n=100 in v6)
- treatment_type 67.3% — tied for best at n=50
- client_type 87.8% — consistently approaching 90%
- `CAL0038d600` now correctly predicts Yes (emergency walk-in fix working)
- appointment_booked ranges 72-85% across runs — No/Inconclusive boundary remains noisy

---

## Persistent Error Calls (appear in 3+ runs)

These ~10 calls appear in error lists across nearly all runs:

| Call ID | Primary Issue | Diagnosis |
|---------|--------------|-----------|
| `CAL0039a82c...` | All 4 fields wrong | **Gold label issue** — voicemail/automated message, gold=Yes/Existing/Wellness Screening |
| `CAL0198423279...` | client_type, treatment_type | **Gold label issue** — wrong number call, gold=Existing/Preventive Care |
| `CAL01983d5be...` | client_type, appointment | **Edge case** — new owner of existing patient, pending callback |
| `CAL011c013...` | client_type, reason | **Prompt issue** — medication advice call, model infers Existing (has vet elsewhere) |
| `CAL019846b3...` | treatment_type, reason | **Ambiguous** — post-surgical follow-up: gold=Diagnostic, model=Surgical |
| `CAL019838be7...` | treatment_type, reason | **Boundary** — routine bloodwork: gold=Wellness Screening, model=Diagnostic Lab |
| `CAL019838bea...` | treatment_type | **Gold label issue** — rescheduling call, no medical content discussed |
| `CAL019847429...` | client_type | **Persistent** — model sees prescription history = Existing, gold=New |
| `CAL01983cac3...` | client_type | **Persistent** — gold=Existing, model=New |
| `CAL0198481b9...` | appointment, treatment | **Edge case** — emergency hospital walk-in, model says Inconclusive |

---

## v8 — Fix Inconclusive Anchor, Decision Tree, Other/Dermatology Gates

**Key tweaks:**
- **Fix N**: Removed "22% Inconclusive" frequency anchor from both prompts. Replaced with "Use Inconclusive sparingly" / "Inconclusive should be rare."
- **Fix O**: Added sequential decision tree (Yes → No → Inconclusive) to force evaluation order before categorizing.
- **Fix P**: Added info-gathering = No bullets (completed calls where question answered = No, not Inconclusive).
- **Fix Q**: Added "Other" gate rules — 5-point checklist before using Other, plus "common traps" list.
- **Fix R**: Added Dermatology recognition section (allergies, ear infections, Apoquel/Cytopoint).
- **Fix S**: Expanded Wellness Screening vs Diagnostic Lab with inline examples and key test.
- **Fix T**: Tightened classification prompt appointment_booked with expanded No criteria and "Inconclusive should be rare."

**Results:** appt 79.2% | client 85.4% | treatment 66.7% | reason 27.3%

**Key findings:**
- **Over-correction on Inconclusive→No**: 6x gold=Inconclusive predicted as No. The decision tree + "sparingly" language pushed the model too hard away from Inconclusive. v7 had 4x of this error; v8 has 6x.
- Only 2x No→Inconclusive (down from v7's 2x) and 1x Yes→Inconclusive (down from v7's 2x) — the original problem (over-use of Inconclusive) IS fixed, but replaced by the opposite error.
- treatment_type: Still 2x Wellness Screening→Diagnostic Lab despite expanded examples. 1x Urgent Care→Dermatology (Fix R over-fired). 1x Other still present (gold=Preventive Care, persistent gold label call).
- Dermatology gate catching too broadly — "Urgent Care / Sick Pet" with skin symptoms routed to Dermatology instead.
- **Net assessment:** Fixes N-T shifted the error distribution but didn't reduce total errors. Decision tree needs softening for Inconclusive — admin/pending callbacks shouldn't be forced to No.

---

## Key Error Patterns Remaining (after v7)

### treatment_type (67.3%, target 80%)
- Preventive Care→Other (2x): admin/rescheduling calls — **gold label issue** (no medical content in transcript)
- Wellness Screening→Diagnostic Services (2x): bloodwork context ambiguous from transcript alone
- Preventive Care→sub-category (2x): model over-specifying (Parasite Prevention, Dental Care)
- Emergency parent→sub (1x): persistent granularity mismatch
- Vaccinations→parent (1x): parent-default still slightly overcorrecting

### appointment_booked (81.6%, target 90%)
- No→Inconclusive (2x): model still struggles with "I'll call back" = No
- Inconclusive→No (4x): model applies "caller chose not to book" too broadly
- Yes→Inconclusive (2x): emergency walk-ins and voicemails

### reason_not_booked (31.6%, target 85%)
- Cascading from wrong appointment_booked: ~4 errors
- Price Objection→Service Not Offered: 2x (persistent misclassification)
- Getting Info→other categories: 2x
- gold=null→model generates reason: 2-3x (much improved from ~6/run)

---

## Gold Label Issues Identified (~12 calls)

These calls have gold labels that cannot be derived from the transcript content alone (labels were likely assigned from CRM/appointment system data):

1. **Voicemail/automated messages** (3-4 calls): Gold assigns specific treatment_type and appointment_booked=Yes for calls that are just automated greetings
2. **Wrong number calls** (2-3 calls): Gold assigns Existing/Preventive Care for calls where caller dialed the wrong clinic
3. **Admin/rescheduling calls** (2-3 calls): Gold assigns specific treatment_type for calls where only appointment scheduling was discussed, no medical content
4. **Inter-clinic record transfers** (1-2 calls): Gold assigns Urgent Care for purely administrative file-transfer calls

**Impact estimate:** Fixing these gold labels would boost treatment_type by ~5-8pp and appointment_booked by ~2-3pp.

---

## v8 Rebaseline (corrected gold labels, 2026-03-14)

**Results:** appt 93.3% | client 95.6% | treatment 62.2% | reason 57.9%

**Note:** Same v8 prompts as v8, but 8 gold label corrections applied (see tasks/gold_label_audit.md). This is the true baseline for v9 comparison.

**Key findings:**
- **appointment_booked 93.3% — PASSES 90% target.** Gold label fixes (voicemail, wrong numbers) accounted for most of the improvement.
- **client_type 95.6% — PASSES 90% target.** Fixing the 2 swapped New/Existing labels helped.
- treatment_type 62.2% — actually dropped from v8's 66.7%. Top confusions: Preventive Care→Other (3x), Annual Exams→parent (3x), Emergency parent→sub (2x).
- reason_not_booked 57.9% — large jump from v8's 27.3%. The gold label fixes removed many false errors.
- Only 3 appointment_booked errors remain (2x No→Inconclusive, 1x Inconclusive→No).

---

## v9.3 — Field-Decomposed Pipeline, Minimal Rules (2026-03-14)

**Architecture:** 4 parallel Pro calls (one per field, raw transcript) + Python assembly
**Model:** gemini-2.5-pro, individual calls (no batching)
**Key changes from v9.0:** Stripped excessive rules, trust model judgment, added calibrating examples, strict enum schemas, callback-is-No rule

| Version | appointment_booked | client_type | treatment_type | reason_not_booked |
|---------|:---:|:---:|:---:|:---:|
| v9.0 (initial) | 92.0% | 94.0% | 58.0% | 50.0% |
| v9.1 (examples+enums) | 92.0% | 90.0% | 52.0% | 52.6% |
| v9.2 (minimal rules) | 88.0% | 96.0% | 68.0% | 52.4% |
| **v9.3 (callback fix)** | **92.0%** | **96.0%** | **70.0%** | **63.2%** |

**Key findings:**
- Stripping rules and trusting model judgment improved treatment_type from 52% → 70% (best ever)
- client_type 96% — highest ever, well above target
- appointment_booked 92% — PASSES target with just minimal guidance
- reason_not_booked 63.2% — improving via cascade break from better appointment_booked
- The "less rules, more trust" approach from the original script analysis was validated
- Remaining treatment_type errors: 4x admin/rescheduling→Other (likely gold label issues), 3x Urgent Care parent/sub boundary, 2x Emergency/Urgent Care confusion
- **Blocked on n=100 validation** — Gemini Pro daily quota (1000 RPD) exhausted. Upgraded to Tier 1, will retest tomorrow.

---

## Flash Model Evaluation (2026-03-14)

Tested gemini-2.5-flash as a cheaper alternative to Pro for each field.

### Single-step Flash (v9.3 prompts, n=100)

| Field | Flash (n=100) | Pro (n=50) | Target | Flash viable? |
|-------|:---:|:---:|:---:|:---:|
| appointment_booked | 89.0% | 92.0% | 90% | Borderline — 1pp below |
| client_type | 91.0% | 96.0% | 90% | **Yes** — passes target |
| treatment_type | 45.0% | 70.0% | 80% | **No** — too weak |
| reason_not_booked | 48.6% | 63.2% | 85% | **No** — too weak |

### Two-Step Flash for treatment_type (Flash parent → Flash sub)

| Version | n=50 | n=100 | Change |
|---------|:---:|:---:|:---:|
| v1 (basic) | 64.0% | 47.0% | — |
| v2 (improved Step 1) | 72.0% | 54.0% | +7pp |
| v3 (Step 2 guidance) | — | 59.0% | +5pp |
| v4 (Step 1 examples) | — | 56.0% | -3pp (overcorrected) |

**Key findings:**
- Two-step Flash beats single-step Flash by ~14pp at n=100 (59% vs 45%)
- Two-step Flash at n=50 (72%) beat single-step Pro at n=50 (70%), but n=100 shows Flash ceiling at ~56-59%
- Step 1 (parent classification) works well — most errors are Step 2 (sub-category selection)
- Flash lacks reasoning power for fine-grained sub-category distinctions (Urgent Care parent vs sub, Preventive Care sub-categories)
- **Recommended hybrid:** Flash Step 1 + Pro Step 2 for treatment_type

### Recommended Production Architecture

| Field | Model | Rationale |
|-------|-------|-----------|
| appointment_booked | Flash | Simple Yes/No/Inconclusive — Flash performs near target |
| client_type | Flash | Passes 90% target at n=100 |
| treatment_type | Flash Step 1 + Pro Step 2 | Flash picks parent (11 choices), Pro picks sub (2-5 choices) |
| reason_not_booked | Pro | 28 categories, needs reasoning. Only fires for appt=No calls (~30-40%) |

---

## Cost Per Run

| Run Type | Reasoning Cost | Classification Cost | Total | Per-Call |
|----------|:---:|:---:|:---:|:---:|
| 50 calls | ~$0.21 | ~$0.005 | ~$0.21 | ~$0.004 |
| 100 calls | ~$0.41 | ~$0.010 | ~$0.42 | ~$0.004 |
