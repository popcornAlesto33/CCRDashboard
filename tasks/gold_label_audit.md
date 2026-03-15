# Gold Label Audit

## Overview

Audit of ~10 persistent error calls that appear in error lists across 3+ prompt engineering runs (v1–v8). These calls have gold labels that are either clearly incorrect, debatable, or represent genuine edge cases.

**Source:** `CallData/VetCare_CallInsight_Labels - labels.csv`
**Identified from:** `tasks/test_results.md` — Persistent Error Calls section
**Impact estimate:** Fixing confirmed errors would boost treatment_type by ~5-8pp and appointment_booked by ~2-3pp.

---

## Category 1: Clearly Wrong Gold Labels

### 1. `CAL0039a82c50d340abaf5823d06785a4a0` — Automated Greeting (No Conversation)

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | Yes |
| client_type | Existing |
| treatment_type | Preventive Care – Wellness Screening |
| reason_not_booked | (null) |

**Transcript (182 chars — complete):**
> Agent: Hello, you have reached the Blue Sky Animal Hospital. Our office is currently closed today between 12 and 1pm if you'd like to schedule an appointment, you have the option of.

**Diagnosis:** Automated voicemail greeting that cuts off mid-sentence. No caller interaction whatsoever. Gold labels are impossible to derive from transcript — they were likely assigned from CRM/appointment system data.

**Recommended fix:**
- appointment_booked → Inconclusive (or N/A)
- client_type → Inconclusive
- treatment_type → N/A (missed call)
- reason_not_booked → (null)

**Status:** [x] Reviewed / [x] Fixed (2026-03-14)

---

### 2. `CAL0198423279ec78fdaf35068fb9b8d99d` — Wrong Number Call

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | No |
| client_type | Existing |
| treatment_type | Preventive Care |
| reason_not_booked | 4. Meant to call competitor hospital |

**Transcript (329 chars — complete):**
> Agent: Good morning. Welland Animal Hospital. How can I help you? Caller: Hi, is this the Cat and Dog clinic? Agent: Nope, this is the Welland Animal Hospital. You're looking for the one in six. That's okay. Caller: Oh, I'm sorry. Yeah, I called the wrong number. I'm sorry. Agent: No worries. Have a nice day. Caller: Thank you.

**Diagnosis:** Caller dialed the wrong clinic entirely. Gold has client_type=Existing but caller has never been to Welland Animal Hospital. treatment_type=Preventive Care is a guess — nothing medical was discussed. appointment_booked=No and reason=4 are correct.

**Recommended fix:**
- client_type → New
- treatment_type → Other (no medical content discussed)
- appointment_booked → No (correct, keep)
- reason_not_booked → 4. Meant to call competitor hospital (correct, keep)

**Status:** [x] Reviewed / [x] Fixed (2026-03-14)

---

### 3. `CAL019838beaef3720196945678ed36fe5d` — Pure Rescheduling Call

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | Yes |
| client_type | Existing |
| treatment_type | Preventive Care |
| reason_not_booked | (null) |

**Transcript (548 chars — complete):**
> Agent: Hi, NK RePet Hospital Village speaking. How can I help you today? Caller: Oops. Hi. My. Oh, there's an ambulance going by. Just. Agent: It's okay? Caller: I have an appointment to bring my dog in tomorrow, and I'm just wondering if I can move it to later in the day. Agent: Yes, I think so. Would you like to come? Like at 4pm? Caller: That would be perfect. Thank you. Yes. Agent: Okay. I'm just booking now. Like 4pM Perfect for tomorrow. 4pm See you tomorrow. Okay, thanks. Bye. Caller: Okay. Thank you. I appreciate that. Okay. Bye. Bye.

**Diagnosis:** Caller reschedules an existing appointment to 4pm tomorrow. No medical content discussed — treatment_type=Preventive Care cannot be derived from transcript alone. appointment_booked=Yes and client_type=Existing are reasonable.

**Recommended fix:**
- treatment_type → Other (no medical content in transcript; gold label should reflect transcript content only)
- appointment_booked → Yes (correct, keep)
- client_type → Existing (correct, keep)

**Decision:** User confirmed: gold labels should be derived from transcript content only, not CRM data. Since no medical content was discussed, treatment_type=Other.

**Status:** [x] Reviewed / [x] Fixed (2026-03-14)

---

### 4. `CAL019838be793178a7b613d26902e61915` — Inter-Clinic Bloodwork Results Inquiry

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | Inconclusive |
| client_type | Existing |
| treatment_type | Preventive Care – Wellness Screening |
| reason_not_booked | (null) |

**Transcript (1226 chars — complete):**
> Caller (Brian Nolan) went to Chelsea Clinic this morning for a blood test for his dog. Phoned Chelsea for results but no vet available. Dr. Penny is in Wakefield, so he's calling the Wakefield clinic to get results. Agent says Penny is in appointments, and "Anna's going to call you back."

**Diagnosis:** This is an inter-clinic administrative call. The blood test was done at Chelsea, not this clinic. Caller wants results relayed from a different location. No appointment was discussed or booked at this clinic. treatment_type=Wellness Screening refers to work done elsewhere.

**Recommended fix:**
- Keep all fields as-is.

**Decision:** User decided to keep treatment_type=Wellness Screening — the underlying service is bloodwork regardless of where it was performed. All other fields are correct.

**Status:** [x] Reviewed / [x] Kept as-is (2026-03-14)

---

## Category 2: Model Appears Correct, Gold Label Likely Wrong

### 5. `CAL01983cac376972818d9cef6bb1cba1fe` — Puppy Vaccination (New Client)

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | Yes |
| client_type | Existing |
| treatment_type | Preventive Care – Vaccinations |
| reason_not_booked | (null) |

**Transcript summary:** Caller wants to book second puppy vaccination. Agent asks for pet name (Amor) and last name (Maddi). When agent asks where the first vaccine was administered, caller says: **"No, not here. This is the first time I'm coming here. I actually got it from Barry."** Appointment booked for 10:30 today.

**Diagnosis:** Caller explicitly states this is their first visit to this clinic. Gold label client_type=Existing is incorrect. Model prediction of New is correct.

**Recommended fix:**
- client_type → New

**Status:** [x] Reviewed / [x] Fixed (2026-03-14)

---

### 6. `CAL019847429fae7401a033f30149097005` — Prescription Refill (Existing Client)

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | Inconclusive |
| client_type | New |
| treatment_type | Retail – Prescriptions |
| reason_not_booked | (null) |

**Transcript summary:** Barb Miller calls about her cat Sophie. Says **"I've had my cat Sophie in there over the last month and a half, I guess twice for an upper respiratory problem."** Agent looks up her medication history (doxycycline, claviceptin), references specific dates (June, July). Dr. Rauch/Dr. Val will approve a refill Monday and call her back.

**Diagnosis:** Caller has a well-established relationship — two visits in 1.5 months, agent has full medication history, uses first names. Gold label client_type=New is clearly incorrect. Model prediction of Existing is correct.

**Recommended fix:**
- client_type → Existing

**Status:** [x] Reviewed / [x] Fixed (2026-03-14)

---

### 7. `CAL011c013103cb49bc84379c12992b27a6` — Medication Advice Call (Debatable Client Type)

**Gold labels:**
| Field | Gold Value |
|-------|-----------|
| appointment_booked | No |
| client_type | New |
| treatment_type | Urgent Care / Sick Pet |
| reason_not_booked | 1. Caller Procrastination |

**Transcript summary:** Caller asks Kingston Regional Pet Hospital about chlorhexidine soap prescribed for her cat's recurring nail infections. She applied too much and wants to know about side effects / ingestion risk. Agent provides advice. Call is purely informational — caller never intended to book.

**Diagnosis:** The cat has an established treatment protocol (prescribed chlorhexidine for recurring condition). However, it's unclear whether this medication was prescribed by THIS clinic or another one. The caller calls Kingston Regional Pet Hospital, which is an emergency hospital — she may not be a regular client there.

**Model prediction:** Existing (based on established treatment protocol).
**Gold label:** New.
**Verdict:** Ambiguous. If the prescription came from a different clinic and she's calling the emergency hospital for after-hours advice, New is correct. If the prescription came from this clinic, Existing is correct. Transcript doesn't clarify definitively.

**Also debatable:** reason_not_booked="1. Caller Procrastination" — caller never intended to book, she called for advice only. Could argue this is more accurately "9. Client/appt query" or even that the reason category doesn't apply well to pure advice calls.

**Recommended fix:**
- No changes — client_type=New kept (emergency hospital likely not her regular vet)
- reason_not_booked=1. Caller Procrastination kept (debatable but not clearly wrong)

**Status:** [x] Reviewed / [x] Kept as-is (2026-03-14)

---

## Category 3: Edge Cases / Ambiguous (No Change Recommended)

### 8. `CAL01983d5be63a7410a179dc571fa6230a` — New Owner of Existing Patient

**Gold labels:** No / New / Preventive Care / (null)

**Transcript summary:** Caller has a newly rehomed cat (Marsha) whose records are already at this clinic from the previous owner. Caller dropped in before and was told to wait for the ownership transfer. Wants Revolution (parasite prevention). Agent finds Marsha's file but needs to update ownership. Ends with "we'll get back to you."

**Diagnosis:** Classic edge case — pet is Existing, but caller is New. Gold label New is reasonable since the prompt defines client_type as about the CALLER, not the pet. appointment_booked=No is debatable (could be Inconclusive since clinic will call back).

**No change recommended** — reasonable gold label, good test of prompt's edge case handling.

---

### 9. `CAL019846b31014783aa28b43775196dbc0` — Post-Surgical Follow-Up

**Gold labels:** Inconclusive / Existing / Diagnostic Services / (null)

**Transcript summary:** Caller (Holmes) has an appointment to check stitches from surgery last week. Asks about appointment time (11 o'clock) and whether there's a charge for the recheck.

**Diagnosis:** treatment_type boundary case — gold=Diagnostic Services (checking stitches = diagnostic evaluation), model often predicts Surgical Services (the follow-up is FOR a surgery). Both are defensible. appointment_booked=Inconclusive is debatable — caller already HAS an appointment (could argue Yes).

**No change recommended** — legitimate boundary case.

---

### 10. `CAL0198481b99eb767a890f8febab7b9b06` — Emergency Walk-In

**Gold labels:** Yes / New / Urgent Care / Sick Pet / (null)

**Transcript summary:** Caller is at a cottage with their dog (normally cared for in Whitby). Dog has a known shoulder condition and is now not weight-bearing on the other front leg. Calls Kingston Regional Pet Hospital. Agent tells them it's walk-in based and to come in.

**Diagnosis:** Gold labels are reasonable. Model sometimes predicts Inconclusive for appointment_booked because no specific time was confirmed — but the prompt now says "emergency walk-in where caller agrees to come in = Yes." Good test of that rule.

**No change recommended** — tests emergency walk-in handling.

---

## Summary of Recommended Changes

| Call ID | Field | Current Gold | Recommended | Confidence |
|---------|-------|-------------|-------------|------------|
| `CAL0039a82c...` | appointment_booked | Yes | Inconclusive | High |
| `CAL0039a82c...` | client_type | Existing | Inconclusive | High |
| `CAL0039a82c...` | treatment_type | Wellness Screening | N/A (missed call) | High |
| `CAL0198423279...` | client_type | Existing | New | High |
| `CAL0198423279...` | treatment_type | Preventive Care | Other | High |
| `CAL019838bea...` | treatment_type | Preventive Care | Other | High |
| `CAL019838be7...` | treatment_type | Wellness Screening | Wellness Screening (kept) | — |
| `CAL01983cac3...` | client_type | Existing | New | High |
| `CAL019847429...` | client_type | New | Existing | High |

**Applied: 8 field-level changes across 5 calls** (2026-03-14)
**Kept as-is: Call 4 (inter-clinic bloodwork) — treatment_type=Wellness Screening retained per user decision**
**No change: 3 calls (edge cases / ambiguous — calls 8, 9, 10)**

---

## Systematic Audit (2026-03-15) — Full 510 Dataset

### Audit Method
Ran automated transcript analysis against all 510 gold labels. Flagged calls where:
- Transcript content contradicts the gold label (wrong number, voicemail, no medical content)
- Cross-field consistency violated (appointment_booked=Yes + reason populated, No + no reason)
- Sub-category assigned but transcript lacks supporting keywords

### Findings Summary

| Severity | Count | Description |
|----------|:---:|-------------|
| HIGH | 4 | Wrong number with treatment type (3), Yes + reason populated (1) |
| MEDIUM | 119 | 97 specific sub-category with no detectable medical keywords, 21 No without reason, 1 admin with specific sub |
| LOW | 58 | Label/keyword mismatches (vaccination label but no vaccine mention, etc.) |

**Unique calls flagged: ~150/510 (29%)**

### HIGH Severity (require immediate fix)

| Call ID | Issue | Current Gold | Recommended |
|---------|-------|-------------|-------------|
| `CAL008a74d19188416c...` | Wrong number | tt=Preventive Care | tt=Other (already fixed in v1 audit) |
| `CAL019839456cd27079...` | Wrong number | tt=Preventive Care – Vaccinations, ab=Yes | Review — caller may have found right clinic after initial confusion |
| `CAL0198473f7d2a7b25...` | Yes + reason | ab=Yes, rnb=2b. Full schedule | rnb should be null (if appointment was booked) |
| `CAL0198490ce41a7081...` | Wrong number | tt=Retail – Food Orders | tt=Other or review transcript |

**Status:** [ ] Reviewed / [ ] Fixed

### MEDIUM Severity: appointment_booked=No with no reason (21 calls)

These calls have appointment_booked=No but reason_not_booked is empty. Per our classification rules, every No should have a reason. Need human review to assign appropriate reason categories.

**Status:** [ ] Reviewed / [ ] Fixed

### Analysis: NO_MEDICAL_WITH_SPECIFIC_SUB (97 calls)

97 calls (19% of dataset) have a specific sub-category label but no detectable medical keywords in the transcript. After expanding the keyword list, 25 of these have truly zero medical-related content. Most of the 97 are likely correct labels — the callers use natural language (brand names, colloquial terms) that keyword matching doesn't catch.

**Impact on model accuracy:** These 97 calls are disproportionately responsible for treatment_type errors. The model cannot determine the sub-category without medical context in the transcript. Options:
1. Downgrade these to parent categories (loses specificity but improves accuracy)
2. Accept ~65% as the ceiling for treatment_type given current gold labels
3. Manually review and fix the subset that are genuinely mislabeled

**Status:** [ ] Reviewed / [ ] Decision made
