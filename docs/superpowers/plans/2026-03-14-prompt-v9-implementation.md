# Prompt v9: Field-Decomposed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 2-stage LLM pipeline with 4 parallel field-specific Pro classifiers + Python assembly, breaking cascading errors and enabling per-field iteration.

**Architecture:** 4 parallel Gemini 2.5 Pro calls (one per field, each reads raw transcript), with reason_not_booked depending on appointment_booked. Python assembly applies deterministic rules and builds final JSON. Validation script orchestrates via shared ThreadPoolExecutor(15) with global rate limiter.

**Tech Stack:** Python 3, OpenAI SDK (Gemini-compatible), concurrent.futures, existing CSV data loaders.

**Spec:** `docs/superpowers/specs/2026-03-14-prompt-v9-pipeline-redesign.md`

---

## Chunk 1: Field-Specific Prompts

### Task 1: Define appointment_booked prompt

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py` (add prompt constant at top)

- [ ] **Step 1: Write the prompt constant**

Add after the imports and before `load_data()`:

```python
# ============================================================
# V9 FIELD-SPECIFIC PROMPTS
# ============================================================

V9_APPOINTMENT_BOOKED_PROMPT = """You are a veterinary call transcript analyst. Your ONLY task is to determine whether an appointment was booked in this call.

Read the transcript and determine: Did the caller book an appointment?

## Answer: Yes, No, or Inconclusive

**EVALUATION ORDER (follow these steps in sequence):**

1. FIRST, check if YES:
   - A specific date/time is confirmed
   - At an emergency/walk-in clinic: agent tells caller to come in and caller agrees — even without a specific time
   - Caller already has a confirmed appointment and is calling to adjust it (reschedule, cancellation list) — they already have a booking
   - Agent says "we'll see you at [time]" and caller confirms

2. SECOND, check if NO (the caller chose not to book, even if politely):
   - Caller asks about pricing, gets info, and ends call without scheduling
   - Caller is told the schedule is full and ends the call
   - Caller says "I'll think about it" / "I'll call back" / "let me talk to my partner"
   - Caller calls for advice only and never intended to book (medication questions, symptom questions)
   - Caller gathers information (services, hours, pricing) and ends call without scheduling — this is a completed interaction, not a pending one
   - The call ends naturally after the caller's question was answered, with no mention of scheduling

3. ONLY IF NEITHER APPLIES, use INCONCLUSIVE:
   - Inconclusive means the outcome depends on a future event that hasn't happened yet
   - Call goes to voicemail or automated system
   - Clinic will call back ("the doctor will review and we'll get back to you") — outcome depends on a future CLINIC action
   - Inter-clinic consultation where no direct booking occurs
   - The call is administrative (checking results, asking about records) — no appointment was the purpose

If you can determine what the CALLER decided, it is not Inconclusive.

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "Yes" | "No" | "Inconclusive"
}

Return JSON ONLY. No commentary outside the JSON.""".strip()
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add appointment_booked field-specific prompt"
```

---

### Task 2: Define reason_not_booked prompt

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Write the prompt constant**

Add after `V9_APPOINTMENT_BOOKED_PROMPT`:

```python
V9_REASON_NOT_BOOKED_PROMPT = """You are a veterinary call transcript analyst. Your ONLY task is to determine why an appointment was NOT booked in this call.

You are given:
1. The raw transcript
2. The appointment_booked decision (already determined): {appointment_booked}

## Rules

- If appointment_booked is "Yes": output null immediately. No reasoning needed.
- If appointment_booked is "Inconclusive": output null. Exception: only populate if the transcript contains an explicit, clear barrier (e.g., "we're fully booked so you'll have to call back").
- If appointment_booked is "No": determine the specific reason from the categories below.

## Categories

Choose the MOST SPECIFIC matching category:

- "1. Caller Procrastination" — caller says "I'll think about it" / "I'll call back" with NO price discussion
- "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote" — pricing/cost discussed AT ANY POINT and caller doesn't book. If they asked "how much?" and didn't book, this is ALWAYS 1a
- "1b. Caller Procrastination - Need to check with partner"
- "1c. Caller Procrastination - Getting information for someone else" — caller explicitly on behalf of someone else, or inter-clinic call
- "2. Scheduling Issue"
- "2a. Scheduling Issue - Walk ins not available / no same day appt" — wants same-day/walk-in, told none available
- "2b. Scheduling Issue - Full schedule" — wants upcoming appointment, schedule full for multiple days/weeks
- "2c. Scheduling Issue - Not open / no availability on evenings"
- "2d. Scheduling Issue - Not open / no availability on weekends"
- "3. Service/treatment not offered"
- "3a. Service/treatment not offered - Grooming"
- "3b. Service/treatment not offered - Pet Adoption"
- "3c. Service/treatment not offered - Exotics"
- "3d. Service/treatment not offered - Farm / Large Animals"
- "3e. Service/treatment not offered - Birds"
- "3f. Service/treatment not offered - Reptiles"
- "3g. Service/treatment not offered - Pocket Pets"
- "4. Meant to call competitor hospital" — caller dialed the wrong clinic
- "5. Meant to call low cost / free service provider"
- "6. Emergency care not offered"
- "7. File Transferred"
- "8. Medication/food order"
- "9. Client/appt query (non-medical)" — caller had a medical need but only made an administrative inquiry. ONLY for appointment_booked=No, never for Inconclusive.
- "10. Missed call"
- "11. No transcription"

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "<exact category string>" | null
}

Return JSON ONLY. No commentary outside the JSON.""".strip()
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add reason_not_booked field-specific prompt"
```

---

### Task 3: Define treatment_type prompt

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Write the prompt constant**

Add after `V9_REASON_NOT_BOOKED_PROMPT`:

```python
V9_TREATMENT_TYPE_PROMPT = """You are a veterinary call transcript analyst. Your ONLY task is to determine what veterinary service was discussed in this call.

Read the transcript and classify the service into EXACTLY ONE of the categories listed below.

## CRITICAL: Parent vs. Sub-Category

When in doubt between a parent category and a sub-category, ALWAYS use the parent. Over-specification is the #1 error pattern.

- Use the PARENT category when the transcript describes a general concern WITHOUT a specific intervention
- Only use a SUB-CATEGORY when the transcript names or implies THE SPECIFIC INTERVENTION

**NOT specific interventions (use PARENT):** "we'll take a look", "bring them in for an exam", "the doctor will check", "physical examination", "we'll see what's going on"

**Specific interventions (required for sub-category):** "we'll run bloodwork", "we need to do X-rays", "start her on antibiotics", "dental cleaning scheduled", "allergy testing"

In your reasoning, you MUST state: (a) the specific intervention mentioned, OR (b) "no specific intervention mentioned — using parent category."

## Key Rules

### Emergency vs. Urgent Care
Classify by ACTUAL SERVICE, not hospital name:
- Emergency & Critical Care: actual emergency intervention (trauma, poisoning, seizures, overnight hospitalization, directed to emergency clinic)
- Urgent Care / Sick Pet: sick pet needing prompt attention but routine interaction (advice, medication questions, stable-patient triage) — even at an emergency hospital

### Preventive Care: Parent vs. Sub
- Parent: general wellness visits, new pet checkups, multiple preventive services discussed
- Sub only when that service is the SOLE AND EXPLICIT purpose ("I need to get my dog his shots" = Vaccinations)

### Wellness Screening vs. Diagnostic Lab
- Routine bloodwork (annual, pre-op, wellness) = Preventive Care – Wellness Screening
- Symptom-driven bloodwork (investigating a problem) = Diagnostic Services – Lab Testing

### Dermatology
Use only when PRIMARY reason is skin, coat, ear, or allergy issue. Do not use when skin/ear is secondary to a more urgent concern.

### Retail
- Refilling existing prescription = Retail – Prescriptions
- New flea/tick/heartworm prevention plan = Preventive Care – Parasite Prevention

### Dental
- Dental cleanings and extractions = Surgical Services – Dental Care
- Routine dental checkup as part of wellness = Preventive Care

### "Other" — LAST RESORT
Before using "Other", verify ALL of these:
1. NOT about a sick pet, injury, or medical concern
2. NOT about scheduling/rescheduling any appointment type
3. NOT about medications, food, or prescriptions
4. NOT a missed call or voicemail with no content
5. The topic genuinely does not fit ANY existing category

## Categories

Choose EXACTLY ONE:

Preventive Care
Preventive Care – Vaccinations
Preventive Care – Parasite Prevention
Preventive Care – Annual Exams
Preventive Care – Wellness Screening (Bloodwork, Urinalysis, Fecals)
Urgent Care / Sick Pet
Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)
Urgent Care – Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)
Urgent Care – Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)
Surgical Services
Surgical Services – Spays and Neuters
Surgical Services – Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)
Surgical Services – Orthopedic Surgeries (ACL Repairs, Fracture Repair — Sometimes Referred Out)
Surgical Services – Emergency Surgeries (Pyometra, C-Sections, GDV)
Surgical Services – Dental Care (Cleanings, Extractions)
Diagnostic Services
Diagnostic Services – X-Rays (Digital Radiography)
Diagnostic Services – Ultrasound
Diagnostic Services – In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)
Diagnostic Services – ECG or Blood Pressure Monitoring
Emergency & Critical Care
Emergency & Critical Care – Stabilization (Trauma, Poisoning, Seizures)
Emergency & Critical Care – Overnight Hospitalization
Emergency & Critical Care – Fluid Therapy, Oxygen Therapy, Intensive Monitoring
Emergency & Critical Care – Referred to an Emergency Hospital
Dermatology
Dermatology – Allergies
Dermatology – Ear Infections
Retail
Retail – Food Orders
Retail – Prescriptions
End of Life Care
End of Life Care – In-Home Euthanasia
End of Life Care – In-Clinic Euthanasia
N/A (missed call)
Other

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes and naming the intervention (or stating none)",
  "answer": "<exact category string from list above>"
}

Return JSON ONLY. No commentary outside the JSON.""".strip()
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add treatment_type field-specific prompt"
```

---

### Task 4: Define client_type prompt

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Write the prompt constant**

Add after `V9_TREATMENT_TYPE_PROMPT`:

```python
V9_CLIENT_TYPE_PROMPT = """You are a veterinary call transcript analyst. Your ONLY task is to determine whether the caller is a new or existing client at THIS specific clinic.

"Existing" means the CALLER has been a client at THIS specific clinic before. Not the pet — the caller.

## Signals

**Existing (need at least one concrete signal):**
- Agent looks up file/account and FINDS it for this caller
- Pet already in the system under this caller's name
- Caller references a past visit AT THIS CLINIC or ongoing medication prescribed here
- Caller uses a specific doctor's name at this clinic

**New:**
- Caller asks "do you accept new patients?"
- Asks about location/hours/pricing as if unfamiliar
- Agent asks for phone number to CREATE a new file
- Agent says "no record found"
- Caller mentions they have a vet elsewhere

**Edge cases (classify as New):**
- New owner of a pet with an existing file from previous owner = New
- Caller has a vet at a different clinic, calling this one for the first time = New
- Caller has one pet on file but calling about a brand new pet with no history = lean New

**Inconclusive:** Should be extremely rare. Only use when there are genuinely zero signals either way AND the transcript provides no clues.

Casual or friendly tone alone does NOT indicate Existing — require concrete evidence.

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "New" | "Existing" | "Inconclusive"
}

Return JSON ONLY. No commentary outside the JSON.""".strip()
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add client_type field-specific prompt"
```

---

## Chunk 2: Pipeline Orchestration

### Task 5: Add rate-limited field classifier function

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Add imports and rate limiter**

Add `import threading` and `import time` to the imports at top (time may already be there — only add if missing). Add after the prompt constants:

```python
# ============================================================
# V9 PIPELINE: RATE-LIMITED FIELD CLASSIFIER
# ============================================================

# Global rate limiter: max 15 concurrent Pro requests
_v9_semaphore = threading.Semaphore(15)

def run_v9_field_call(
    client: OpenAI,
    model: str,
    call_id: str,
    transcript: str,
    system_prompt: str,
    step_name: str,
) -> dict:
    """Run a single field-specific LLM call with rate limiting and retry.

    Returns {"call_id": ..., "reasoning": ..., "answer": ...} or
            {"call_id": ..., "error": ...} on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    for attempt in range(3):
        _v9_semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            # Accumulate all v9 field usage into "reasoning" bucket (all use Pro)
            track_usage(resp, "reasoning")
            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 1.0
                logger.warning(f"  {step_name} {call_id} attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  {step_name} {call_id} failed after 3 attempts: {e}")
                return {"call_id": call_id, "error": str(e)}
        finally:
            _v9_semaphore.release()
```

- [ ] **Step 2: Add batched field classifier function**

Add after `run_v9_field_call`:

```python
def run_v9_field_batch(
    client: OpenAI,
    model: str,
    calls: List[Dict],
    system_prompt: str,
    batch_size: int,
    step_name: str,
) -> dict:
    """Run a field classifier for all calls using thread pool.

    Each call gets its own API call (one transcript per request).
    The global semaphore (_v9_semaphore) handles rate limiting.
    batch_size is reserved for future multi-transcript batching but
    currently unused — all calls run as individual requests via thread pool.

    Returns {call_id: {"reasoning": ..., "answer": ...}}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    logger.info(f"  {step_name}: processing {len(calls)} calls via thread pool")

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(
                run_v9_field_call,
                client, model, c["id"], c["transcript"], system_prompt, step_name,
            ): c["id"]
            for c in calls
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                results[cid] = result
            except Exception as e:
                logger.error(f"  {step_name} {cid} thread error: {e}")
                results[cid] = {"call_id": cid, "error": str(e)}

    return results
```

- [ ] **Step 3: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add rate-limited field classifier functions"
```

---

### Task 6: Add reason_not_booked with appointment_booked dependency

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Add reason_not_booked runner that injects appointment_booked**

Add after `run_v9_field_batch`:

```python
def run_v9_reason_not_booked(
    client: OpenAI,
    model: str,
    calls: List[Dict],
    appointment_results: Dict[str, dict],
    batch_size: int,
) -> dict:
    """Run reason_not_booked with appointment_booked injected into the prompt.

    For calls where appointment_booked is Yes or Inconclusive, short-circuits to null.
    Only calls the LLM for appointment_booked=No.
    """
    results = {}
    calls_needing_llm = []

    for c in calls:
        cid = c["id"]
        appt = appointment_results.get(cid, {})
        appt_answer = appt.get("answer", "")

        if appt_answer == "Yes":
            results[cid] = {"call_id": cid, "reasoning": "appointment_booked=Yes, skipping", "answer": None}
        elif appt_answer == "Inconclusive":
            results[cid] = {"call_id": cid, "reasoning": "appointment_booked=Inconclusive, defaulting to null", "answer": None}
        elif appt_answer == "No":
            calls_needing_llm.append(c)
        else:
            # appointment_booked failed — skip
            results[cid] = {"call_id": cid, "reasoning": f"appointment_booked unavailable ({appt_answer}), skipping", "answer": None}

    if calls_needing_llm:
        logger.info(f"  reason_not_booked: {len(calls_needing_llm)}/{len(calls)} calls need LLM (appointment_booked=No)")
        prompt = V9_REASON_NOT_BOOKED_PROMPT.replace("{appointment_booked}", "No")
        llm_results = run_v9_field_batch(
            client, model, calls_needing_llm, prompt, batch_size, "reason_not_booked",
        )
        results.update(llm_results)
    else:
        logger.info(f"  reason_not_booked: 0/{len(calls)} calls need LLM (no appointment_booked=No)")

    return results
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add reason_not_booked runner with cascade break"
```

---

### Task 7: Add Python assembly function

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Add the assembly function**

Add after `run_v9_reason_not_booked`:

```python
def v9_assemble(
    calls: List[Dict],
    appt_results: Dict[str, dict],
    reason_results: Dict[str, dict],
    treatment_results: Dict[str, dict],
    client_results: Dict[str, dict],
) -> Dict[str, Dict]:
    """Assemble final predictions from 4 field outputs. Pure Python, no LLM.

    Applies cross-field consistency rules:
    - appointment_booked=Yes + reason populated -> null out reason, log warning
    - appointment_booked=No + reason=null -> log warning
    """
    predictions = {}

    for c in calls:
        cid = c["id"]
        appt = appt_results.get(cid, {}).get("answer")
        reason = reason_results.get(cid, {}).get("answer")
        treatment = treatment_results.get(cid, {}).get("answer")
        client = client_results.get(cid, {}).get("answer")

        # Cross-field consistency
        if appt == "Yes" and reason is not None:
            logger.warning(f"  Assembly {cid}: appointment_booked=Yes but reason_not_booked={reason!r} — setting to null")
            reason = None
        if appt == "No" and reason is None:
            logger.warning(f"  Assembly {cid}: appointment_booked=No but reason_not_booked is null — flagging for review")

        predictions[cid] = {
            "call_id": cid,
            "appointment_booked": appt,
            "client_type": client,
            "treatment_type": treatment,
            "reason_not_booked": reason,
            # TODO: Name extraction stubbed out for v9 initial implementation.
            # Names are not scored in accuracy metrics so this doesn't affect validation.
            # Can be added later by parsing reasoning outputs or regex on raw transcript.
            "stated_hospital_name": None,
            "stated_patient_name": None,
            "agent_name": None,
        }

    return predictions
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: add Python assembly with consistency checks"
```

---

### Task 8: Wire up the v9 pipeline in main()

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py`

- [ ] **Step 1: Add --pipeline v9 CLI argument**

In the `main()` function, after the existing `parser.add_argument` calls (around line 347), add:

```python
    parser.add_argument("--pipeline", default="v8", choices=["v8", "v9"],
                        help="Pipeline mode: v8 (two-model) or v9 (field-decomposed)")
    parser.add_argument("--v9-batch-size", type=int, default=2,
                        help="Batch size for v9 field classifiers (default: 2)")
```

- [ ] **Step 2: Replace the entire pipeline execution + results section in main()**

Replace everything from `if args.single_model:` (line 372) through the end of `main()` (line 431) with the following single block. This handles all three pipeline modes (v9, single-model, two-model) and the shared results/output section:

```python
    # Initialize optional debug variables
    reasoning_debug = None  # v9 mode
    reasoning = None  # two-model mode

    if args.pipeline == "v9":
        logger.info(f"Running v9 field-decomposed pipeline ({args.reasoning_model})")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Phase 1: appointment_booked, treatment_type, client_type in parallel
        logger.info("Phase 1: appointment_booked + treatment_type + client_type (parallel)...")

        with ThreadPoolExecutor(max_workers=3) as phase1_pool:
            appt_future = phase1_pool.submit(
                run_v9_field_batch,
                client, args.reasoning_model, calls,
                V9_APPOINTMENT_BOOKED_PROMPT, args.v9_batch_size, "appointment_booked",
            )
            treatment_future = phase1_pool.submit(
                run_v9_field_batch,
                client, args.reasoning_model, calls,
                V9_TREATMENT_TYPE_PROMPT, args.v9_batch_size, "treatment_type",
            )
            client_future = phase1_pool.submit(
                run_v9_field_batch,
                client, args.reasoning_model, calls,
                V9_CLIENT_TYPE_PROMPT, args.v9_batch_size, "client_type",
            )

            appt_results = appt_future.result()
            treatment_results = treatment_future.result()
            client_results = client_future.result()

        logger.info(f"Phase 1 complete: appt={len(appt_results)}, treatment={len(treatment_results)}, client={len(client_results)}")

        # Phase 2: reason_not_booked (depends on appointment_booked)
        logger.info("Phase 2: reason_not_booked (depends on appointment_booked)...")
        reason_results = run_v9_reason_not_booked(
            client, args.reasoning_model, calls, appt_results, args.v9_batch_size,
        )
        logger.info(f"Phase 2 complete: reason={len(reason_results)}")

        # Phase 3: Python assembly
        logger.info("Phase 3: Assembly...")
        predictions = v9_assemble(calls, appt_results, reason_results, treatment_results, client_results)

        mode = f"v9-field-decomposed ({args.reasoning_model})"

        # Save reasoning for debugging
        reasoning_debug = {
            cid: {
                "appointment_booked": appt_results.get(cid, {}),
                "reason_not_booked": reason_results.get(cid, {}),
                "treatment_type": treatment_results.get(cid, {}),
                "client_type": client_results.get(cid, {}),
            }
            for cid in [c["id"] for c in calls]
        }

    elif args.single_model:
        logger.info(f"Running single-model validation with {args.classification_model}")
        results_list = run_single_model_batch(
            client, args.classification_model, calls, args.classification_batch_size
        )
        mode = f"single-model ({args.classification_model})"
        predictions = {}
        for item in results_list:
            cid = item.get("call_id")
            if cid:
                if item.get("appointment_booked") == "Yes":
                    item["reason_not_booked"] = None
                predictions[str(cid)] = item

    else:
        logger.info(f"Running two-model validation: {args.reasoning_model} -> {args.classification_model}")

        logger.info("Step 1: Reasoning...")
        reasoning = run_reasoning_batch(client, args.reasoning_model, calls, args.reasoning_batch_size)
        logger.info(f"Step 1 complete: {len(reasoning)} reasoning summaries")

        logger.info("Step 2: Classification...")
        reasoning_items = [{"call_id": cid, "reasoning": r} for cid, r in reasoning.items()]
        results_list = run_classification_batch(
            client, args.classification_model, reasoning_items, args.classification_batch_size
        )
        mode = f"two-model ({args.reasoning_model} -> {args.classification_model})"
        predictions = {}
        for item in results_list:
            cid = item.get("call_id")
            if cid:
                if item.get("appointment_booked") == "Yes":
                    item["reason_not_booked"] = None
                predictions[str(cid)] = item

    # ---- Shared results section (all pipeline modes) ----

    logger.info(f"Got {len(predictions)} predictions")

    # Compute accuracy
    gold = {cid: data[cid] for cid in call_ids if cid in data}
    results = compute_accuracy(predictions, gold)
    print_results(results, mode)
    print_token_summary(args.reasoning_model, args.classification_model)

    # Save detailed output if requested
    if args.output:
        output_data = {
            "mode": mode,
            "num_calls": len(call_ids),
            "results": {
                field: {
                    "accuracy": d["accuracy"],
                    "correct": d["correct"],
                    "total": d["total"],
                    "mismatches": d["mismatches"],
                }
                for field, d in results.items()
            },
            "predictions": predictions,
        }
        if reasoning_debug is not None:
            output_data["field_reasoning"] = reasoning_debug
        if reasoning is not None:
            output_data["reasoning"] = {cid: r for cid, r in reasoning.items() if cid in predictions}

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Detailed results saved to {args.output}")
```

- [ ] **Step 3: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('Scripts/validate_prompt_engineering.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: wire up v9 pipeline in main() with --pipeline flag"
```

---

## Chunk 3: Baseline & First Run

### Task 9: Run v8 baseline against corrected gold labels

This establishes the clean baseline post-gold-label fixes.

**Files:**
- No code changes — just run the existing pipeline

- [ ] **Step 1: Run v8 baseline**

Run:
```bash
python Scripts/validate_prompt_engineering.py --max-calls 50 --output results_v8_baseline_corrected.json
```

Expected: Accuracy results printed. Save the numbers — these are the "before" for v9 comparison.

- [ ] **Step 2: Record baseline in test_results.md**

Add a new section to `tasks/test_results.md`:

```markdown
## v8 Rebaseline (corrected gold labels)

**Results:** appt X% | client X% | treatment X% | reason X%

**Note:** Same v8 prompts, but 8 gold label corrections applied (see tasks/gold_label_audit.md). This is the true baseline for v9 comparison.
```

- [ ] **Step 3: Commit**

```bash
git add tasks/test_results.md
git commit -m "prompt v8: rebaseline with corrected gold labels"
```

---

### Task 10: Run v9 pipeline first test

**Files:**
- No code changes — run the new pipeline

- [ ] **Step 1: Run v9 pipeline**

Run:
```bash
python Scripts/validate_prompt_engineering.py --max-calls 50 --pipeline v9 --v9-batch-size 2 --output results_v9_initial.json
```

Expected: Accuracy results for all 4 fields. Compare against v8 baseline.

- [ ] **Step 2: Audit mismatches**

For each mismatch in the output:
1. Read the transcript
2. Check if the model's prediction is reasonable
3. If gold label looks wrong → document in `tasks/gold_label_audit.md` (do NOT modify CSV without human sign-off)
4. If model is wrong → note the error pattern for prompt iteration

- [ ] **Step 3: Record results in test_results.md**

Add:

```markdown
## v9.0 — Field-Decomposed Pipeline (initial)

**Architecture:** 4 parallel Pro calls (appointment_booked, treatment_type, client_type, reason_not_booked) + Python assembly
**Model:** gemini-2.5-pro, batch_size=2

**Results:** appt X% | client X% | treatment X% | reason X%

**Key findings:**
- [fill in after run]
```

- [ ] **Step 4: Commit**

```bash
git add tasks/test_results.md
git commit -m "prompt v9.0: initial field-decomposed pipeline results"
```

---

## Chunk 4: Per-Field Prompt Iteration

### Task 11: Iterate on appointment_booked prompt

This task is executed by a dedicated subagent (Subagent A).

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py` — `V9_APPOINTMENT_BOOKED_PROMPT`

- [ ] **Step 1: Analyze v9.0 appointment_booked mismatches**

Read `results_v9_initial.json`. For each appointment_booked mismatch:
- Read the transcript
- Understand why the model predicted what it did
- Identify the error pattern (No→Inconclusive? Inconclusive→No? Yes→No?)

- [ ] **Step 2: Adjust the prompt**

Based on error patterns, tweak `V9_APPOINTMENT_BOOKED_PROMPT`. Common adjustments:
- If too many Inconclusive→No: soften the No criteria
- If too many No→Inconclusive: add more examples of what counts as No
- Add specific examples for the most common error patterns

- [ ] **Step 3: Test appointment_booked only**

Run the full pipeline but focus on appointment_booked accuracy:
```bash
python Scripts/validate_prompt_engineering.py --max-calls 50 --pipeline v9 --v9-batch-size 2 --output results_v9_appt_iter1.json
```

- [ ] **Step 4: Audit mismatches, check gold labels**

For each remaining mismatch, verify gold label correctness. Document any new findings in `tasks/gold_label_audit.md`.

- [ ] **Step 5: Repeat steps 2-4** until appointment_booked reaches 88%+ (leave final 2pp for tuning after all fields stabilize)

- [ ] **Step 6: Commit final prompt**

```bash
git add Scripts/validate_prompt_engineering.py
git commit -m "prompt v9: iterate appointment_booked prompt to X%"
```

---

### Task 12: Iterate on treatment_type prompt (parallel with Task 11)

This task is executed by a dedicated subagent (Subagent B).

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py` — `V9_TREATMENT_TYPE_PROMPT`

- [ ] **Step 1–6: Same process as Task 11** but for treatment_type

Focus on:
- Parent vs sub-category errors
- Wellness Screening vs Diagnostic Lab
- Emergency vs Urgent Care
- Target: 75%+ (leave headroom for final tuning)

---

### Task 13: Iterate on client_type prompt (parallel with Tasks 11-12)

This task is executed by a dedicated subagent (Subagent C).

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py` — `V9_CLIENT_TYPE_PROMPT`

- [ ] **Step 1–6: Same process as Task 11** but for client_type

Focus on:
- "THIS clinic" rule effectiveness
- New owner of existing patient edge cases
- Target: 90%+ (already near target)

---

### Task 14: Iterate on reason_not_booked prompt (after Task 11 stabilizes)

This task is executed by a dedicated subagent (Subagent D). **Starts after Task 11 (appointment_booked) reaches 88%+.**

**Files:**
- Modify: `Scripts/validate_prompt_engineering.py` — `V9_REASON_NOT_BOOKED_PROMPT`

- [ ] **Step 1–6: Same process as Task 11** but for reason_not_booked

Focus on:
- Price Objection vs Procrastination confusion
- Cascading error reduction (should be much better now with cascade break)
- Target: 65%+ (realistic first milestone)

---

## Chunk 5: Final Validation & Cleanup

### Task 15: Full 50-call validation with all iterated prompts

**Files:**
- No code changes

- [ ] **Step 1: Run final v9 validation**

```bash
python Scripts/validate_prompt_engineering.py --max-calls 50 --pipeline v9 --v9-batch-size 2 --output results_v9_final.json
```

- [ ] **Step 2: Compare against v8 baseline**

Print side-by-side comparison of v8 baseline vs v9 final for all 4 fields.

- [ ] **Step 3: Final gold label audit**

Review all remaining mismatches. Document any new gold label issues.

- [ ] **Step 4: Record final results in test_results.md**

- [ ] **Step 5: Commit and push**

```bash
git add Scripts/validate_prompt_engineering.py tasks/test_results.md tasks/gold_label_audit.md
git commit -m "prompt v9: final results — appt X% | client X% | treatment X% | reason X%"
git push origin main
```

---

### Task 16: Batch size experimentation (optional)

Only if v9 meets targets at batch_size=2.

- [ ] **Step 1: Test batch_size=4**

```bash
python Scripts/validate_prompt_engineering.py --max-calls 50 --pipeline v9 --v9-batch-size 4 --output results_v9_bs4.json
```

- [ ] **Step 2: Compare accuracy**

If accuracy drops >2pp on any field, stay at batch_size=2.
If accuracy holds, batch_size=4 is preferred for production (fewer API calls).

- [ ] **Step 3: Record findings**

Document batch size impact in test_results.md.
