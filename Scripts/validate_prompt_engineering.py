#!/usr/bin/env python3
"""
Validate the two-model prompt engineering pipeline against labeled data.

Usage:
  python Scripts/validate_prompt_engineering.py --max-calls 50  # quick test
  python Scripts/validate_prompt_engineering.py                  # full 510 validation
  python Scripts/validate_prompt_engineering.py --single-model   # test legacy mode

Reads CallData/Transcript_details.csv + labels CSV, runs the pipeline,
and compares against human labels. Outputs accuracy per field.
"""
import os
import sys
import csv
import json
import random
import argparse
import logging
import threading
import time
from collections import Counter
from typing import Dict, List, Any

try:
    from dotenv import load_dotenv
    # .env lives in Scripts/ directory alongside this script
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from openai import OpenAI

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate")

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

## Few-Shot Examples

### Example: Existing client, urgent, appointment booked
Transcript excerpt: "Agent: What's going on with Luna? Caller: She's been vomiting foam and seems in pain. Agent: Let me get her in today — I have 9am open. Caller: That works, thank you."
Reasoning: Agent offers a specific time (9am), caller confirms. A date/time is confirmed.
Answer: Yes

### Example: Reschedule counts as Yes
Transcript excerpt: "Caller: I need to move Henry's appointment from Saturday to Tuesday. Agent: I have 2pm on Tuesday. Caller: Perfect, book that."
Reasoning: Caller already had an appointment and is rescheduling to a new confirmed time. They have a booking.
Answer: Yes

### Example: Schedule full, caller leaves without booking
Transcript excerpt: "Caller: I need a same-day appointment. Agent: We're fully booked today. I can give you the SmartVet number. Caller: Okay, thanks."
Reasoning: Caller wanted same-day, told none available, given alternative number. No appointment was scheduled. The caller chose to end the call.
Answer: No

### Example: Price shopping, no booking
Transcript excerpt: "Caller: How much to bring my dog in for an exam? Agent: It's $122 for the visit. Caller: Wow. Okay. Thank you."
Reasoning: Caller gathered pricing information and ended the call without scheduling. This is a completed interaction, not a pending one.
Answer: No

### Example: Emergency walk-in, no firm time
Transcript excerpt: "Caller: My dog ran into a porcupine, quills everywhere. Agent: You should come in to us, it'll be about $1020. Caller: Okay."
Reasoning: Agent tells caller to come in, caller says "okay" — but no specific time is set, and "okay" is ambiguous (could mean acknowledging price, not committing). Outcome depends on whether the caller actually comes in.
Answer: Inconclusive

### Example: Administrative call, no booking attempted
Transcript excerpt: "Caller: I'm calling to check on Max's blood test results. Agent: Results look normal, Dr. Chen will go over them at his next appointment."
Reasoning: This is an administrative call checking results. No appointment was being booked — the purpose was information retrieval. No booking was attempted or discussed.
Answer: Inconclusive

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "Yes" | "No" | "Inconclusive"
}

Return JSON ONLY. No commentary outside the JSON.""".strip()

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

## Few-Shot Examples

### Example: Price shopping — always 1a
Transcript excerpt: "Caller: How much to bring my dog in for an exam? Agent: It's $122. Caller: Wow. Okay. Thank you."
Reasoning: Caller asked about pricing ("how much"), received an answer, and ended the call without booking. Price was discussed, so this is always 1a regardless of whether there was an explicit objection.
Answer: "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"

### Example: No same-day availability
Transcript excerpt: "Caller: I need a same-day appointment for my cat. Agent: We're fully booked today, I can give you the SmartVet number. Caller: Okay, thanks."
Reasoning: Caller wanted same-day, told none available. This is a scheduling issue specifically about same-day/walk-in availability.
Answer: "2a. Scheduling Issue - Walk ins not available / no same day appt"

### Example: Appointment booked — null
Transcript excerpt: "Agent: We'll see you tomorrow at 2. Caller: Great, thanks!"
Reasoning: appointment_booked=Yes, so reason is null. No reasoning needed.
Answer: null

### Example: Emergency walk-in, Inconclusive — null
Transcript excerpt: "Caller: My dog ran into a porcupine. Agent: Come in to us, it'll be about $1020. Caller: Okay."
Reasoning: appointment_booked=Inconclusive. No explicit barrier in transcript — caller said "okay" and the outcome is pending. Default to null.
Answer: null

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "<exact category string>" | null
}

Return JSON ONLY. No commentary outside the JSON.""".strip()

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

## Few-Shot Examples

### Example: Symptoms only, no specific intervention — use PARENT
Transcript excerpt: "Caller: My cat has been limping since yesterday. Agent: Let's get her in tomorrow at 2 and the doctor will take a look."
Reasoning: Cat is limping (sick pet), but no specific intervention mentioned — agent says "the doctor will take a look." No specific intervention mentioned — using parent category.
Answer: Urgent Care / Sick Pet

### Example: Specific diagnostic intervention — use SUB-CATEGORY
Transcript excerpt: "Caller: Luna's been vomiting foam and seems in pain, shallow breathing. Agent: Let's get her in at 9am, we'll run some diagnostics."
Reasoning: Cat is vomiting with pain and breathing issues. Agent explicitly schedules a diagnostic appointment — this implies diagnosis and treatment of an illness. Specific intervention: diagnostic appointment for vomiting/pain.
Answer: Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)

### Example: Emergency — actual trauma/critical intervention
Transcript excerpt: "Caller: My dog ran into a porcupine, quills all over his face. Agent: You should come in to us right away, it'll be about $1020 for stabilization."
Reasoning: Porcupine quills require emergency removal/stabilization. This is actual emergency intervention (trauma), not routine urgent care. Specific intervention: stabilization for trauma.
Answer: Emergency & Critical Care – Stabilization (Trauma, Poisoning, Seizures)

### Example: Prescription refill — Retail
Transcript excerpt: "Caller: Henry is running out of his medication, can I get a refill? Agent: Sure, let me pull up his file."
Reasoning: Caller is refilling an existing prescription. This is retail, not preventive care. Specific intervention: prescription refill.
Answer: Retail – Prescriptions

### Example: Vaccinations as sole purpose — Preventive Care sub
Transcript excerpt: "Caller: I want to book shots for my dog Alan. Agent: Let me grab your phone number to open a file. How about tomorrow at 12?"
Reasoning: Sole stated purpose is "shots" (vaccinations). No other services discussed. Specific intervention: vaccinations only.
Answer: Preventive Care – Vaccinations

### Example: Price inquiry about general exam — Diagnostic Services parent
Transcript excerpt: "Caller: How much to bring my dog in for an exam? Agent: It's $122 for the visit. Caller: Wow. Okay. Thank you."
Reasoning: Caller asking about a general exam — no specific intervention mentioned. This is a diagnostic/evaluation visit. No specific intervention mentioned — using parent category.
Answer: Diagnostic Services

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

## Few-Shot Examples

### Example: Existing — agent finds file, references past care
Transcript excerpt: "Agent: What's the name? Caller: Luna, she's a tabby. Agent: I see Luna here, last visit was in March. What's going on with her?"
Reasoning: Agent looks up the pet and finds an existing record ("I see Luna here, last visit was in March"). Concrete signal: file found in system.
Answer: Existing

### Example: Existing — caller references ongoing medication
Transcript excerpt: "Caller: Henry is running out of his medication, can I get a refill? Agent: Let me pull up his file — yes, I see the prescription."
Reasoning: Caller has an existing prescription at this clinic, and agent finds the file. Concrete signal: ongoing medication prescribed here.
Answer: Existing

### Example: New — agent creates file
Transcript excerpt: "Caller: I want to book shots for my dog Alan. Agent: Let me grab your phone number to open a file. What's the number?"
Reasoning: Agent asks for phone number to "open a file" — this means creating a new record. Concrete signal: agent needs to CREATE a new file.
Answer: New

### Example: New — price shopping, unfamiliar with clinic
Transcript excerpt: "Caller: How much to bring my dog in for an exam? Agent: It's $122 for the visit. Caller: Wow. Okay. Thank you."
Reasoning: Caller asks about basic pricing as if unfamiliar with the clinic. No file lookup, no mention of past visits. Concrete signal: asking about pricing as if first-time.
Answer: New

## Output Format

Return JSON:
{
  "reasoning": "Your step-by-step reasoning citing specific transcript quotes",
  "answer": "New" | "Existing" | "Inconclusive"
}

Return JSON ONLY. No commentary outside the JSON.""".strip()

# ============================================================
# V9 PER-FIELD STRICT ENUM RESPONSE SCHEMAS
# ============================================================

V9_APPOINTMENT_BOOKED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "appointment_booked_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["Yes", "No", "Inconclusive"]}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

V9_CLIENT_TYPE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "client_type_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["New", "Existing", "Inconclusive"]}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

V9_TREATMENT_TYPE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "treatment_type_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {
                    "type": "string",
                    "enum": [
                        "Preventive Care",
                        "Preventive Care \u2013 Vaccinations",
                        "Preventive Care \u2013 Parasite Prevention",
                        "Preventive Care \u2013 Annual Exams",
                        "Preventive Care \u2013 Wellness Screening (Bloodwork, Urinalysis, Fecals)",
                        "Urgent Care / Sick Pet",
                        "Urgent Care \u2013 Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)",
                        "Urgent Care \u2013 Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)",
                        "Urgent Care \u2013 Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)",
                        "Surgical Services",
                        "Surgical Services \u2013 Spays and Neuters",
                        "Surgical Services \u2013 Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)",
                        "Surgical Services \u2013 Orthopedic Surgeries (ACL Repairs, Fracture Repair \u2014 Sometimes Referred Out)",
                        "Surgical Services \u2013 Emergency Surgeries (Pyometra, C-Sections, GDV)",
                        "Surgical Services \u2013 Dental Care (Cleanings, Extractions)",
                        "Diagnostic Services",
                        "Diagnostic Services \u2013 X-Rays (Digital Radiography)",
                        "Diagnostic Services \u2013 Ultrasound",
                        "Diagnostic Services \u2013 In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)",
                        "Diagnostic Services \u2013 ECG or Blood Pressure Monitoring",
                        "Emergency & Critical Care",
                        "Emergency & Critical Care \u2013 Stabilization (Trauma, Poisoning, Seizures)",
                        "Emergency & Critical Care \u2013 Overnight Hospitalization",
                        "Emergency & Critical Care \u2013 Fluid Therapy, Oxygen Therapy, Intensive Monitoring",
                        "Emergency & Critical Care \u2013 Referred to an Emergency Hospital",
                        "Dermatology",
                        "Dermatology \u2013 Allergies",
                        "Dermatology \u2013 Ear Infections",
                        "Retail",
                        "Retail \u2013 Food Orders",
                        "Retail \u2013 Prescriptions",
                        "End of Life Care",
                        "End of Life Care \u2013 In-Home Euthanasia",
                        "End of Life Care \u2013 In-Clinic Euthanasia",
                        "N/A (missed call)",
                        "Other"
                    ]
                }
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

V9_REASON_NOT_BOOKED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reason_not_booked_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {
                    "type": ["string", "null"],
                    "enum": [
                        "1. Caller Procrastination",
                        "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote",
                        "1b. Caller Procrastination - Need to check with partner",
                        "1c. Caller Procrastination - Getting information for someone else",
                        "2. Scheduling Issue",
                        "2a. Scheduling Issue - Walk ins not available / no same day appt",
                        "2b. Scheduling Issue - Full schedule",
                        "2c. Scheduling Issue - Not open / no availability on evenings",
                        "2d. Scheduling Issue - Not open / no availability on weekends",
                        "3. Service/treatment not offered",
                        "3a. Service/treatment not offered - Grooming",
                        "3b. Service/treatment not offered - Pet Adoption",
                        "3c. Service/treatment not offered - Exotics",
                        "3d. Service/treatment not offered - Farm / Large Animals",
                        "3e. Service/treatment not offered - Birds",
                        "3f. Service/treatment not offered - Reptiles",
                        "3g. Service/treatment not offered - Pocket Pets",
                        "4. Meant to call competitor hospital",
                        "5. Meant to call low cost / free service provider",
                        "6. Emergency care not offered",
                        "7. File Transferred",
                        "8. Medication/food order",
                        "9. Client/appt query (non-medical)",
                        "10. Missed call",
                        "11. No transcription",
                        None
                    ]
                }
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

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
    response_schema=None,
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
                response_format=response_schema or {"type": "json_object"},
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


def run_v9_field_batch(
    client: OpenAI,
    model: str,
    calls: List[Dict],
    system_prompt: str,
    batch_size: int,
    step_name: str,
    response_schema=None,
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
                response_schema,
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


def run_v9_reason_not_booked(
    client: OpenAI,
    model: str,
    calls: List[Dict],
    appointment_results: Dict[str, dict],
    batch_size: int,
    response_schema=None,
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
            response_schema,
        )
        results.update(llm_results)
    else:
        logger.info(f"  reason_not_booked: 0/{len(calls)} calls need LLM (no appointment_booked=No)")

    return results


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


def _load_module():
    """Import the main analysis script as a module."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "analyze_script",
        os.path.join(script_dir, "03_CallRail_Transcripts_Analyze_Buckets.py"),
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(project_dir: str):
    """Load and join transcripts with labels. Includes ALL examples (including few-shot)."""
    transcripts = {}
    transcript_path = os.path.join(project_dir, "CallData", "Transcript_details.csv")
    with open(transcript_path, "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] != "NULL" and row[1].strip():
                transcripts[row[0]] = row[1]

    labels = {}
    labels_path = os.path.join(project_dir, "CallData", "VetCare_CallInsight_Labels - labels.csv")
    with open(labels_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row

    joined = {}
    for cid in transcripts:
        if cid in labels:
            joined[cid] = {
                "id": cid,
                "transcript": transcripts[cid],
                "labels": labels[cid],
            }

    return joined


# ============================================================
# TOKEN / COST TRACKING
# ============================================================

# Gemini pricing (per 1M tokens) as of 2025
PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # OpenAI pricing for comparison
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

token_usage = {"reasoning": {"input": 0, "output": 0}, "classification": {"input": 0, "output": 0}}


def track_usage(resp, step: str):
    """Extract token usage from API response."""
    if resp.usage:
        token_usage[step]["input"] += resp.usage.prompt_tokens or 0
        token_usage[step]["output"] += resp.usage.completion_tokens or 0


def print_token_summary(reasoning_model: str, classification_model: str):
    """Print token usage and estimated cost."""
    r = token_usage["reasoning"]
    c = token_usage["classification"]
    total_input = r["input"] + c["input"]
    total_output = r["output"] + c["output"]

    print(f"\n{'='*70}")
    print(f"  TOKEN USAGE & COST ESTIMATE")
    print(f"{'='*70}")
    print(f"\n  Step 1 — Reasoning ({reasoning_model}):")
    print(f"    Input:  {r['input']:,} tokens")
    print(f"    Output: {r['output']:,} tokens")
    print(f"\n  Step 2 — Classification ({classification_model}):")
    print(f"    Input:  {c['input']:,} tokens")
    print(f"    Output: {c['output']:,} tokens")
    print(f"\n  Total:")
    print(f"    Input:  {total_input:,} tokens")
    print(f"    Output: {total_output:,} tokens")

    # Cost estimate
    r_price = PRICING.get(reasoning_model, {"input": 0, "output": 0})
    c_price = PRICING.get(classification_model, {"input": 0, "output": 0})

    r_cost = (r["input"] / 1_000_000 * r_price["input"]) + (r["output"] / 1_000_000 * r_price["output"])
    c_cost = (c["input"] / 1_000_000 * c_price["input"]) + (c["output"] / 1_000_000 * c_price["output"])
    total_cost = r_cost + c_cost

    if r_price["input"] > 0 or c_price["input"] > 0:
        print(f"\n  Estimated cost:")
        print(f"    Reasoning:      ${r_cost:.4f}")
        print(f"    Classification: ${c_cost:.4f}")
        print(f"    Total:          ${total_cost:.4f}")
        if total_input > 0:
            calls_count = sum(1 for v in token_usage.values() if v["input"] > 0)
            print(f"    Per-call avg:   ${total_cost / 50:.5f}")  # approximate
    else:
        print(f"\n  (No pricing data for {reasoning_model}/{classification_model})")

    print(f"{'='*70}\n")


# ============================================================
# API CALL FUNCTIONS
# ============================================================

def run_reasoning_batch(client: OpenAI, model: str, calls: List[Dict], batch_size: int) -> Dict[str, str]:
    """Run Step 1 reasoning on a list of calls."""
    mod = _load_module()

    all_reasoning = {}
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]

    for i, batch in enumerate(batches):
        logger.info(f"  Reasoning batch {i+1}/{len(batches)} ({len(batch)} calls)")
        payload = {
            "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in batch],
        }
        messages = [
            {"role": "system", "content": mod.REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            track_usage(resp, "reasoning")
            result = json.loads(resp.choices[0].message.content)
            for item in result.get("calls", []):
                cid = item.get("call_id")
                if cid:
                    all_reasoning[str(cid)] = item.get("reasoning", "")
        except Exception as e:
            logger.error(f"  Reasoning batch {i+1} failed: {e}")

    return all_reasoning


def run_classification_batch(
    client: OpenAI, model: str, reasoning_items: List[Dict], batch_size: int
) -> List[Dict]:
    """Run Step 2 classification on reasoning summaries."""
    mod = _load_module()

    all_results = []
    batches = [reasoning_items[i:i + batch_size] for i in range(0, len(reasoning_items), batch_size)]

    for i, batch in enumerate(batches):
        logger.info(f"  Classification batch {i+1}/{len(batches)} ({len(batch)} calls)")
        payload = {"calls": batch}
        messages = [
            {"role": "system", "content": mod.CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=mod.CLASSIFICATION_RESPONSE_SCHEMA,
            )
            track_usage(resp, "classification")
            result = json.loads(resp.choices[0].message.content)
            all_results.extend(result.get("calls", []))
        except Exception as e:
            logger.error(f"  Classification batch {i+1} failed: {e}")

    return all_results


def run_single_model_batch(
    client: OpenAI, model: str, calls: List[Dict], batch_size: int
) -> List[Dict]:
    """Run single-model mode."""
    mod = _load_module()

    all_results = []
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]

    for i, batch in enumerate(batches):
        logger.info(f"  Single-model batch {i+1}/{len(batches)} ({len(batch)} calls)")
        payload = {
            "analysis_version": "validation",
            "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in batch],
        }
        messages = [
            {"role": "system", "content": mod.SINGLE_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=mod.CLASSIFICATION_RESPONSE_SCHEMA,
            )
            track_usage(resp, "classification")
            result = json.loads(resp.choices[0].message.content)
            all_results.extend(result.get("calls", []))
        except Exception as e:
            logger.error(f"  Single-model batch {i+1} failed: {e}")

    return all_results


def _load_original_module():
    """Import the original (pre-v1) analysis script as a module."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "original_script",
        os.path.join(project_dir, "Script 03 OLD - ORIGINAL.py"),
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_original_prompt_batch(
    client: OpenAI, model: str, calls: List[Dict], batch_size: int
) -> List[Dict]:
    """Run the original (pre-v1) single-model prompt."""
    orig = _load_original_module()

    all_results = []
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]

    for i, batch in enumerate(batches):
        logger.info(f"  Original-prompt batch {i+1}/{len(batches)} ({len(batch)} calls)")
        payload = {
            "analysis_version": "validation_original",
            "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in batch],
        }
        messages = [
            {"role": "system", "content": orig.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            track_usage(resp, "classification")
            result = json.loads(resp.choices[0].message.content)
            all_results.extend(result.get("calls", []))
        except Exception as e:
            logger.error(f"  Original-prompt batch {i+1} failed: {e}")

    return all_results


def compute_accuracy(predictions: Dict[str, Dict], gold: Dict[str, Dict]):
    """Compute accuracy per field. Returns dict of field -> {correct, total, accuracy, mismatches}."""
    fields = ["appointment_booked", "client_type", "treatment_type", "reason_not_booked"]
    results = {}

    for field in fields:
        correct = 0
        total = 0
        mismatches = []

        for cid, pred in predictions.items():
            if cid not in gold:
                continue

            gold_val = gold[cid]["labels"].get(field, "").strip()
            pred_val = (pred.get(field) or "").strip()

            # Normalize empty/null
            if not gold_val or gold_val.lower() in ("null", "none"):
                gold_val = ""
            if not pred_val or pred_val.lower() in ("null", "none"):
                pred_val = ""

            # For reason_not_booked, only score when gold has a value OR pred has a value
            if field == "reason_not_booked" and not gold_val and not pred_val:
                continue

            total += 1
            if gold_val == pred_val:
                correct += 1
            else:
                mismatches.append({
                    "call_id": cid,
                    "gold": gold_val or "(null)",
                    "predicted": pred_val or "(null)",
                })

        accuracy = correct / total if total > 0 else 0
        results[field] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "mismatches": sorted(mismatches, key=lambda x: x["call_id"]),
        }

    return results


def print_results(results: Dict, mode: str):
    """Print formatted accuracy results."""
    targets = {
        "appointment_booked": 0.90,
        "client_type": 0.90,
        "treatment_type": 0.80,
        "reason_not_booked": 0.85,
    }

    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS -- {mode} mode")
    print(f"{'='*70}")

    for field, data in results.items():
        target = targets.get(field, 0)
        status = "PASS" if data["accuracy"] >= target else "FAIL"
        print(f"\n  {field}:")
        print(f"    Accuracy: {data['accuracy']:.1%} ({data['correct']}/{data['total']}) "
              f"[target: {target:.0%}] [{status}]")

        if data["mismatches"]:
            print(f"    Mismatches ({len(data['mismatches'])}):")
            for m in data["mismatches"][:10]:
                print(f"      {m['call_id']}: gold={m['gold']!r} vs pred={m['predicted']!r}")
            if len(data["mismatches"]) > 10:
                print(f"      ... and {len(data['mismatches']) - 10} more")

    print(f"\n{'='*70}")

    # Treatment type confusion matrix (top misclassifications)
    tt_mismatches = results["treatment_type"]["mismatches"]
    if tt_mismatches:
        confusion = Counter((m["gold"], m["predicted"]) for m in tt_mismatches)
        print("\n  Top treatment_type confusions:")
        for (gold, pred), count in confusion.most_common(10):
            print(f"    {gold!r} -> {pred!r}: {count}x")
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate prompt engineering pipeline")
    parser.add_argument("--max-calls", type=int, default=0, help="Max calls to validate (0=all)")
    parser.add_argument("--reasoning-batch-size", type=int, default=4)
    parser.add_argument("--classification-batch-size", type=int, default=8)
    parser.add_argument("--reasoning-model", default=os.getenv("REASONING_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--classification-model", default=os.getenv("CLASSIFICATION_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--single-model", action="store_true", help="Test single-model mode")
    parser.add_argument("--random", action="store_true", help="Randomly sample calls instead of first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="", help="Save detailed results to JSON file")
    parser.add_argument("--pipeline", default="v8", choices=["v8", "v9", "original"],
                        help="Pipeline mode: v8 (two-model), v9 (field-decomposed), or original (pre-v1 prompt)")
    parser.add_argument("--v9-batch-size", type=int, default=2,
                        help="Batch size for v9 field classifiers (default: 2)")
    args = parser.parse_args()

    # Load data
    data = load_data(project_dir)
    logger.info(f"Loaded {len(data)} labeled examples")

    # Sample or limit
    call_ids = sorted(data.keys())
    if args.max_calls > 0:
        if args.random:
            random.seed(args.seed)
            call_ids = random.sample(call_ids, min(args.max_calls, len(call_ids)))
        else:
            call_ids = call_ids[:args.max_calls]
    logger.info(f"Validating {len(call_ids)} calls")

    calls = [{"id": cid, "transcript": data[cid]["transcript"]} for cid in call_ids]

    # Initialize LLM client via OpenAI-compatible API
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not found in environment or .env")
    client = OpenAI(api_key=api_key, base_url=DEFAULT_LLM_BASE_URL, max_retries=3)

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
                V9_APPOINTMENT_BOOKED_SCHEMA,
            )
            treatment_future = phase1_pool.submit(
                run_v9_field_batch,
                client, args.reasoning_model, calls,
                V9_TREATMENT_TYPE_PROMPT, args.v9_batch_size, "treatment_type",
                V9_TREATMENT_TYPE_SCHEMA,
            )
            client_future = phase1_pool.submit(
                run_v9_field_batch,
                client, args.reasoning_model, calls,
                V9_CLIENT_TYPE_PROMPT, args.v9_batch_size, "client_type",
                V9_CLIENT_TYPE_SCHEMA,
            )

            appt_results = appt_future.result()
            treatment_results = treatment_future.result()
            client_results = client_future.result()

        logger.info(f"Phase 1 complete: appt={len(appt_results)}, treatment={len(treatment_results)}, client={len(client_results)}")

        # Phase 2: reason_not_booked (depends on appointment_booked)
        logger.info("Phase 2: reason_not_booked (depends on appointment_booked)...")
        reason_results = run_v9_reason_not_booked(
            client, args.reasoning_model, calls, appt_results, args.v9_batch_size,
            V9_REASON_NOT_BOOKED_SCHEMA,
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

    elif args.pipeline == "original":
        logger.info(f"Running original (pre-v1) prompt with {args.classification_model}")
        results_list = run_original_prompt_batch(
            client, args.classification_model, calls, args.classification_batch_size
        )
        mode = f"original-prompt ({args.classification_model})"
        predictions = {}
        for item in results_list:
            cid = item.get("call_id")
            if cid:
                if item.get("appointment_booked") == "Yes":
                    item["reason_not_booked"] = None
                predictions[str(cid)] = item

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


if __name__ == "__main__":
    main()
