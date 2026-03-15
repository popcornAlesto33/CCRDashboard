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

# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

PROVIDERS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url_default": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "reasoning_model_default": "gemini-2.5-pro",
        "classification_model_default": "gemini-2.5-flash",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_default": "https://api.openai.com/v1",
        "reasoning_model_default": "gpt-5",
        "classification_model_default": "gpt-4o-mini",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "base_url_default": "https://api.anthropic.com/v1/",
        "reasoning_model_default": "claude-sonnet-4-5-20250514",
        "classification_model_default": "claude-haiku-4-5-20251001",
    },
}

# Legacy fallback: LLM_API_KEY / LLM_BASE_URL still work if no provider is specified
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate")

# ============================================================
# V9 FIELD-SPECIFIC PROMPTS
# ============================================================

V9_APPOINTMENT_BOOKED_PROMPT = """You are a veterinary call transcript analyst. Determine whether an appointment was booked in this call.

## Categories

- **Yes**: An appointment was confirmed (date/time set, caller agreed to come in, or existing appointment adjusted)
- **No**: The caller did not end up with an appointment — includes: chose not to book, schedule was full, was put on a cancellation list, or clinic offered to check and call back
- **Inconclusive**: The call was not about booking (administrative, checking results, voicemail) so appointment status doesn't apply

Key distinction: if the caller TRIED to book but couldn't (schedule full, no same-day, put on callback list), that is **No** — they left the call without an appointment. Inconclusive is for calls where booking was never the purpose.

## Examples

### Yes — urgent booking
Transcript: "Agent: Can you come in at 9am? Caller: Yes, that works."
Answer: Yes

### No — got info and left
Transcript: "Caller: How much for an exam? Agent: $122. Caller: Okay, thanks. Bye."
Answer: No

### No — schedule full, clinic will check and call back
Transcript: "Agent: We're booked up, but let me take your number and see if we can squeeze you in. Caller: Okay, it's 555-1234."
Answer: No

### Inconclusive — admin call, not about booking
Transcript: "Caller: I'm calling to check on Max's blood test results. Agent: Results look normal."
Answer: Inconclusive

## Output Format

Return JSON with your reasoning and answer:
{"reasoning": "...", "answer": "Yes" | "No" | "Inconclusive"}

Return JSON ONLY.""".strip()

V9_REASON_NOT_BOOKED_PROMPT = """You are a veterinary call transcript analyst. Determine why an appointment was NOT booked in this call.

The appointment_booked decision has already been made: {appointment_booked}

## Rules

- If appointment_booked is "Yes": answer is null.
- If appointment_booked is "Inconclusive": answer is null (unless there's an explicit barrier like "we're fully booked").
- If appointment_booked is "No": choose the most specific matching category below.

## Key Distinctions

- **Parent vs sub-category:** Use the PARENT category (e.g., "1. Caller Procrastination" or "2. Scheduling Issue") unless the sub-category is a clear, unambiguous match. When in doubt, use the parent.
- **Price Objection (1a):** If pricing/cost was discussed AT ANY POINT and the caller didn't book → use 1a. This is aggressive by design — any price discussion + no booking = 1a.
- "I'll think about it" with NO price discussion → 1 (Procrastination)
- Caller cancels and says they'll reschedule later → 1 (Procrastination), NOT 9 (Client/appt query)
- Wants same-day, told none available → 2a
- Schedule full for days/weeks → 2b
- If scheduling was the issue but you're unsure between 2a/2b/2c/2d → use parent "2. Scheduling Issue"

## Examples

### Price discussed + no booking → always 1a
Transcript: "Caller: How much for a spay? Agent: $350. Caller: Okay, I'll think about it."
Answer: "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"
Why: Price was discussed and no booking was made. Always 1a when price comes up.

### Caller cancels and will reschedule → Procrastination
Transcript: "Caller: I need to cancel Thursday's appointment, something came up. I'll call back next week."
Answer: "1. Caller Procrastination"
Why: Caller is postponing, not objecting to price or encountering a scheduling barrier.

### Schedule too far out → Scheduling Issue (parent)
Transcript: "Caller: When's the soonest for a spay? Agent: June 17th. Caller: That's too far out. I'll look elsewhere."
Answer: "2. Scheduling Issue"
Why: Scheduling availability is the barrier. Use parent unless clearly same-day (2a) or multi-week full (2b).

## Categories

- "1. Caller Procrastination"
- "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"
- "1b. Caller Procrastination - Need to check with partner"
- "1c. Caller Procrastination - Getting information for someone else"
- "2. Scheduling Issue"
- "2a. Scheduling Issue - Walk ins not available / no same day appt"
- "2b. Scheduling Issue - Full schedule"
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
- "4. Meant to call competitor hospital"
- "5. Meant to call low cost / free service provider"
- "6. Emergency care not offered"
- "7. File Transferred"
- "8. Medication/food order"
- "9. Client/appt query (non-medical)"
- "10. Missed call"
- "11. No transcription"

## Output Format

Return JSON with your reasoning and answer:
{"reasoning": "...", "answer": "<exact category string>" | null}

Return JSON ONLY.""".strip()

V9_TREATMENT_TYPE_PROMPT = """You are a veterinary call transcript analyst. Determine what veterinary service was discussed in each call.

Choose EXACTLY ONE category from the list below. Match the level of specificity that best fits — use a sub-category when the call is clearly about that specific service, use the parent when the call is general or covers multiple services.

## Key Guidelines

- Classify based on the PRIMARY reason for the call
- If a sick pet has specific symptoms (vomiting, ear infection, limping), use the relevant sub-category — you don't need a named procedure
- Emergency & Critical Care requires actual emergency-level situations (trauma, poisoning, critical symptoms) — not just calling an emergency hospital
- Routine bloodwork (annual, wellness, pre-op) = Preventive Care – Wellness Screening, NOT Diagnostic Services. Only use Diagnostic when the bloodwork is investigating a specific problem.
- Admin/rescheduling calls: if the transcript mentions WHAT the appointment is for (surgery, sick visit, etc.), classify by THAT service — only default to Preventive Care when the service type is truly unknown
- "Other" is a LAST RESORT. Before using Other, check:
  - Rescheduling/cancelling calls → classify by the underlying service if mentioned, else Preventive Care
  - Nail trims, microchip scans, general checkups → Preventive Care
  - Medication ordering for an existing condition → Retail – Prescriptions
  - Service not offered (exotics, grooming) → still use the relevant category (e.g., "Surgical Services" if they asked about a procedure)
  - If the caller mentioned ANY medical concern → classify by that concern

## Examples

### Sick cat with specific symptoms → sub-category
Transcript: "Caller: Luna's been vomiting foam and seems in pain, shallow breathing. Agent: Let's get her in at 9am."
Answer: Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)
Why: Specific symptoms (vomiting, pain) clearly map to diagnosis/treatment of illness.

### Dog limping, no further detail → parent
Transcript: "Caller: My dog's been limping. Agent: Let's get him in tomorrow. Caller: Okay, thanks."
Answer: Urgent Care / Sick Pet
Why: Limping could be many things — not enough to pick a specific sub-category.

### Ear problem → Dermatology sub
Transcript: "Caller: His ear is all black inside. Agent: We should take a look at that."
Answer: Dermatology – Ear Infections
Why: Ear issue is the primary complaint, maps directly to the ear infections sub-category.

### Solely vaccinations → Preventive Care sub
Transcript: "Caller: I want to book shots for my dog. Agent: We have tomorrow at 12."
Answer: Preventive Care – Vaccinations
Why: Sole stated purpose is vaccinations, nothing else discussed.

### Annual checkup → Annual Exams sub
Transcript: "Caller: I need to schedule a yearly appointment for my cat. Agent: We have an opening on Thursday."
Answer: Preventive Care – Annual Exams
Why: Caller explicitly requests annual/yearly appointment.

### Vaccines + checkup → Vaccinations (primary purpose wins)
Transcript: "Caller: I need to bring my dog in for his booster shot. Agent: We can also do a quick checkup while he's here."
Answer: Preventive Care – Vaccinations
Why: Primary purpose is vaccinations. The checkup is secondary — classify by what the caller called about.

### Prescription refill → Retail
Transcript: "Caller: Henry's running out of his medication, can I get a refill?"
Answer: Retail – Prescriptions
Why: Existing prescription refill, not a new medical concern.

### Porcupine quills, emergency → Emergency
Transcript: "Caller: Dog ran into a porcupine, quills everywhere. Agent: Come in right away, it'll be $1020."
Answer: Emergency & Critical Care – Stabilization (Trauma, Poisoning, Seizures)
Why: Acute trauma requiring emergency intervention.

### Multiple services including surgery → prioritize surgery
Transcript: "Caller: I need to book a spay and vaccines for my dog."
Answer: Surgical Services – Spays and Neuters
Why: When multiple services are discussed, prioritize the most significant medical procedure.

### Rescheduling call, no medical detail → Preventive Care (default)
Transcript: "Caller: I need to move my appointment to next week. Agent: Sure, how about Tuesday at 3?"
Answer: Preventive Care
Why: Admin/rescheduling call with no medical content discussed. Default to Preventive Care, NOT Other.

### Routine bloodwork → Wellness Screening, NOT Diagnostic
Transcript: "Caller: I'd like to schedule bloodwork for my senior dog, he's due for his annual check. Agent: Sure, we can do that Thursday."
Answer: Preventive Care – Wellness Screening (Bloodwork, Urinalysis, Fecals)
Why: Routine/annual bloodwork is Wellness Screening under Preventive Care, not Diagnostic Services.

### Surgery rescheduling → classify by the surgery, not as Preventive Care
Transcript: "Caller: I need to reschedule Bella's spay from next week. Agent: How about the 15th?"
Answer: Surgical Services – Spays and Neuters
Why: Even though this is a rescheduling call, the underlying service (spay) is mentioned — classify by that.

### Elderly dog declining → End of Life Care, not Urgent Care
Transcript: "Caller: My 16-year-old lab isn't eating and can barely walk anymore. We think it might be time. Agent: I'm so sorry. Would you like to discuss options?"
Answer: End of Life Care
Why: Context (very old pet, "might be time") indicates end-of-life discussion, not just a sick pet visit.

## Categories

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

Return JSON with your reasoning and answer:
{"reasoning": "...", "answer": "<exact category from list above>"}

Return JSON ONLY.""".strip()

V9_CLIENT_TYPE_PROMPT = """You are a veterinary call transcript analyst. Determine whether the caller is a new or existing client at THIS specific clinic.

"Existing" means the CALLER (not the pet) has been a client at this clinic before.

## Signals

**Existing:** Agent finds their file, pet already in system, caller references past visits here, knows doctor names
**New:** Asks "do you accept new patients?", unfamiliar with pricing/location, agent creates new file, mentions having a vet elsewhere

Casual/friendly tone alone does not indicate Existing — require concrete evidence.
Inconclusive should be extremely rare.

## Examples

### Existing — agent finds file
Transcript: "Agent: What's the name? Caller: Luna. Agent: I see Luna here, last visit was March."
Answer: Existing

### New — agent creates file
Transcript: "Caller: I want to book shots for my dog. Agent: Let me grab your number to open a file."
Answer: New

### New — price shopping, unfamiliar
Transcript: "Caller: How much for an exam? Agent: $122. Caller: Okay thanks. Bye."
Answer: New

## Output Format

Return JSON with your reasoning and answer:
{"reasoning": "...", "answer": "New" | "Existing" | "Inconclusive"}

Return JSON ONLY.""".strip()

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


def _make_batched_prompt(base_prompt: str) -> str:
    """Convert a single-transcript prompt into a multi-transcript batched prompt."""
    return base_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. Process EACH transcript independently — do not let one transcript influence your classification of another.\n\n'
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", "answer": ...}, ...]}\n'
        'Return JSON ONLY.'
    )


def _make_batched_schema(single_schema: dict) -> dict:
    """Convert a single-result schema into a batched results schema."""
    single_item = single_schema["json_schema"]["schema"].copy()
    item_props = {**single_item["properties"], "call_id": {"type": "string"}}
    item_required = list(single_item["required"]) + ["call_id"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": single_schema["json_schema"]["name"] + "_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_props,
                            "required": item_required,
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            }
        }
    }


def _run_v9_multi_call(client, model, calls_batch, system_prompt, step_name, response_schema=None):
    """Send multiple transcripts in a single API call. Returns {call_id: result}."""
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    for attempt in range(3):
        _v9_semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                response_format=response_schema or {"type": "json_object"},
            )
            track_usage(resp, "reasoning")
            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(item.get("call_id", "")): item for item in items}
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 1.0
                logger.warning(f"  {step_name} batch attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  {step_name} batch failed after 3 attempts: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
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
    """Run a field classifier for all calls.

    If batch_size <= 1: one transcript per API call (original approach).
    If batch_size > 1: multiple transcripts per API call (batched).
    All batches run concurrently via thread pool.

    Returns {call_id: {"reasoning": ..., "answer": ...}}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    if batch_size <= 1:
        # Single-transcript mode
        logger.info(f"  {step_name}: processing {len(calls)} calls individually via thread pool")
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
    else:
        # Multi-transcript batched mode
        batched_prompt = _make_batched_prompt(system_prompt)
        batched_schema = _make_batched_schema(response_schema) if response_schema else None
        batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
        logger.info(f"  {step_name}: processing {len(calls)} calls in {len(batches)} batches of {batch_size}")

        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(
                    _run_v9_multi_call,
                    client, model, batch, batched_prompt, step_name, batched_schema,
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"  {step_name} batch thread error: {e}")

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


import re as _re

_MEDICAL_KEYWORDS = _re.compile(
    r'(sick|vomit|limp|pain|bleed|itch|sneez|cough|infect|allergy|diarrhea|seiz|surgery|spay|neuter'
    r'|dental|vaccin|shot|booster|rabies|bloodwork|blood.?work|blood test|x-ray|xray|ultrasound'
    r'|emergency|euthan|lump|swollen|limping|not eating|not drinking|discharge|medication|prescription'
    r'|refill|pills?|tablets?|apoquel|cytopoint|gabapentin|bravecto|interceptor|revolution|advantage'
    r'|frontline|heartgard|metacam|rimadyl|prednisone|doxycycline|amoxicillin|clavamox|cerenia'
    r'|zylkene|cosequin|food order|kibble|diet|royal canin|hills|purina|science diet'
    r'|flea|tick|heartworm|worm|deworming|annual|yearly|wellness|checkup|check.?up'
    r'|fecal|urine|stool|anal gland|abscess|hot spot|rash|bump|growth|mass|wound|bite|sting'
    r'|broken|fracture|torn|rupture|blocked|obstruct|put down|put.{0,5}sleep|quality of life'
    r'|microchip|nail trim|groom|boarding|kennel|records|transfer|file'
    r'|eye.{0,10}(?:red|swollen|discharge)|ear.{0,10}(?:smell|black|discharge)'
    r'|shak|trembl|collaps|faint|convuls|panting|drool|foam)',
    _re.IGNORECASE,
)


def flag_transcript(transcript: str) -> List[str]:
    """Flag transcript quality/ambiguity issues. Returns list of flag strings.

    Flags:
        very_short: <3 speaker turns or <100 chars
        voicemail: caller speaks <5 words (automated greeting)
        garbled: >5% of words are redacted [MEDICAL_CONDITION] etc.
        wrong_number: caller says "wrong number"
        no_medical_content: no medical keywords detected + not a reschedule/admin call
        admin_no_medical: rescheduling/admin call with no medical keywords
    """
    flags = []
    turns = len(_re.findall(r'(Agent:|Caller:)', transcript))
    caller_parts = _re.findall(r'Caller:\s*(.*?)(?=Agent:|$)', transcript, _re.DOTALL)
    caller_words = sum(len(p.split()) for p in caller_parts)
    total_words = len(transcript.split())
    redacted = len(_re.findall(r'\[(MEDICAL_CONDITION|DRUG|MEDICAL_PROCESS|INJURY)\]', transcript))

    if turns < 3 or len(transcript) < 100:
        flags.append('very_short')
    if caller_words < 5:
        flags.append('voicemail')
    if total_words > 0 and redacted / total_words > 0.05:
        flags.append('garbled')
    if _re.search(r'wrong number|called the wrong', transcript, _re.IGNORECASE) and len(transcript) < 500:
        flags.append('wrong_number')

    has_medical = bool(_MEDICAL_KEYWORDS.search(transcript))
    is_reschedule = bool(_re.search(
        r'reschedule|cancel.{1,15}appointment|move.{1,15}appointment|change.{1,15}appointment',
        transcript, _re.IGNORECASE,
    ))

    if not has_medical:
        if is_reschedule:
            flags.append('admin_no_medical')
        elif turns >= 4:
            flags.append('no_medical_content')

    return flags


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

        # Transcript quality flags
        transcript_flags = flag_transcript(c.get("transcript", ""))

        predictions[cid] = {
            "call_id": cid,
            "appointment_booked": appt,
            "client_type": client,
            "treatment_type": treatment,
            "reason_not_booked": reason,
            "transcript_flags": transcript_flags,
            # TODO: Name extraction stubbed out for v9 initial implementation.
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
    parser.add_argument("--reasoning-model", default=None,
                        help="Override reasoning model (default: provider's default)")
    parser.add_argument("--classification-model", default=None,
                        help="Override classification model (default: provider's default)")
    parser.add_argument("--single-model", action="store_true", help="Test single-model mode")
    parser.add_argument("--random", action="store_true", help="Randomly sample calls instead of first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="", help="Save detailed results to JSON file")
    parser.add_argument("--pipeline", default="v8", choices=["v8", "v9", "original"],
                        help="Pipeline mode: v8 (two-model), v9 (field-decomposed), or original (pre-v1 prompt)")
    parser.add_argument("--v9-batch-size", type=int, default=2,
                        help="Batch size for v9 field classifiers (default: 2)")
    parser.add_argument("--provider", default=None, choices=["gemini", "openai", "anthropic"],
                        help="LLM provider (reads provider-specific keys from .env). "
                             "If omitted, falls back to LLM_API_KEY/LLM_BASE_URL.")
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

    # Initialize LLM client — provider-aware
    if args.provider:
        prov = PROVIDERS[args.provider]
        api_key = os.getenv(prov["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{prov['api_key_env']} not found in environment or .env")
        base_url = os.getenv(prov["base_url_env"], prov["base_url_default"])
        # Apply provider defaults for models if not overridden
        if not args.reasoning_model:
            args.reasoning_model = prov["reasoning_model_default"]
        if not args.classification_model:
            args.classification_model = prov["classification_model_default"]
        logger.info(f"Provider: {args.provider} | base_url: {base_url}")
    else:
        # Legacy: use LLM_API_KEY / LLM_BASE_URL
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY not found in environment or .env. Use --provider or set LLM_API_KEY.")
        base_url = DEFAULT_LLM_BASE_URL
        if not args.reasoning_model:
            args.reasoning_model = os.getenv("REASONING_MODEL", "gemini-2.5-pro")
        if not args.classification_model:
            args.classification_model = os.getenv("CLASSIFICATION_MODEL", "gemini-2.5-flash")

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
    logger.info(f"Models: reasoning={args.reasoning_model}, classification={args.classification_model}")

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
