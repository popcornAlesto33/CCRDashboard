#!/usr/bin/env python3
"""
Validate A1b architecture: 2-call pipeline with context passing.

  Call 1: appointment_booked (V9 prompt, strict schema, bs=15)
  Call 2: client_type + treatment_type + reason_not_booked
          (combined V9 prompts with appointment result as context, strict schema, bs=15)

Usage:
  python Scripts/validate_prompt_engineering_vA1b.py --max-calls 200 --batch-size 15
  python Scripts/validate_prompt_engineering_vA1b.py --batch-size 15  # full 510
  python Scripts/validate_prompt_engineering_vA1b.py --batch-size 15 --output results_a1b.json
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

try:
    from dotenv import load_dotenv
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
        "model_default": "gemini-2.5-flash",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_default": "https://api.openai.com/v1",
        "model_default": "gpt-4o-mini",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "base_url_default": "https://api.anthropic.com/v1/",
        "model_default": "claude-haiku-4-5-20251001",
    },
}

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_a1b")

# ============================================================
# V9 PROMPTS (Call 1: appointment_booked)
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

# Individual V9 prompts for client_type, treatment_type, reason_not_booked
# are no longer needed — they're integrated into A1B_COMBO_PROMPT above.

TREATMENT_TYPE_ENUMS = [
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
    "Other",
]

REASON_NOT_BOOKED_ENUMS = [
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
    None,
]

# ============================================================
# A1b CALL 2: COMBINED 3-FIELD PROMPT
# ============================================================

A1B_COMBO_PROMPT = """You are a veterinary medicine expert and call transcript analyst. You have deep knowledge of veterinary services, common symptoms, diagnostic pathways, and how clinics triage patients. Use this clinical expertise alongside the transcript to classify each field below.

The appointment_booked decision has already been made: {appointment_booked}

Classify the following 3 fields from this call transcript.

---

# FIELD 1: client_type

Determine whether the caller is a new or existing client at THIS specific clinic. "Existing" means the CALLER (not the pet) has been a client at this clinic before.

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

### New — no record found, agent creates file
Transcript: "Caller: I'd like to bring my cat in for a checkup. Agent: Sure, what's your last name? ... I'm not finding you in our system. Let me set up a new file."
Answer: New

---

# FIELD 2: treatment_type

Determine what veterinary service was discussed. Choose EXACTLY ONE category from the list below. Match the level of specificity that best fits — use a sub-category when the call is clearly about that specific service, use the parent when the call is general or covers multiple services.

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

---

# FIELD 3: reason_not_booked

Determine why an appointment was NOT booked.

## Rules

- If appointment_booked is "Yes": answer is null.
- If appointment_booked is "Inconclusive": answer is null (unless there's an explicit barrier like "we're fully booked").
- If appointment_booked is "No": choose the most specific matching category below.

## Key Distinctions

- **Parent vs sub-category:** Use the PARENT category (e.g., "1. Caller Procrastination" or "2. Scheduling Issue") unless the sub-category is a clear, unambiguous match. When in doubt, use the parent.
- **Price Objection (1a):** Use when the caller's PRIMARY reason for not booking was the price. The caller asked about cost and that was the deciding factor. If price was discussed but something ELSE was the real barrier (schedule full, service not offered, caller just procrastinating), use the real barrier instead.
- "I'll think about it" with NO price discussion → 1 (Procrastination)
- Caller cancels and says they'll reschedule later → 1 (Procrastination), NOT 9 (Client/appt query)
- Caller asked about a service the clinic doesn't offer → 3 (Service not offered), even if price was also discussed
- Wants same-day, told none available → 2a
- Schedule full for days/weeks → 2b
- If scheduling was the issue but you're unsure between 2a/2b/2c/2d → use parent "2. Scheduling Issue"
- Caller seeking a specific service style the clinic doesn't offer (holistic, outdoor euthanasia, exotics) → 3 (Service not offered)

## Examples

### Pure price shopping → 1a
Transcript: "Caller: How much for an exam? Agent: $122. Caller: Wow, okay. Thanks. Bye."
Answer: "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"
Why: Caller's sole purpose was pricing. Price was the barrier.

### Price asked BUT scheduling was the real barrier → Scheduling
Transcript: "Caller: How much for a spay? Agent: $350. Caller: Okay, when's the soonest? Agent: June 17th. Caller: That's too far out."
Answer: "2. Scheduling Issue"
Why: Caller accepted the price and asked to book. The scheduling gap was the real barrier.

### Caller cancels and will reschedule → Procrastination
Transcript: "Caller: I need to cancel Thursday's appointment, something came up. I'll call back next week."
Answer: "1. Caller Procrastination"
Why: Caller is postponing, not objecting to price or encountering a scheduling barrier.

### Service not offered → even if price discussed
Transcript: "Caller: Do you do holistic treatments? Agent: No. Caller: How much for a regular visit? Agent: $122. Caller: Okay, thanks."
Answer: "3. Service/treatment not offered"
Why: Primary reason: clinic doesn't offer what caller wanted. Price question was secondary.

### Same-day not available → 2a specifically
Transcript: "Caller: Can I get my puppy in today? He's been coughing. Agent: We're pretty booked, let me check... I'll take your number and call you back."
Answer: "2a. Scheduling Issue - Walk ins not available / no same day appt"
Why: Caller wanted same-day. Use 2a when the caller specifically needed today/now and was told none available.

### Caller canceled + no availability → still Procrastination
Transcript: "Caller: I need to cancel my appointment. Is there anything this week? Agent: We're pretty booked this week. Caller: Okay, I'll call back later."
Answer: "1. Caller Procrastination"
Why: The caller initiated the cancellation. Even though the schedule is full, the root cause is the caller's decision to cancel — not a scheduling barrier.

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

---

# Output Format

Return JSON with your reasoning and ALL 3 field answers:
{"reasoning": "...", "client_type": "New"|"Existing"|"Inconclusive", "treatment_type": "<exact category from list above>", "reason_not_booked": "<exact category or null>"}

Return JSON ONLY.""".strip()

A1B_COMBO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "a1b_combo_3field",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "client_type": {"type": "string", "enum": ["New", "Existing", "Inconclusive"]},
                "treatment_type": {"type": "string", "enum": TREATMENT_TYPE_ENUMS},
                "reason_not_booked": {"type": ["string", "null"], "enum": REASON_NOT_BOOKED_ENUMS},
            },
            "required": ["reasoning", "client_type", "treatment_type", "reason_not_booked"],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# RATE-LIMITED API CALLS
# ============================================================

_semaphore = threading.Semaphore(15)

# Token tracking
token_usage = {"call1": {"input": 0, "output": 0}, "call2": {"input": 0, "output": 0}}
_token_lock = threading.Lock()

PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def track_usage(resp, step: str):
    if resp.usage:
        # Map step names like "call2(appt=Yes)" → "call2"
        bucket = "call1" if step.startswith("call1") else "call2"
        with _token_lock:
            token_usage[bucket]["input"] += resp.usage.prompt_tokens or 0
            token_usage[bucket]["output"] += resp.usage.completion_tokens or 0


def _call_single(client, model, call_id, transcript, system_prompt, step, schema=None):
    """Single LLM call with retry and rate limiting."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                response_format=schema or {"type": "json_object"},
            )
            track_usage(resp, step)
            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step} {call_id} attempt {attempt+1}: {e}")
            else:
                logger.error(f"  {step} {call_id} failed: {e}")
                return {"call_id": call_id, "error": str(e)}
        finally:
            _semaphore.release()


def _call_batch(client, model, calls_batch, system_prompt, step, schema=None):
    """Batched LLM call: multiple transcripts in one request."""
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    batched_prompt = system_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently — do not let one transcript '
        'influence your classification of another.\n\n'
        'Return JSON: {"results": [{"call_id": "...", ...}, ...]}\n'
        'Return JSON ONLY.'
    )
    # Build batched schema
    batched_schema = None
    if schema:
        single_props = schema["json_schema"]["schema"]["properties"].copy()
        single_req = list(schema["json_schema"]["schema"]["required"])
        item_props = {**single_props, "call_id": {"type": "string"}}
        item_required = single_req + ["call_id"]
        batched_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema["json_schema"]["name"] + "_batch",
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

    messages = [
        {"role": "system", "content": batched_prompt},
        {"role": "user", "content": payload},
    ]
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                response_format=batched_schema or {"type": "json_object"},
            )
            track_usage(resp, step)
            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(it.get("call_id", "")): it for it in items}
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step} batch attempt {attempt+1}: {e}")
            else:
                logger.error(f"  {step} batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def run_field_batch(client, model, calls, system_prompt, batch_size, step, schema=None):
    """Run a field across all calls with batching and thread pool."""
    results = {}
    if batch_size <= 1:
        logger.info(f"  {step}: {len(calls)} calls individually")
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(_call_single, client, model, c["id"], c["transcript"],
                            system_prompt, step, schema): c["id"]
                for c in calls
            }
            for f in as_completed(futures):
                cid = futures[f]
                try:
                    results[cid] = f.result()
                except Exception as e:
                    results[cid] = {"call_id": cid, "error": str(e)}
    else:
        batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
        logger.info(f"  {step}: {len(calls)} calls in {len(batches)} batches of {batch_size}")
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(_call_batch, client, model, batch, system_prompt,
                            step, schema): i
                for i, batch in enumerate(batches)
            }
            for f in as_completed(futures):
                try:
                    results.update(f.result())
                except Exception as e:
                    logger.error(f"  {step} batch thread error: {e}")
    return results


# ============================================================
# A1b PIPELINE
# ============================================================

def run_a1b(client, model, calls, batch_size):
    """A1b: Call 1 (appointment_booked) → Call 2 (3 fields with context)."""

    # Call 1: appointment_booked
    logger.info(f"Call 1: appointment_booked ({len(calls)} calls, bs={batch_size})")
    appt_results = run_field_batch(
        client, model, calls, V9_APPOINTMENT_BOOKED_PROMPT, batch_size,
        "call1", V9_APPOINTMENT_BOOKED_SCHEMA,
    )
    logger.info(f"Call 1 done: {len(appt_results)} results")

    # Group calls by appointment answer
    calls_by_appt = {"Yes": [], "No": [], "Inconclusive": []}
    error_calls = []
    for c in calls:
        cid = c["id"]
        answer = appt_results.get(cid, {}).get("answer", "")
        if answer in calls_by_appt:
            calls_by_appt[answer].append(c)
        else:
            error_calls.append(c)

    logger.info(f"Appt split: Yes={len(calls_by_appt['Yes'])}, No={len(calls_by_appt['No'])}, "
                f"Inconclusive={len(calls_by_appt['Inconclusive'])}, error={len(error_calls)}")

    # Call 2: combo for each appointment group (parallel across groups)
    combo_results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for appt_answer, group_calls in calls_by_appt.items():
            if not group_calls:
                continue
            prompt = A1B_COMBO_PROMPT.replace("{appointment_booked}", appt_answer)
            futures[pool.submit(
                run_field_batch, client, model, group_calls, prompt,
                batch_size, f"call2(appt={appt_answer})", A1B_COMBO_SCHEMA,
            )] = appt_answer

        for future in as_completed(futures):
            appt_answer = futures[future]
            try:
                combo_results.update(future.result())
            except Exception as e:
                logger.error(f"  Call 2 (appt={appt_answer}) failed: {e}")

    logger.info(f"Call 2 done: {len(combo_results)} results")

    # Assemble predictions
    predictions = {}
    for c in calls:
        cid = c["id"]
        appt = appt_results.get(cid, {}).get("answer")
        combo = combo_results.get(cid, {})
        reason = combo.get("reason_not_booked")

        # Cross-field consistency
        if appt == "Yes" and reason is not None:
            reason = None
        if appt == "No" and reason is None:
            logger.warning(f"  {cid}: appt=No but reason=null")

        predictions[cid] = {
            "call_id": cid,
            "appointment_booked": appt,
            "client_type": combo.get("client_type"),
            "treatment_type": combo.get("treatment_type"),
            "reason_not_booked": reason,
        }

    return predictions


# ============================================================
# DATA LOADING
# ============================================================

def load_data(project_dir: str):
    """Load and join transcripts with labels."""
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
# SCORING
# ============================================================

def compute_accuracy(predictions, gold):
    """Compute accuracy per field."""
    fields = ["appointment_booked", "client_type", "treatment_type", "reason_not_booked"]
    results = {}

    for field in fields:
        correct = total = 0
        mismatches = []

        for cid, pred in predictions.items():
            if cid not in gold:
                continue
            gold_val = gold[cid]["labels"].get(field, "").strip()
            pred_val = (pred.get(field) or "").strip()

            if not gold_val or gold_val.lower() in ("null", "none"):
                gold_val = ""
            if not pred_val or pred_val.lower() in ("null", "none"):
                pred_val = ""

            if field == "reason_not_booked" and not gold_val and not pred_val:
                continue

            total += 1
            if gold_val == pred_val:
                correct += 1
            else:
                mismatches.append({"call_id": cid, "gold": gold_val or "(null)", "predicted": pred_val or "(null)"})

        results[field] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0,
            "mismatches": sorted(mismatches, key=lambda x: x["call_id"]),
        }

    return results


def print_results(results, mode, model):
    targets = {"appointment_booked": 0.90, "client_type": 0.90, "treatment_type": 0.80, "reason_not_booked": 0.85}

    print(f"\n{'='*70}")
    print(f"  A1b VALIDATION RESULTS — {mode}")
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

    # Treatment type confusion matrix
    tt = results["treatment_type"]["mismatches"]
    if tt:
        confusion = Counter((m["gold"], m["predicted"]) for m in tt)
        print("\n  Top treatment_type confusions:")
        for (gold, pred), count in confusion.most_common(10):
            print(f"    {gold!r} -> {pred!r}: {count}x")
        print()

    # Token costs
    c1 = token_usage["call1"]
    c2 = token_usage["call2"]
    total_in = c1["input"] + c2["input"]
    total_out = c1["output"] + c2["output"]
    price = PRICING.get(model, {"input": 0, "output": 0})
    cost = (total_in / 1_000_000 * price["input"]) + (total_out / 1_000_000 * price["output"])

    print(f"  Tokens: Call1={c1['input']+c1['output']:,} | Call2={c2['input']+c2['output']:,} | Total={total_in+total_out:,}")
    if price["input"] > 0:
        print(f"  Cost: ${cost:.4f}")
    print(f"{'='*70}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate A1b: 2-call pipeline with context passing")
    parser.add_argument("--max-calls", type=int, default=0, help="Max calls (0=all)")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for both calls")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--random", action="store_true", help="Randomly sample instead of first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="", help="Save results to JSON")
    parser.add_argument("--provider", default=None, choices=["gemini", "openai", "anthropic"])
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

    # Initialize client
    if args.provider:
        prov = PROVIDERS[args.provider]
        api_key = os.getenv(prov["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{prov['api_key_env']} not set")
        base_url = os.getenv(prov["base_url_env"], prov["base_url_default"])
        model = args.model or prov["model_default"]
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or LLM_API_KEY not set")
        base_url = DEFAULT_LLM_BASE_URL
        model = args.model or "gemini-2.5-flash"

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
    logger.info(f"Model: {model}, batch_size: {args.batch_size}")

    # Run A1b
    predictions = run_a1b(client, model, calls, args.batch_size)
    logger.info(f"Got {len(predictions)} predictions")

    # Score
    gold = {cid: data[cid] for cid in call_ids if cid in data}
    results = compute_accuracy(predictions, gold)
    print_results(results, f"A1b 2-call ({model}, bs={args.batch_size})", model)

    # Save
    if args.output:
        output_data = {
            "mode": f"a1b ({model}, bs={args.batch_size})",
            "num_calls": len(call_ids),
            "results": {
                field: {"accuracy": d["accuracy"], "correct": d["correct"],
                        "total": d["total"], "mismatches": d["mismatches"]}
                for field, d in results.items()
            },
            "predictions": predictions,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
