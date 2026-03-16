#!/usr/bin/env python3
"""
Test: Free-text classification + separate enum formatting step.

Compares three architectures against A1b baseline:
  A1b (baseline): Flash classify with strict JSON schema (existing)
  D (3-step):     Flash classify free-text → Flash format to enum (per field)
  E (shared):     Flash classify free-text → ONE Flash call formats ALL fields

Hypothesis: removing strict enum constraints lets the model think more freely,
then a cheap formatting call maps to enums without the logit masking penalty.

Usage:
  python Scripts/test_freetext_formatter.py --max-calls 100
  python Scripts/test_freetext_formatter.py --max-calls 50 --fields treatment_type reason_not_booked
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_freetext")

# ============================================================
# PROMPTS — same classification prompts but WITHOUT strict schemas
# ============================================================

# Import the actual prompts from validate script
sys.path.insert(0, script_dir)
from validate_prompt_engineering import (
    V9_APPOINTMENT_BOOKED_PROMPT,
    V9_CLIENT_TYPE_PROMPT,
    V9_TREATMENT_TYPE_PROMPT,
    V9_REASON_NOT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA,
    V9_CLIENT_TYPE_SCHEMA,
    V9_TREATMENT_TYPE_SCHEMA,
    V9_REASON_NOT_BOOKED_SCHEMA,
    load_data,
)

# Import A1b 2-step prompts from Script 03
sys.path.insert(0, script_dir)
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("script03", os.path.join(script_dir, "03_CallRail_Transcripts_Analyze_Buckets.py"))
_mod = module_from_spec(_spec)
# Prevent script03 from executing its main logic
_orig_argv = sys.argv
sys.argv = [""]
try:
    _spec.loader.exec_module(_mod)
except SystemExit:
    pass
finally:
    sys.argv = _orig_argv
REASONING_SYSTEM_PROMPT = _mod.REASONING_SYSTEM_PROMPT
CLASSIFICATION_SYSTEM_PROMPT = _mod.CLASSIFICATION_SYSTEM_PROMPT
CLASSIFICATION_RESPONSE_SCHEMA = _mod.CLASSIFICATION_RESPONSE_SCHEMA

# Override track_usage to handle arbitrary step names
token_usage = {"input": 0, "output": 0}
_token_lock = threading.Lock()

def track_usage(resp, step: str):
    """Safe token tracker that accepts any step name."""
    if resp.usage:
        with _token_lock:
            token_usage["input"] += resp.usage.prompt_tokens or 0
            token_usage["output"] += resp.usage.completion_tokens or 0

# ============================================================
# ENUM LISTS (for the formatter step)
# ============================================================

APPOINTMENT_BOOKED_ENUMS = ["Yes", "No", "Inconclusive"]

CLIENT_TYPE_ENUMS = ["New", "Existing", "Inconclusive"]

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
    None,  # null is valid
]

FIELD_ENUMS = {
    "appointment_booked": APPOINTMENT_BOOKED_ENUMS,
    "client_type": CLIENT_TYPE_ENUMS,
    "treatment_type": TREATMENT_TYPE_ENUMS,
    "reason_not_booked": REASON_NOT_BOOKED_ENUMS,
}

FIELD_PROMPTS = {
    "appointment_booked": V9_APPOINTMENT_BOOKED_PROMPT,
    "client_type": V9_CLIENT_TYPE_PROMPT,
    "treatment_type": V9_TREATMENT_TYPE_PROMPT,
    "reason_not_booked": V9_REASON_NOT_BOOKED_PROMPT,
}

FIELD_SCHEMAS = {
    "appointment_booked": V9_APPOINTMENT_BOOKED_SCHEMA,
    "client_type": V9_CLIENT_TYPE_SCHEMA,
    "treatment_type": V9_TREATMENT_TYPE_SCHEMA,
    "reason_not_booked": V9_REASON_NOT_BOOKED_SCHEMA,
}

# ============================================================
# FORMATTER PROMPTS
# ============================================================

PER_FIELD_FORMATTER_PROMPT = """You are a formatting assistant. Map the free-text classification below to the EXACT matching enum value.

## Free-text classification:
{free_text}

## Valid enum values:
{enum_list}

Rules:
- Pick the SINGLE best match from the enum list
- Use the EXACT string from the list (copy it character-for-character)
- If the free-text says null, none, or not applicable, answer null
- Do NOT re-analyze or second-guess — just map the text to the closest enum

Return JSON: {{"answer": "<exact enum value or null>"}}
Return JSON ONLY.""".strip()

SHARED_FORMATTER_PROMPT = """You are a formatting assistant. Map each free-text classification to the EXACT matching enum value from its respective list.

## Free-text classifications:
{classifications_json}

## Valid enum values per field:
{enums_json}

Rules:
- For EACH field, pick the SINGLE best match from that field's enum list
- Use the EXACT strings from the lists (copy character-for-character)
- If a free-text value says null, none, or not applicable, set answer to null
- Do NOT re-analyze or second-guess the classifications — just format them
- Process each field INDEPENDENTLY — do not let one field's answer influence another

Return JSON: {{"results": {{"appointment_booked": "<exact enum>", "client_type": "<exact enum>", "treatment_type": "<exact enum>", "reason_not_booked": "<exact enum or null>"}}}}
Return JSON ONLY.""".strip()

# ============================================================
# API CALL INFRASTRUCTURE
# ============================================================

_semaphore = threading.Semaphore(15)


def make_flash_client():
    """Create Gemini Flash client."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=3)


def call_single(client, model, call_id, transcript, system_prompt, step_name, schema=None):
    """Single API call with retry and rate limiting."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0.0}
    if schema:
        kwargs["response_format"] = schema
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)
            track_usage(resp, step_name)
            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step_name} {call_id} attempt {attempt+1}: {e}")
            else:
                logger.error(f"  {step_name} {call_id} failed: {e}")
                return {"call_id": call_id, "error": str(e)}
        finally:
            _semaphore.release()


def call_batch(client, model, calls_batch, system_prompt, step_name, schema=None):
    """Batched API call: multiple transcripts in one request."""
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])

    batched_prompt = system_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently.\n\n'
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", "answer": ...}, ...]}\n'
        'Return JSON ONLY.'
    )

    messages = [
        {"role": "system", "content": batched_prompt},
        {"role": "user", "content": payload},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0.0}
    if schema:
        # Build batched schema
        single_item = schema["json_schema"]["schema"].copy()
        item_props = {**single_item["properties"], "call_id": {"type": "string"}}
        item_required = list(single_item["required"]) + ["call_id"]
        kwargs["response_format"] = {
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
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)
            track_usage(resp, step_name)
            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(item.get("call_id", "")): item for item in items}
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step_name} batch attempt {attempt+1}: {e}")
            else:
                logger.error(f"  {step_name} batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def run_field_batched(client, model, calls, system_prompt, batch_size, step_name, schema=None):
    """Run a field across all calls with batching and thread pool."""
    results = {}
    if batch_size <= 1:
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(call_single, client, model, c["id"], c["transcript"],
                            system_prompt, step_name, schema): c["id"]
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
        logger.info(f"  {step_name}: {len(calls)} calls in {len(batches)} batches of {batch_size}")
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(call_batch, client, model, batch, system_prompt,
                            step_name, schema): i
                for i, batch in enumerate(batches)
            }
            for f in as_completed(futures):
                try:
                    results.update(f.result())
                except Exception as e:
                    logger.error(f"  {step_name} batch error: {e}")
    return results


# ============================================================
# ARCHITECTURE A1b: BASELINE (strict schema)
# ============================================================

def run_a1b(client, model, calls, fields, batch_size):
    """A1b baseline: classify with strict JSON schema per field."""
    logger.info(f"=== A1b (baseline strict schema) ===")
    field_results = {}

    # Independent fields in parallel
    independent = [f for f in fields if f != "reason_not_booked"]
    with ThreadPoolExecutor(max_workers=len(independent)) as pool:
        futures = {}
        for field in independent:
            futures[pool.submit(
                run_field_batched, client, model, calls,
                FIELD_PROMPTS[field], batch_size, f"a1b_{field}",
                FIELD_SCHEMAS[field],
            )] = field
        for f in as_completed(futures):
            field = futures[f]
            field_results[field] = f.result()
            logger.info(f"  A1b {field}: {len(field_results[field])} results")

    # Dependent: reason_not_booked
    if "reason_not_booked" in fields:
        reason_results = {}
        calls_needing_llm = []
        appt_results = field_results.get("appointment_booked", {})

        for c in calls:
            cid = c["id"]
            appt_answer = appt_results.get(cid, {}).get("answer", "")
            if appt_answer == "No":
                calls_needing_llm.append(c)
            else:
                reason_results[cid] = {"call_id": cid, "answer": None,
                                       "reasoning": f"appt={appt_answer}, skipping"}

        if calls_needing_llm:
            prompt = V9_REASON_NOT_BOOKED_PROMPT.replace("{appointment_booked}", "No")
            llm_results = run_field_batched(
                client, model, calls_needing_llm, prompt, batch_size,
                "a1b_reason", V9_REASON_NOT_BOOKED_SCHEMA,
            )
            reason_results.update(llm_results)
        field_results["reason_not_booked"] = reason_results

    return field_results


# ============================================================
# ARCHITECTURE A1b: 2-STEP (reasoning + classification, all fields)
# ============================================================

def run_a1b_twostep(client, model, calls, fields, reasoning_bs, classify_bs):
    """Real A1b: Step 1 (reasoning from transcript) → Step 2 (classify from reasoning).

    Both steps process ALL fields in a single call (not per-field).
    """
    logger.info(f"=== A1b 2-step (reasoning → classification, strict schema) ===")

    # Step 1: Reasoning — Flash reads transcript → reasoning summary
    logger.info(f"  Step 1: reasoning ({len(calls)} calls, bs={reasoning_bs})")
    reasoning_results = run_field_batched(
        client, model, calls, REASONING_SYSTEM_PROMPT, reasoning_bs,
        "a1b_reasoning", None,  # json_object, no strict schema for reasoning
    )

    # Step 2: Classification — Flash reads reasoning → strict JSON
    # Build calls with reasoning as the "transcript"
    classify_calls = []
    for c in calls:
        cid = c["id"]
        reasoning = reasoning_results.get(cid, {})
        if reasoning.get("error"):
            continue
        # Extract reasoning text — could be in "reasoning" key or "calls" array
        r_text = reasoning.get("reasoning", "")
        if not r_text and "calls" in reasoning:
            # Script 03 format: {"calls": [{"call_id": ..., "reasoning": ...}]}
            for rc in reasoning.get("calls", []):
                if str(rc.get("call_id", "")) == cid:
                    r_text = rc.get("reasoning", "")
                    break
        if not r_text:
            r_text = json.dumps(reasoning)
        classify_calls.append({"id": cid, "transcript": r_text})

    logger.info(f"  Step 2: classification ({len(classify_calls)} calls, bs={classify_bs})")
    classify_results = run_field_batched(
        client, model, classify_calls, CLASSIFICATION_SYSTEM_PROMPT, classify_bs,
        "a1b_classify", CLASSIFICATION_RESPONSE_SCHEMA,
    )

    # Parse results — Script 03 schema returns {"calls": [{"call_id": ..., fields...}]}
    field_results = {f: {} for f in fields}
    for cid, result in classify_results.items():
        if result.get("error"):
            for f in fields:
                field_results[f][cid] = {"call_id": cid, "error": result["error"]}
            continue

        # Results might be nested in "calls" array or flat
        items = result.get("calls", [result])
        for item in items:
            item_cid = str(item.get("call_id", cid))
            for f in fields:
                val = item.get(f)
                field_results[f][item_cid] = {
                    "call_id": item_cid,
                    "answer": val,
                    "reasoning": reasoning_results.get(item_cid, {}).get("reasoning", ""),
                }

    return field_results


def run_a1b_twostep_freetext(client, model, calls, fields, reasoning_bs, classify_bs, format_bs):
    """A1b + E: 2-step reasoning → free-text classification → shared formatter.

    Step 1: reasoning from transcript (same as A1b)
    Step 2: classify from reasoning WITHOUT strict schema (free text)
    Step 3: shared formatter maps all fields to enums
    """
    logger.info(f"=== A1b+E (reasoning → free-text classify → shared formatter) ===")

    # Step 1: Reasoning
    logger.info(f"  Step 1: reasoning ({len(calls)} calls, bs={reasoning_bs})")
    reasoning_results = run_field_batched(
        client, model, calls, REASONING_SYSTEM_PROMPT, reasoning_bs,
        "a1b_e_reasoning", None,
    )

    # Step 2: Classification WITHOUT strict schema
    classify_calls = []
    for c in calls:
        cid = c["id"]
        reasoning = reasoning_results.get(cid, {})
        if reasoning.get("error"):
            continue
        r_text = reasoning.get("reasoning", "")
        if not r_text and "calls" in reasoning:
            for rc in reasoning.get("calls", []):
                if str(rc.get("call_id", "")) == cid:
                    r_text = rc.get("reasoning", "")
                    break
        if not r_text:
            r_text = json.dumps(reasoning)
        classify_calls.append({"id": cid, "transcript": r_text})

    logger.info(f"  Step 2: free-text classification ({len(classify_calls)} calls, bs={classify_bs})")
    classify_results = run_field_batched(
        client, model, classify_calls, CLASSIFICATION_SYSTEM_PROMPT, classify_bs,
        "a1b_e_classify", None,  # NO strict schema
    )

    # Parse free-text results
    free_text_per_call = {}
    for cid, result in classify_results.items():
        if result.get("error"):
            free_text_per_call[cid] = {f: None for f in fields}
            continue
        items = result.get("calls", [result])
        for item in items:
            item_cid = str(item.get("call_id", cid))
            free_text_per_call[item_cid] = {f: item.get(f) for f in fields}

    # Step 3: Shared formatter
    logger.info(f"  Step 3: shared formatter ({len(free_text_per_call)} calls)")
    enums_for_prompt = {}
    for field in fields:
        enums_for_prompt[field] = [e for e in FIELD_ENUMS[field] if e is not None]
        if None in FIELD_ENUMS[field]:
            enums_for_prompt[field].append("null")

    field_results = {f: {} for f in fields}
    items_to_format = []
    for cid, classifications in free_text_per_call.items():
        items_to_format.append({
            "id": cid,
            "classifications": {f: str(v) if v is not None else "null" for f, v in classifications.items()},
        })

    batches = [items_to_format[i:i + format_bs] for i in range(0, len(items_to_format), format_bs)]
    logger.info(f"  Step 3: {len(items_to_format)} items in {len(batches)} batches of {format_bs}")

    for batch in batches:
        payload = json.dumps([
            {"call_id": it["id"], "classifications": it["classifications"]}
            for it in batch
        ])
        prompt = f"""You are a formatting assistant. For each call, map the free-text classifications to the EXACT enum values.

## Valid enum values per field:
{json.dumps(enums_for_prompt, indent=2)}

Rules:
- For EACH call and EACH field, pick the best match from that field's enum list
- Use EXACT strings from the lists
- If a value is "null" or "None", keep it as null
- Do NOT re-analyze — just format
- Process each call and field INDEPENDENTLY

Return JSON: {{"results": [{{"call_id": "...", "appointment_booked": "...", "client_type": "...", "treatment_type": "...", "reason_not_booked": "... or null"}}]}}
Return JSON ONLY."""

        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            track_usage(resp, "a1b_e_format")
            result = json.loads(resp.choices[0].message.content)
            for r in result.get("results", []):
                cid = str(r.get("call_id", ""))
                for field in fields:
                    val = r.get(field)
                    if val == "null" or val is None:
                        val = None
                    field_results[field][cid] = {
                        "call_id": cid,
                        "answer": val,
                        "reasoning": reasoning_results.get(cid, {}).get("reasoning", ""),
                    }
        except Exception as e:
            logger.error(f"  A1b+E format batch failed: {e}")
            for it in batch:
                for field in fields:
                    field_results[field][it["id"]] = {"call_id": it["id"], "error": str(e)}
        finally:
            _semaphore.release()

    return field_results


# ============================================================
# ARCHITECTURE D: 3-STEP (free-text + per-field formatter)
# ============================================================

def format_per_field(client, model, free_text_results, field_name, batch_size):
    """Format free-text answers to strict enums, one field at a time."""
    enum_list = FIELD_ENUMS[field_name]
    enum_str = "\n".join(f"- {e}" if e is not None else "- null" for e in enum_list)
    formatted = {}

    items = []
    for cid, result in free_text_results.items():
        if result.get("error"):
            formatted[cid] = result
            continue
        answer = result.get("answer", "")
        if answer is None or (isinstance(answer, str) and answer.lower() in ("null", "none", "")):
            # Already null — no need to format
            formatted[cid] = {"call_id": cid, "answer": None, "reasoning": result.get("reasoning", "")}
            continue
        items.append({"id": cid, "free_text": str(answer), "reasoning": result.get("reasoning", "")})

    if not items:
        return formatted

    logger.info(f"  D format {field_name}: formatting {len(items)} answers")

    # Batch the formatting calls
    if batch_size <= 1 or len(items) <= 1:
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {}
            for item in items:
                prompt = PER_FIELD_FORMATTER_PROMPT.format(
                    free_text=item["free_text"],
                    enum_list=enum_str,
                )
                futures[pool.submit(
                    call_single, client, model, item["id"], prompt,
                    "You are a formatting assistant. Return JSON ONLY.",
                    f"d_format_{field_name}",
                )] = item
            for f in as_completed(futures):
                item = futures[f]
                try:
                    result = f.result()
                    formatted[item["id"]] = {
                        "call_id": item["id"],
                        "answer": result.get("answer"),
                        "reasoning": item["reasoning"],
                        "free_text_original": item["free_text"],
                    }
                except Exception as e:
                    formatted[item["id"]] = {"call_id": item["id"], "error": str(e)}
    else:
        # Batch formatting: send multiple items in one call
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        for batch in batches:
            payload = json.dumps([
                {"call_id": it["id"], "free_text": it["free_text"]}
                for it in batch
            ])
            prompt = f"""You are a formatting assistant. Map each free-text classification to the EXACT matching enum value.

## Valid enum values for {field_name}:
{enum_str}

Rules:
- Pick the SINGLE best match from the enum list for each item
- Use the EXACT string from the list (copy character-for-character)
- If the free-text says null, none, or not applicable, answer null
- Do NOT re-analyze — just map text to closest enum

You will receive a JSON array. Return JSON: {{"results": [{{"call_id": "...", "answer": "<exact enum or null>"}}]}}
Return JSON ONLY."""

            _semaphore.acquire()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": payload},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                track_usage(resp, f"d_format_{field_name}")
                result = json.loads(resp.choices[0].message.content)
                for r in result.get("results", []):
                    cid = str(r.get("call_id", ""))
                    orig_item = next((it for it in batch if it["id"] == cid), None)
                    formatted[cid] = {
                        "call_id": cid,
                        "answer": r.get("answer"),
                        "reasoning": orig_item["reasoning"] if orig_item else "",
                        "free_text_original": orig_item["free_text"] if orig_item else "",
                    }
            except Exception as e:
                logger.error(f"  D format batch failed: {e}")
                for it in batch:
                    formatted[it["id"]] = {"call_id": it["id"], "error": str(e)}
            finally:
                _semaphore.release()

    return formatted


def run_d(client, model, calls, fields, classify_bs, format_bs):
    """Architecture D: free-text classify → per-field format to enum."""
    logger.info(f"=== D (3-step: free-text + per-field formatter) ===")
    field_results = {}

    # Step 1: Classify WITHOUT strict schema (free-text answers)
    independent = [f for f in fields if f != "reason_not_booked"]
    free_text_results = {}

    with ThreadPoolExecutor(max_workers=len(independent)) as pool:
        futures = {}
        for field in independent:
            # Use same prompt but NO schema — just json_object
            futures[pool.submit(
                run_field_batched, client, model, calls,
                FIELD_PROMPTS[field], classify_bs, f"d_freetext_{field}",
                None,  # <-- NO strict schema
            )] = field
        for f in as_completed(futures):
            field = futures[f]
            free_text_results[field] = f.result()
            logger.info(f"  D free-text {field}: {len(free_text_results[field])} results")

    # reason_not_booked (depends on appointment_booked free-text)
    if "reason_not_booked" in fields:
        reason_free = {}
        calls_needing_llm = []
        appt_free = free_text_results.get("appointment_booked", {})

        for c in calls:
            cid = c["id"]
            appt_answer = _normalize_appt(appt_free.get(cid, {}).get("answer", ""))
            if appt_answer == "No":
                calls_needing_llm.append(c)
            else:
                reason_free[cid] = {"call_id": cid, "answer": None,
                                    "reasoning": f"appt={appt_answer}, skipping"}

        if calls_needing_llm:
            prompt = V9_REASON_NOT_BOOKED_PROMPT.replace("{appointment_booked}", "No")
            llm_results = run_field_batched(
                client, model, calls_needing_llm, prompt, classify_bs,
                "d_freetext_reason", None,
            )
            reason_free.update(llm_results)
        free_text_results["reason_not_booked"] = reason_free

    # Step 2: Format each field's free-text to strict enum
    for field in fields:
        if field in free_text_results:
            field_results[field] = format_per_field(
                client, model, free_text_results[field], field, format_bs,
            )

    return field_results


# ============================================================
# ARCHITECTURE E: SHARED FORMATTER (one call formats all fields)
# ============================================================

def run_e(client, model, calls, fields, classify_bs, format_bs):
    """Architecture E: free-text classify → ONE shared format call for all fields."""
    logger.info(f"=== E (shared formatter: one call formats all fields) ===")

    # Step 1: Same as D — classify without strict schema
    independent = [f for f in fields if f != "reason_not_booked"]
    free_text_results = {}

    with ThreadPoolExecutor(max_workers=len(independent)) as pool:
        futures = {}
        for field in independent:
            futures[pool.submit(
                run_field_batched, client, model, calls,
                FIELD_PROMPTS[field], classify_bs, f"e_freetext_{field}",
                None,
            )] = field
        for f in as_completed(futures):
            field = futures[f]
            free_text_results[field] = f.result()
            logger.info(f"  E free-text {field}: {len(free_text_results[field])} results")

    if "reason_not_booked" in fields:
        reason_free = {}
        calls_needing_llm = []
        appt_free = free_text_results.get("appointment_booked", {})

        for c in calls:
            cid = c["id"]
            appt_answer = _normalize_appt(appt_free.get(cid, {}).get("answer", ""))
            if appt_answer == "No":
                calls_needing_llm.append(c)
            else:
                reason_free[cid] = {"call_id": cid, "answer": None,
                                    "reasoning": f"appt={appt_answer}, skipping"}

        if calls_needing_llm:
            prompt = V9_REASON_NOT_BOOKED_PROMPT.replace("{appointment_booked}", "No")
            llm_results = run_field_batched(
                client, model, calls_needing_llm, prompt, classify_bs,
                "e_freetext_reason", None,
            )
            reason_free.update(llm_results)
        free_text_results["reason_not_booked"] = reason_free

    # Step 2: ONE shared formatter call per transcript (or batched)
    logger.info(f"  E: formatting all fields with shared formatter")

    # Build enums reference (exclude null from display, handle separately)
    enums_for_prompt = {}
    for field in fields:
        enums_for_prompt[field] = [e for e in FIELD_ENUMS[field] if e is not None]
        if None in FIELD_ENUMS[field]:
            enums_for_prompt[field].append("null")

    field_results = {f: {} for f in fields}

    # Batch the shared formatter calls
    items_to_format = []
    for c in calls:
        cid = c["id"]
        classifications = {}
        for field in fields:
            fr = free_text_results.get(field, {}).get(cid, {})
            answer = fr.get("answer")
            classifications[field] = str(answer) if answer is not None else "null"
        items_to_format.append({"id": cid, "classifications": classifications})

    # Process in batches
    batches = [items_to_format[i:i + format_bs] for i in range(0, len(items_to_format), format_bs)]
    logger.info(f"  E format: {len(items_to_format)} items in {len(batches)} batches of {format_bs}")

    for batch in batches:
        if len(batch) == 1:
            # Single item
            item = batch[0]
            prompt = SHARED_FORMATTER_PROMPT.format(
                classifications_json=json.dumps(item["classifications"], indent=2),
                enums_json=json.dumps(enums_for_prompt, indent=2),
            )
            result = call_single(
                client, model, item["id"], prompt,
                "You are a formatting assistant. Return JSON ONLY.",
                "e_format_shared",
            )
            if not result.get("error"):
                mapped = result.get("results", result)
                for field in fields:
                    val = mapped.get(field)
                    if val == "null" or val is None:
                        val = None
                    orig_fr = free_text_results.get(field, {}).get(item["id"], {})
                    field_results[field][item["id"]] = {
                        "call_id": item["id"],
                        "answer": val,
                        "reasoning": orig_fr.get("reasoning", ""),
                        "free_text_original": str(orig_fr.get("answer", "")),
                    }
        else:
            # Multi-item batch
            payload = json.dumps([
                {"call_id": it["id"], "classifications": it["classifications"]}
                for it in batch
            ])
            prompt = f"""You are a formatting assistant. For each call, map the free-text classifications to the EXACT enum values.

## Valid enum values per field:
{json.dumps(enums_for_prompt, indent=2)}

Rules:
- For EACH call and EACH field, pick the best match from that field's enum list
- Use EXACT strings from the lists
- If a value is "null" or "None", keep it as null
- Do NOT re-analyze — just format
- Process each call and field INDEPENDENTLY

Return JSON: {{"results": [{{"call_id": "...", "appointment_booked": "...", "client_type": "...", "treatment_type": "...", "reason_not_booked": "... or null"}}]}}
Return JSON ONLY."""

            _semaphore.acquire()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": payload},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                track_usage(resp, "e_format_shared")
                result = json.loads(resp.choices[0].message.content)
                for r in result.get("results", []):
                    cid = str(r.get("call_id", ""))
                    for field in fields:
                        val = r.get(field)
                        if val == "null" or val is None:
                            val = None
                        orig_fr = free_text_results.get(field, {}).get(cid, {})
                        field_results[field][cid] = {
                            "call_id": cid,
                            "answer": val,
                            "reasoning": orig_fr.get("reasoning", ""),
                            "free_text_original": str(orig_fr.get("answer", "")),
                        }
            except Exception as e:
                logger.error(f"  E format batch failed: {e}")
                for it in batch:
                    for field in fields:
                        field_results[field][it["id"]] = {"call_id": it["id"], "error": str(e)}
            finally:
                _semaphore.release()

    return field_results


# ============================================================
# SCORING
# ============================================================

def score_field(predictions, gold_data, field):
    """Score predictions vs gold labels. Returns (correct, total, errors)."""
    correct = total = 0
    errors = []
    for cid, pred in predictions.items():
        if cid not in gold_data:
            continue
        gold_val = gold_data[cid]["labels"].get(field, "").strip()
        pred_val = pred.get("answer")
        if pred_val is None:
            pred_val = ""
        elif isinstance(pred_val, str):
            pred_val = pred_val.strip()

        # Normalize nulls
        if not gold_val or gold_val.lower() in ("null", "none"):
            gold_val = ""
        if not pred_val or (isinstance(pred_val, str) and pred_val.lower() in ("null", "none")):
            pred_val = ""

        # Skip when both are empty (reason_not_booked for booked calls)
        if field == "reason_not_booked" and not gold_val and not pred_val:
            continue

        total += 1
        if gold_val == pred_val:
            correct += 1
        else:
            errors.append({
                "call_id": cid,
                "pred": pred_val or "(null)",
                "gold": gold_val or "(null)",
                "free_text": pred.get("free_text_original", ""),
                "reasoning": pred.get("reasoning", "")[:150],
            })

    return correct, total, errors


def _normalize_appt(answer):
    """Normalize free-text appointment_booked to Yes/No/Inconclusive."""
    if not answer:
        return ""
    answer = str(answer).strip()
    lower = answer.lower()
    if lower in ("yes", "true", "booked", "confirmed"):
        return "Yes"
    elif lower in ("no", "false", "not booked", "not confirmed"):
        return "No"
    elif lower in ("inconclusive", "unclear", "unknown"):
        return "Inconclusive"
    # Already in correct format
    if answer in ("Yes", "No", "Inconclusive"):
        return answer
    return answer


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test free-text + formatter architectures")
    parser.add_argument("--max-calls", type=int, default=100, help="Max calls to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--classify-bs", type=int, default=15, help="Classification batch size")
    parser.add_argument("--format-bs", type=int, default=15, help="Formatter batch size")
    parser.add_argument("--fields", nargs="+",
                        default=["appointment_booked", "client_type", "treatment_type", "reason_not_booked"],
                        help="Fields to test")
    parser.add_argument("--skip-a0b", action="store_true", help="Skip A0b tests")
    parser.add_argument("--reasoning-bs", type=int, default=4, help="Reasoning batch size (A1b)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load data
    data = load_data(project_dir)
    all_ids = list(data.keys())
    random.seed(args.seed)
    sample_ids = random.sample(all_ids, min(args.max_calls, len(all_ids)))
    calls = [data[cid] for cid in sample_ids]
    logger.info(f"Testing {len(calls)} calls, fields: {args.fields}")

    client = make_flash_client()
    model = "gemini-2.5-flash"

    all_results = {}

    # ── A0b: per-field Flash + strict schema (single-step baseline) ──
    if not args.skip_a0b:
        token_usage["input"] = 0; token_usage["output"] = 0
        a0b_results = run_a1b(client, model, calls, args.fields, args.classify_bs)
        all_results["A0b"] = a0b_results

    # ── A1b: 2-step reasoning → classification (strict schema) ──
    token_usage["input"] = 0; token_usage["output"] = 0
    a1b_results = run_a1b_twostep(client, model, calls, args.fields, args.reasoning_bs, args.classify_bs)
    all_results["A1b"] = a1b_results

    # ── A1b+E: 2-step reasoning → free-text classify → shared formatter ──
    token_usage["input"] = 0; token_usage["output"] = 0
    a1b_e_results = run_a1b_twostep_freetext(client, model, calls, args.fields, args.reasoning_bs, args.classify_bs, args.format_bs)
    all_results["A1b+E"] = a1b_e_results

    # ── Scoring ──
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 70)

    header = f"{'Field':<25}"
    for arch in all_results:
        header += f" | {arch:>10}"
    logger.info(header)
    logger.info("-" * len(header))

    summary = {}
    for arch_name, arch_results in all_results.items():
        summary[arch_name] = {}
        for field in args.fields:
            if field in arch_results:
                correct, total, errors = score_field(arch_results[field], data, field)
                acc = correct / total * 100 if total > 0 else 0
                summary[arch_name][field] = {"accuracy": acc, "correct": correct, "total": total, "n_errors": len(errors)}

    for field in args.fields:
        row = f"{field:<25}"
        for arch_name in all_results:
            s = summary.get(arch_name, {}).get(field, {})
            acc = s.get("accuracy", 0)
            row += f" | {acc:>9.1f}%"
        logger.info(row)

    # Averages
    row = f"{'AVERAGE':<25}"
    for arch_name in all_results:
        accs = [s.get("accuracy", 0) for s in summary.get(arch_name, {}).values()]
        avg = sum(accs) / len(accs) if accs else 0
        row += f" | {avg:>9.1f}%"
    logger.info(row)

    # ── Error Comparison (treatment_type & reason_not_booked) ──
    for field in ["treatment_type", "reason_not_booked"]:
        if field not in args.fields:
            continue
        logger.info(f"\n--- {field} Error Comparison ---")
        for arch_name, arch_results in all_results.items():
            if field in arch_results:
                _, _, errors = score_field(arch_results[field], data, field)
                logger.info(f"\n  [{arch_name}] Top 5 errors:")
                for e in errors[:5]:
                    ft = f" (free_text: {e['free_text'][:60]})" if e.get("free_text") else ""
                    logger.info(f"    {e['call_id']}: pred={e['pred']!r} gold={e['gold']!r}{ft}")

    # ── Save results ──
    output_path = args.output or f"results_freetext_formatter_n{len(calls)}.json"
    output = {
        "n": len(calls),
        "seed": args.seed,
        "classify_bs": args.classify_bs,
        "format_bs": args.format_bs,
        "summary": summary,
    }
    # Include per-call results for deeper analysis
    for arch_name, arch_results in all_results.items():
        output[arch_name] = {}
        for field, results in arch_results.items():
            output[arch_name][field] = {
                cid: {
                    "answer": r.get("answer"),
                    "reasoning": r.get("reasoning", "")[:200],
                    "free_text_original": r.get("free_text_original", ""),
                }
                for cid, r in results.items()
            }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
