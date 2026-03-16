#!/usr/bin/env python3
"""
Test A0b architecture: Flash single-call, all 4 fields in one prompt.

Compares:
  A0b:   Single Flash call → all fields → strict schema
  A0b+E: Single Flash call → all fields → free text → shared formatter

Usage:
  python Scripts/test_a0b.py --max-calls 100
  python Scripts/test_a0b.py --max-calls 200 --batch-size 15
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
logger = logging.getLogger("test_a0b")

# ============================================================
# LOAD PROMPTS & SCHEMAS FROM SCRIPT 03
# ============================================================

sys.path.insert(0, script_dir)
from validate_prompt_engineering import (
    V9_APPOINTMENT_BOOKED_PROMPT,
    V9_CLIENT_TYPE_PROMPT,
    V9_TREATMENT_TYPE_PROMPT,
    V9_REASON_NOT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA,
    V9_TREATMENT_TYPE_SCHEMA,
    V9_REASON_NOT_BOOKED_SCHEMA,
    load_data,
)

# Load Script 03 for CLASSIFICATION_RESPONSE_SCHEMA (used as basis for combined schema)
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("script03", os.path.join(script_dir, "03_CallRail_Transcripts_Analyze_Buckets.py"))
_mod = module_from_spec(_spec)
_orig_argv = sys.argv
sys.argv = [""]
try:
    _spec.loader.exec_module(_mod)
except SystemExit:
    pass
finally:
    sys.argv = _orig_argv

# ============================================================
# A0b COMBINED PROMPT: all 4 V9 field prompts in one call
# ============================================================

# Strip per-field output instructions (we'll add a combined one at the end)
def _strip_output(prompt):
    """Remove the per-field output format / 'Return JSON ONLY' tail."""
    # Remove everything from "## Output Format" or the final JSON instruction onward
    for marker in ["## Output Format", "Return JSON with your reasoning", "Return JSON ONLY."]:
        idx = prompt.rfind(marker)
        if idx > 0:
            prompt = prompt[:idx]
    return prompt.strip()

A0B_COMBINED_PROMPT = (
    "You are a veterinary call transcript analyst. Classify this call across ALL 4 fields below. "
    "Each field section includes its own rules and examples — follow them precisely.\n\n"
    "# ── FIELD 1: appointment_booked ──\n\n"
    + _strip_output(V9_APPOINTMENT_BOOKED_PROMPT) + "\n\n"
    "# ── FIELD 2: client_type ──\n\n"
    + _strip_output(V9_CLIENT_TYPE_PROMPT) + "\n\n"
    "# ── FIELD 3: treatment_type ──\n\n"
    + _strip_output(V9_TREATMENT_TYPE_PROMPT) + "\n\n"
    "# ── FIELD 4: reason_not_booked ──\n\n"
    + _strip_output(V9_REASON_NOT_BOOKED_PROMPT.replace(
        "{appointment_booked}",
        "your appointment_booked answer from Field 1"
    )) + "\n\n"
    "# ── Output Format ──\n\n"
    "Return JSON with your reasoning and ALL 4 field answers:\n"
    '{"reasoning": "...", "appointment_booked": "Yes"|"No"|"Inconclusive", '
    '"client_type": "New"|"Existing"|"Inconclusive", '
    '"treatment_type": "<exact category from list above>", '
    '"reason_not_booked": "<exact category or null>"}\n\n'
    "Return JSON ONLY."
)

# Combined strict schema: all 4 fields in one response
TREATMENT_TYPE_ENUMS = V9_TREATMENT_TYPE_SCHEMA["json_schema"]["schema"]["properties"]["answer"]["enum"]
REASON_NOT_BOOKED_ENUMS = V9_REASON_NOT_BOOKED_SCHEMA["json_schema"]["schema"]["properties"]["answer"]["enum"]

A0B_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "a0b_all_fields",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "appointment_booked": {"type": "string", "enum": ["Yes", "No", "Inconclusive"]},
                "client_type": {"type": "string", "enum": ["New", "Existing", "Inconclusive"]},
                "treatment_type": {"type": "string", "enum": TREATMENT_TYPE_ENUMS},
                "reason_not_booked": {"type": ["string", "null"], "enum": REASON_NOT_BOOKED_ENUMS},
            },
            "required": ["reasoning", "appointment_booked", "client_type", "treatment_type", "reason_not_booked"],
            "additionalProperties": False,
        },
    },
}

# Batched version
A0B_SCHEMA_BATCH = {
    "type": "json_schema",
    "json_schema": {
        "name": "a0b_all_fields_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "call_id": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "appointment_booked": {"type": "string", "enum": ["Yes", "No", "Inconclusive"]},
                            "client_type": {"type": "string", "enum": ["New", "Existing", "Inconclusive"]},
                            "treatment_type": {"type": "string", "enum": TREATMENT_TYPE_ENUMS},
                            "reason_not_booked": {"type": ["string", "null"], "enum": REASON_NOT_BOOKED_ENUMS},
                        },
                        "required": ["call_id", "reasoning", "appointment_booked", "client_type", "treatment_type", "reason_not_booked"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

FIELD_ENUMS = {
    "appointment_booked": ["Yes", "No", "Inconclusive"],
    "client_type": ["New", "Existing", "Inconclusive"],
    "treatment_type": TREATMENT_TYPE_ENUMS,
    "reason_not_booked": REASON_NOT_BOOKED_ENUMS,
}

FIELDS = ["appointment_booked", "client_type", "treatment_type", "reason_not_booked"]

# ============================================================
# TOKEN TRACKING
# ============================================================

token_usage = {"input": 0, "output": 0}
_token_lock = threading.Lock()

def track_usage(resp):
    if resp.usage:
        with _token_lock:
            token_usage["input"] += resp.usage.prompt_tokens or 0
            token_usage["output"] += resp.usage.completion_tokens or 0

# ============================================================
# API CALLS
# ============================================================

_semaphore = threading.Semaphore(15)


def make_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=3)


def call_batch(client, model, calls_batch, system_prompt, schema=None, step=""):
    """Send a batch of transcripts in one API call. Returns {call_id: result_dict}."""
    # V9 batching format: JSON array of transcripts
    batched_prompt = system_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently — do not let one transcript '
        'influence your classification of another.\n\n'
        'Return JSON: {"results": [{"call_id": "...", ...}, ...]}\n'
        'Return JSON ONLY.'
    )
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    messages = [
        {"role": "system", "content": batched_prompt},
        {"role": "user", "content": payload},
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
            track_usage(resp)
            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(it.get("call_id", "")): it for it in items}
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.5)
                logger.warning(f"  {step} batch attempt {attempt+1}: {e}")
            else:
                logger.error(f"  {step} batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def run_all_batches(client, model, calls, system_prompt, batch_size, schema=None, step=""):
    """Run batches in parallel via thread pool. Returns {call_id: result_dict}."""
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
    logger.info(f"  {step}: {len(calls)} calls in {len(batches)} batches of {batch_size}")

    results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_batch, client, model, batch, system_prompt, schema, step): i
            for i, batch in enumerate(batches)
        }
        for f in as_completed(futures):
            try:
                results.update(f.result())
            except Exception as e:
                logger.error(f"  {step} thread error: {e}")

    return results


# ============================================================
# A0b: SINGLE CALL, STRICT SCHEMA (V9 prompts combined)
# ============================================================

def run_a0b(client, model, calls, batch_size):
    """A0b: single Flash call, all V9 field prompts combined, strict schema."""
    logger.info("=== A0b (single-call, V9 combined prompts, strict schema) ===")
    results = run_all_batches(
        client, model, calls, A0B_COMBINED_PROMPT, batch_size,
        schema=A0B_SCHEMA_BATCH, step="a0b",
    )

    # Convert to per-field format
    field_results = {f: {} for f in FIELDS}
    for cid, item in results.items():
        for f in FIELDS:
            field_results[f][cid] = {"call_id": cid, "answer": item.get(f)}
    return field_results


# ============================================================
# A0b+E: SINGLE CALL FREE TEXT → SHARED FORMATTER
# ============================================================

def run_a0b_freetext(client, model, calls, batch_size, format_bs):
    """A0b+E: single Flash call (no schema) → shared formatter."""
    logger.info("=== A0b+E (single-call free text → shared formatter) ===")

    # Step 1: Classify without strict schema (same V9 combined prompt, no schema)
    free_results = run_all_batches(
        client, model, calls, A0B_COMBINED_PROMPT, batch_size,
        schema=None, step="a0b_e_freetext",
    )

    # Step 2: Shared formatter
    logger.info(f"  A0b+E formatter: {len(free_results)} items")
    enums_for_prompt = {}
    for field in FIELDS:
        enums_for_prompt[field] = [e for e in FIELD_ENUMS[field] if e is not None]
        if None in FIELD_ENUMS[field]:
            enums_for_prompt[field].append("null")

    items_to_format = []
    for cid, item in free_results.items():
        if item.get("error"):
            continue
        classifications = {f: str(item.get(f)) if item.get(f) is not None else "null" for f in FIELDS}
        items_to_format.append({"id": cid, "classifications": classifications})

    field_results = {f: {} for f in FIELDS}
    # Mark errors
    for cid, item in free_results.items():
        if item.get("error"):
            for f in FIELDS:
                field_results[f][cid] = {"call_id": cid, "error": item["error"]}

    batches = [items_to_format[i:i + format_bs] for i in range(0, len(items_to_format), format_bs)]
    logger.info(f"  A0b+E format: {len(items_to_format)} items in {len(batches)} batches of {format_bs}")

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
            track_usage(resp)
            result = json.loads(resp.choices[0].message.content)
            for r in result.get("results", []):
                cid = str(r.get("call_id", ""))
                for field in FIELDS:
                    val = r.get(field)
                    if val == "null" or val is None:
                        val = None
                    field_results[field][cid] = {"call_id": cid, "answer": val}
        except Exception as e:
            logger.error(f"  A0b+E format batch failed: {e}")
            for it in batch:
                for field in FIELDS:
                    field_results[field][it["id"]] = {"call_id": it["id"], "error": str(e)}
        finally:
            _semaphore.release()

    return field_results


# ============================================================
# SCORING
# ============================================================

def score_field(predictions, gold_data, field):
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

        if not gold_val or gold_val.lower() in ("null", "none"):
            gold_val = ""
        if not pred_val or (isinstance(pred_val, str) and pred_val.lower() in ("null", "none")):
            pred_val = ""

        if field == "reason_not_booked" and not gold_val and not pred_val:
            continue

        total += 1
        if gold_val == pred_val:
            correct += 1
        else:
            errors.append({"call_id": cid, "pred": pred_val or "(null)", "gold": gold_val or "(null)"})

    return correct, total, errors


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test A0b architecture variants")
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--format-bs", type=int, default=15)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = load_data(project_dir)
    all_ids = list(data.keys())
    random.seed(args.seed)
    sample_ids = random.sample(all_ids, min(args.max_calls, len(all_ids)))
    calls = [data[cid] for cid in sample_ids]
    logger.info(f"Testing {len(calls)} calls")

    client = make_client()
    model = "gemini-2.5-flash"

    all_results = {}

    # A0b baseline
    token_usage["input"] = 0; token_usage["output"] = 0
    all_results["A0b"] = run_a0b(client, model, calls, args.batch_size)

    # A0b+E (free text + formatter)
    token_usage["input"] = 0; token_usage["output"] = 0
    all_results["A0b+E"] = run_a0b_freetext(client, model, calls, args.batch_size, args.format_bs)

    # Scoring
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: A0b vs A0b+E")
    logger.info("=" * 60)

    header = f"{'Field':<25} | {'A0b':>10} | {'A0b+E':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    summary = {}
    for arch_name, arch_results in all_results.items():
        summary[arch_name] = {}
        for field in FIELDS:
            correct, total, errors = score_field(arch_results[field], data, field)
            acc = correct / total * 100 if total > 0 else 0
            summary[arch_name][field] = {"accuracy": acc, "correct": correct, "total": total}

    for field in FIELDS:
        a0b_acc = summary["A0b"][field]["accuracy"]
        a0be_acc = summary["A0b+E"][field]["accuracy"]
        delta = a0be_acc - a0b_acc
        logger.info(f"{field:<25} | {a0b_acc:>9.1f}% | {a0be_acc:>9.1f}% ({delta:+.1f})")

    a0b_avg = sum(s["accuracy"] for s in summary["A0b"].values()) / len(FIELDS)
    a0be_avg = sum(s["accuracy"] for s in summary["A0b+E"].values()) / len(FIELDS)
    logger.info(f"{'AVERAGE':<25} | {a0b_avg:>9.1f}% | {a0be_avg:>9.1f}% ({a0be_avg - a0b_avg:+.1f})")

    # Save
    output_path = args.output or f"results_a0b_test_n{len(calls)}.json"
    with open(output_path, "w") as f:
        json.dump({"n": len(calls), "seed": args.seed, "summary": summary}, f, indent=2)
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
