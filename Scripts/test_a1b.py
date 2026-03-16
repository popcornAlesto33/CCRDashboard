#!/usr/bin/env python3
"""
Test A1b architecture: Flash 2-call with context passing.

  Call 1: appointment_booked only (per-field V9 prompt)
  Call 2: client_type + treatment_type + reason_not_booked
          (receives appointment_booked result as context)

Compares:
  A1b:   2-call with strict schema
  A1b+E: 2-call free text → shared formatter

Usage:
  python Scripts/test_a1b.py --max-calls 100
  python Scripts/test_a1b.py --max-calls 200 --appt-bs 15 --combo-bs 15
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
logger = logging.getLogger("test_a1b")

# ============================================================
# LOAD PROMPTS
# ============================================================

sys.path.insert(0, script_dir)
from validate_prompt_engineering import (
    V9_APPOINTMENT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA,
    V9_CLIENT_TYPE_PROMPT,
    V9_TREATMENT_TYPE_PROMPT,
    V9_REASON_NOT_BOOKED_PROMPT,
    load_data,
)

# Load Script 03 for CLASSIFICATION_RESPONSE_SCHEMA
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

CLASSIFICATION_SCHEMA = _mod.CLASSIFICATION_RESPONSE_SCHEMA

# ============================================================
# A1b CALL 2 PROMPT: 3 fields with appointment context
# Built from the actual V9 prompts with all few-shot examples
# ============================================================

def _strip_output(prompt):
    """Remove the per-field output format / 'Return JSON ONLY' tail."""
    for marker in ["## Output Format", "Return JSON with your reasoning", "Return JSON ONLY."]:
        idx = prompt.rfind(marker)
        if idx > 0:
            prompt = prompt[:idx]
    return prompt.strip()

A1B_COMBO_PROMPT = (
    "You are a veterinary call transcript analyst. "
    "The appointment_booked decision has already been made: {appointment_booked}\n\n"
    "Classify the following 3 fields from this call transcript. "
    "Each field section includes its own rules and examples — follow them precisely.\n\n"
    "# ── FIELD 1: client_type ──\n\n"
    + _strip_output(V9_CLIENT_TYPE_PROMPT) + "\n\n"
    "# ── FIELD 2: treatment_type ──\n\n"
    + _strip_output(V9_TREATMENT_TYPE_PROMPT) + "\n\n"
    "# ── FIELD 3: reason_not_booked ──\n\n"
    + _strip_output(V9_REASON_NOT_BOOKED_PROMPT.replace(
        "{appointment_booked}",
        "{appointment_booked}"  # keep placeholder — filled at call time
    )) + "\n\n"
    "# ── Output Format ──\n\n"
    "Return JSON with your reasoning and ALL 3 field answers:\n"
    '{{"reasoning": "...", "client_type": "New"|"Existing"|"Inconclusive", '
    '"treatment_type": "<exact category from list above>", '
    '"reason_not_booked": "<exact category or null>"}}\n\n'
    "Return JSON ONLY."
)

# Strict schema for combo call (3 fields)
A1B_COMBO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "combo_3field_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "client_type": {
                    "type": "string",
                    "enum": ["New", "Existing", "Inconclusive"],
                },
                "treatment_type": {
                    "type": "string",
                    "enum": CLASSIFICATION_SCHEMA["json_schema"]["schema"]["properties"]["calls"]["items"]["properties"]["treatment_type"]["enum"],
                },
                "reason_not_booked": {
                    "type": ["string", "null"],
                    "enum": CLASSIFICATION_SCHEMA["json_schema"]["schema"]["properties"]["calls"]["items"]["properties"]["reason_not_booked"]["enum"],
                },
            },
            "required": ["reasoning", "client_type", "treatment_type", "reason_not_booked"],
            "additionalProperties": False,
        },
    },
}

# Batched version of combo schema
A1B_COMBO_SCHEMA_BATCH = {
    "type": "json_schema",
    "json_schema": {
        "name": "combo_3field_batch",
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
                            "client_type": A1B_COMBO_SCHEMA["json_schema"]["schema"]["properties"]["client_type"],
                            "treatment_type": A1B_COMBO_SCHEMA["json_schema"]["schema"]["properties"]["treatment_type"],
                            "reason_not_booked": A1B_COMBO_SCHEMA["json_schema"]["schema"]["properties"]["reason_not_booked"],
                        },
                        "required": ["call_id", "reasoning", "client_type", "treatment_type", "reason_not_booked"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# ENUM LISTS (for formatter)
# ============================================================

FIELD_ENUMS = {
    "appointment_booked": ["Yes", "No", "Inconclusive"],
    "client_type": ["New", "Existing", "Inconclusive"],
    "treatment_type": A1B_COMBO_SCHEMA["json_schema"]["schema"]["properties"]["treatment_type"]["enum"],
    "reason_not_booked": A1B_COMBO_SCHEMA["json_schema"]["schema"]["properties"]["reason_not_booked"]["enum"],
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


def call_batch_appt(client, model, calls_batch, schema=None):
    """Call 1: appointment_booked only, batched."""
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    prompt = V9_APPOINTMENT_BOOKED_PROMPT.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently.\n\n'
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", "answer": ...}, ...]}\n'
        'Return JSON ONLY.'
    )
    # Build batched schema
    batched_schema = None
    if schema:
        single_item = schema["json_schema"]["schema"].copy()
        item_props = {**single_item["properties"], "call_id": {"type": "string"}}
        item_required = list(single_item["required"]) + ["call_id"]
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
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0.0}
    kwargs["response_format"] = batched_schema or {"type": "json_object"}

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
                logger.warning(f"  appt batch attempt {attempt+1}: {e}")
            else:
                logger.error(f"  appt batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def call_batch_combo(client, model, calls_batch, appt_answer, schema=None):
    """Call 2: client_type + treatment_type + reason_not_booked, batched."""
    prompt = A1B_COMBO_PROMPT.replace("{appointment_booked}", appt_answer)
    prompt = prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently.\n\n'
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", '
        '"client_type": "...", "treatment_type": "...", "reason_not_booked": "... or null"}, ...]}\n'
        'Return JSON ONLY.'
    )
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0.0}
    kwargs["response_format"] = schema or {"type": "json_object"}

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
                logger.warning(f"  combo batch attempt {attempt+1}: {e}")
            else:
                logger.error(f"  combo batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


# ============================================================
# A1b: 2-CALL WITH STRICT SCHEMA
# ============================================================

def run_a1b(client, model, calls, appt_bs, combo_bs):
    """A1b: Call 1 (appointment) → Call 2 (3 fields with context)."""
    logger.info("=== A1b (2-call, strict schema) ===")

    # Call 1: appointment_booked
    logger.info(f"  Call 1: appointment_booked ({len(calls)} calls, bs={appt_bs})")
    appt_batches = [calls[i:i + appt_bs] for i in range(0, len(calls), appt_bs)]
    appt_results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_batch_appt, client, model, batch, V9_APPOINTMENT_BOOKED_SCHEMA): i
            for i, batch in enumerate(appt_batches)
        }
        for f in as_completed(futures):
            try:
                appt_results.update(f.result())
            except Exception as e:
                logger.error(f"  appt thread error: {e}")

    logger.info(f"  Call 1 done: {len(appt_results)} results")

    # Group calls by appointment_booked answer for Call 2
    calls_by_appt = {"Yes": [], "No": [], "Inconclusive": [], "error": []}
    for c in calls:
        cid = c["id"]
        appt = appt_results.get(cid, {})
        answer = appt.get("answer", "")
        if answer in calls_by_appt:
            calls_by_appt[answer].append(c)
        else:
            calls_by_appt["error"].append(c)

    logger.info(f"  Appt split: Yes={len(calls_by_appt['Yes'])}, No={len(calls_by_appt['No'])}, "
                f"Inconclusive={len(calls_by_appt['Inconclusive'])}, error={len(calls_by_appt['error'])}")

    # Call 2: combo for each appointment group
    combo_results = {}
    for appt_answer in ["Yes", "No", "Inconclusive"]:
        group_calls = calls_by_appt[appt_answer]
        if not group_calls:
            continue
        logger.info(f"  Call 2 (appt={appt_answer}): {len(group_calls)} calls, bs={combo_bs}")
        batches = [group_calls[i:i + combo_bs] for i in range(0, len(group_calls), combo_bs)]
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(call_batch_combo, client, model, batch, appt_answer, A1B_COMBO_SCHEMA_BATCH): i
                for i, batch in enumerate(batches)
            }
            for f in as_completed(futures):
                try:
                    combo_results.update(f.result())
                except Exception as e:
                    logger.error(f"  combo thread error: {e}")

    # Assemble field results
    field_results = {f: {} for f in FIELDS}
    for c in calls:
        cid = c["id"]
        field_results["appointment_booked"][cid] = {
            "call_id": cid, "answer": appt_results.get(cid, {}).get("answer"),
        }
        combo = combo_results.get(cid, {})
        for f in ["client_type", "treatment_type", "reason_not_booked"]:
            field_results[f][cid] = {"call_id": cid, "answer": combo.get(f)}

    return field_results


# ============================================================
# A1b+E: 2-CALL FREE TEXT → SHARED FORMATTER
# ============================================================

def run_a1b_freetext(client, model, calls, appt_bs, combo_bs, format_bs):
    """A1b+E: 2-call free text → shared formatter."""
    logger.info("=== A1b+E (2-call free text → shared formatter) ===")

    # Call 1: appointment_booked (still with strict schema — only 3 values)
    logger.info(f"  Call 1: appointment_booked ({len(calls)} calls, bs={appt_bs})")
    appt_batches = [calls[i:i + appt_bs] for i in range(0, len(calls), appt_bs)]
    appt_results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_batch_appt, client, model, batch, V9_APPOINTMENT_BOOKED_SCHEMA): i
            for i, batch in enumerate(appt_batches)
        }
        for f in as_completed(futures):
            try:
                appt_results.update(f.result())
            except Exception as e:
                logger.error(f"  appt thread error: {e}")

    # Group by appointment answer
    calls_by_appt = {"Yes": [], "No": [], "Inconclusive": [], "error": []}
    for c in calls:
        cid = c["id"]
        answer = appt_results.get(cid, {}).get("answer", "")
        if answer in calls_by_appt:
            calls_by_appt[answer].append(c)
        else:
            calls_by_appt["error"].append(c)

    # Call 2: combo WITHOUT strict schema (free text)
    combo_free = {}
    for appt_answer in ["Yes", "No", "Inconclusive"]:
        group_calls = calls_by_appt[appt_answer]
        if not group_calls:
            continue
        logger.info(f"  Call 2 free-text (appt={appt_answer}): {len(group_calls)} calls")
        batches = [group_calls[i:i + combo_bs] for i in range(0, len(group_calls), combo_bs)]
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(call_batch_combo, client, model, batch, appt_answer, None): i
                for i, batch in enumerate(batches)
            }
            for f in as_completed(futures):
                try:
                    combo_free.update(f.result())
                except Exception as e:
                    logger.error(f"  combo free thread error: {e}")

    # Step 3: Shared formatter
    enums_for_prompt = {}
    for field in ["client_type", "treatment_type", "reason_not_booked"]:
        enums_for_prompt[field] = [e for e in FIELD_ENUMS[field] if e is not None]
        if None in FIELD_ENUMS[field]:
            enums_for_prompt[field].append("null")

    items_to_format = []
    for cid, item in combo_free.items():
        if item.get("error"):
            continue
        classifications = {}
        for f in ["client_type", "treatment_type", "reason_not_booked"]:
            val = item.get(f)
            classifications[f] = str(val) if val is not None else "null"
        items_to_format.append({"id": cid, "classifications": classifications})

    field_results = {f: {} for f in FIELDS}
    # appointment_booked comes from Call 1 (strict schema)
    for c in calls:
        cid = c["id"]
        field_results["appointment_booked"][cid] = {
            "call_id": cid, "answer": appt_results.get(cid, {}).get("answer"),
        }

    logger.info(f"  Formatter: {len(items_to_format)} items in batches of {format_bs}")
    batches = [items_to_format[i:i + format_bs] for i in range(0, len(items_to_format), format_bs)]

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

Return JSON: {{"results": [{{"call_id": "...", "client_type": "...", "treatment_type": "...", "reason_not_booked": "... or null"}}]}}
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
                for field in ["client_type", "treatment_type", "reason_not_booked"]:
                    val = r.get(field)
                    if val == "null" or val is None:
                        val = None
                    field_results[field][cid] = {"call_id": cid, "answer": val}
        except Exception as e:
            logger.error(f"  Format batch failed: {e}")
            for it in batch:
                for field in ["client_type", "treatment_type", "reason_not_booked"]:
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
    parser = argparse.ArgumentParser(description="Test A1b architecture variants")
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--appt-bs", type=int, default=15, help="Call 1 batch size")
    parser.add_argument("--combo-bs", type=int, default=15, help="Call 2 batch size")
    parser.add_argument("--format-bs", type=int, default=15, help="Formatter batch size")
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

    # A1b baseline
    token_usage["input"] = 0; token_usage["output"] = 0
    all_results["A1b"] = run_a1b(client, model, calls, args.appt_bs, args.combo_bs)

    # A1b+E (free text + formatter)
    token_usage["input"] = 0; token_usage["output"] = 0
    all_results["A1b+E"] = run_a1b_freetext(client, model, calls, args.appt_bs, args.combo_bs, args.format_bs)

    # Scoring
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: A1b vs A1b+E")
    logger.info("=" * 60)

    header = f"{'Field':<25} | {'A1b':>10} | {'A1b+E':>10}"
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
        a1b_acc = summary["A1b"][field]["accuracy"]
        a1be_acc = summary["A1b+E"][field]["accuracy"]
        delta = a1be_acc - a1b_acc
        logger.info(f"{field:<25} | {a1b_acc:>9.1f}% | {a1be_acc:>9.1f}% ({delta:+.1f})")

    a1b_avg = sum(s["accuracy"] for s in summary["A1b"].values()) / len(FIELDS)
    a1be_avg = sum(s["accuracy"] for s in summary["A1b+E"].values()) / len(FIELDS)
    logger.info(f"{'AVERAGE':<25} | {a1b_avg:>9.1f}% | {a1be_avg:>9.1f}% ({a1be_avg - a1b_avg:+.1f})")

    # Error details for key fields
    for field in ["treatment_type", "reason_not_booked"]:
        logger.info(f"\n--- {field} errors ---")
        for arch_name in all_results:
            _, _, errors = score_field(all_results[arch_name][field], data, field)
            logger.info(f"  [{arch_name}] {len(errors)} errors, top 5:")
            for e in errors[:5]:
                logger.info(f"    {e['call_id']}: pred={e['pred']!r} gold={e['gold']!r}")

    # Save
    output_path = args.output or f"results_a1b_test_n{len(calls)}.json"
    with open(output_path, "w") as f:
        json.dump({"n": len(calls), "seed": args.seed, "summary": summary}, f, indent=2)
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
