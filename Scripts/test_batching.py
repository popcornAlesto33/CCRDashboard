#!/usr/bin/env python3
"""
Test batching impact on v9 field prompts.
Compares batch_size=1 (current) vs batch_size=5.
"""
import os, sys, csv, json, threading, time, logging, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_batch")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

_semaphore = threading.Semaphore(15)

# Import prompts and schemas from the main script
sys.path.insert(0, script_dir)
from validate_prompt_engineering import (
    V9_APPOINTMENT_BOOKED_PROMPT, V9_CLIENT_TYPE_PROMPT,
    V9_TREATMENT_TYPE_PROMPT, V9_REASON_NOT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA, V9_CLIENT_TYPE_SCHEMA,
    V9_TREATMENT_TYPE_SCHEMA, V9_REASON_NOT_BOOKED_SCHEMA,
)

# ============================================================
# BATCHED PROMPTS — wrap single-call prompts for multi-transcript
# ============================================================

def make_batched_prompt(base_prompt: str) -> str:
    """Convert a single-transcript prompt into a batched one."""
    # Replace the output format section
    return base_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. Classify EACH one independently.\n\n'
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", "answer": ...}, ...]}\n'
        'Return JSON ONLY.'
    )


def make_batched_schema(single_schema: dict) -> dict:
    """Convert a single-result schema into a batched one."""
    single_item = single_schema["json_schema"]["schema"].copy()
    # Add call_id to the item
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


# ============================================================
# API CALL FUNCTIONS
# ============================================================

def call_single(client, model, call_id, transcript, prompt, step_name, schema=None):
    """Single transcript per API call."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ]
    # GPT-5 doesn't support temperature=0.0
    kwargs = {"model": model, "messages": messages, "response_format": schema or {"type": "json_object"}}
    if "gpt-5" not in model:
        kwargs["temperature"] = 0.0
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)
            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
            else:
                return {"call_id": call_id, "error": str(e)}
        finally:
            _semaphore.release()


def call_batch(client, model, calls_batch, prompt, step_name, schema=None):
    """Multiple transcripts per API call."""
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ]
    kwargs = {"model": model, "messages": messages, "response_format": schema or {"type": "json_object"}}
    if "gpt-5" not in model:
        kwargs["temperature"] = 0.0
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)
            # Track token usage
            if resp.usage:
                in_tok = resp.usage.prompt_tokens or 0
                out_tok = resp.usage.completion_tokens or 0
                logger.info(f"  {step_name} batch ({len(calls_batch)} calls): {in_tok:,} in + {out_tok:,} out = {in_tok+out_tok:,} tokens")
            result = json.loads(resp.choices[0].message.content)
            # Extract results array
            items = result.get("results", [])
            return {str(item.get("call_id", "")): item for item in items}
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step_name} batch attempt {attempt+1} failed: {e}")
            else:
                logger.error(f"  {step_name} batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def run_field_single(client, model, calls, prompt, step_name, schema=None):
    """Run field with batch_size=1 (current approach)."""
    results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_single, client, model, c["id"], c["transcript"], prompt, step_name, schema): c["id"]
            for c in calls
        }
        for f in as_completed(futures):
            cid = futures[f]
            results[cid] = f.result()
    return results


def run_field_batched(client, model, calls, prompt, batch_size, step_name, schema=None):
    """Run field with batch_size > 1."""
    batched_prompt = make_batched_prompt(prompt)
    batched_schema = make_batched_schema(schema) if schema else None

    results = {}
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_batch, client, model, batch, batched_prompt, step_name, batched_schema): i
            for i, batch in enumerate(batches)
        }
        for f in as_completed(futures):
            batch_results = f.result()
            results.update(batch_results)

    return results


# ============================================================
# DATA & SCORING
# ============================================================

def load_data():
    transcripts = {}
    with open(os.path.join(project_dir, "CallData", "Transcript_details.csv"), "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] != "NULL" and row[1].strip():
                transcripts[row[0]] = row[1]
    labels = {}
    with open(os.path.join(project_dir, "CallData", "VetCare_CallInsight_Labels - labels.csv"), "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row
    joined = {}
    for cid in transcripts:
        if cid in labels:
            joined[cid] = {"id": cid, "transcript": transcripts[cid], "labels": labels[cid]}
    return joined


def score_field(predictions, gold_data, field):
    correct = total = 0
    for cid, pred in predictions.items():
        if cid not in gold_data: continue
        gold_val = gold_data[cid]["labels"].get(field, "").strip()
        pred_val = (pred.get("answer") or "").strip()
        if not gold_val or gold_val.lower() in ("null", "none"): gold_val = ""
        if not pred_val or pred_val.lower() in ("null", "none"): pred_val = ""
        if field == "reason_not_booked" and not gold_val and not pred_val: continue
        total += 1
        if gold_val == pred_val: correct += 1
    return correct, total


FIELD_CONFIG = {
    "appointment_booked": (V9_APPOINTMENT_BOOKED_PROMPT, V9_APPOINTMENT_BOOKED_SCHEMA),
    "client_type": (V9_CLIENT_TYPE_PROMPT, V9_CLIENT_TYPE_SCHEMA),
    "treatment_type": (V9_TREATMENT_TYPE_PROMPT, V9_TREATMENT_TYPE_SCHEMA),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--field", default="all", choices=list(FIELD_CONFIG.keys()) + ["all"])
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--skip-single", action="store_true", help="Skip batch_size=1 baseline (use previous results)")
    args = parser.parse_args()

    data = load_data()
    call_ids = sorted(data.keys())[:args.max_calls]
    calls = [{"id": cid, "transcript": data[cid]["transcript"]} for cid in call_ids]

    # Set up client based on provider
    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    else:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    if not api_key:
        raise RuntimeError(f"No API key found for provider '{args.provider}'")

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)

    fields_to_test = FIELD_CONFIG if args.field == "all" else {args.field: FIELD_CONFIG[args.field]}

    print(f"\n{'='*70}")
    print(f"  BATCHING TEST: {args.model} | n={len(calls)} | batch_size=1 vs {args.batch_size}")
    print(f"{'='*70}")

    for field, (prompt, schema) in fields_to_test.items():
        logger.info(f"Testing {field}...")

        # batch_size=1 (skip if --skip-single)
        if args.skip_single:
            results_single = {}
            c1, t1, t_single = 0, 0, 0
        else:
            t0 = time.time()
            results_single = run_field_single(client, args.model, calls, prompt, field, schema)
            t_single = time.time() - t0
            c1, t1 = score_field(results_single, data, field)

        # batch_size=N
        t0 = time.time()
        results_batched = run_field_batched(client, args.model, calls, prompt, args.batch_size, field, schema)
        t_batched = time.time() - t0
        c5, t5 = score_field(results_batched, data, field)

        acc1 = c1/t1 if t1 else 0
        acc5 = c5/t5 if t5 else 0
        delta = acc5 - acc1

        print(f"\n  {field}:")
        if not args.skip_single:
            print(f"    batch_size=1: {acc1:.1%} ({c1}/{t1}) in {t_single:.1f}s ({len(calls)} API calls)")
        print(f"    batch_size={args.batch_size}: {acc5:.1%} ({c5}/{t5}) in {t_batched:.1f}s ({len(calls)//args.batch_size + (1 if len(calls)%args.batch_size else 0)} API calls)")
        if not args.skip_single:
            print(f"    Delta: {delta:+.1%}pp | Speed: {t_single/t_batched:.1f}x faster")

        # Show where they disagree
        disagree = 0
        for cid in call_ids:
            a1 = results_single.get(cid, {}).get("answer", "")
            a5 = results_batched.get(cid, {}).get("answer", "")
            if a1 != a5:
                disagree += 1
                gold = data[cid]["labels"].get(field, "").strip()
                if disagree <= 5:
                    print(f"    DISAGREE {cid[:20]}...: single={a1}, batch={a5}, gold={gold}")
        if disagree > 5:
            print(f"    ... and {disagree - 5} more disagreements")
        print(f"    Total disagreements: {disagree}/{len(call_ids)}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
