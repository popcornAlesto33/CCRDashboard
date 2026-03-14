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
from collections import Counter
from typing import Dict, List, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate")


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

    if args.single_model:
        logger.info(f"Running single-model validation with {args.classification_model}")
        results_list = run_single_model_batch(
            client, args.classification_model, calls, args.classification_batch_size
        )
        mode = f"single-model ({args.classification_model})"
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

    # Index predictions by call_id
    predictions = {}
    for item in results_list:
        cid = item.get("call_id")
        if cid:
            ab = item.get("appointment_booked", "")
            if ab == "Yes":
                item["reason_not_booked"] = None
            predictions[str(cid)] = item

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
        if not args.single_model:
            output_data["reasoning"] = {cid: r for cid, r in reasoning.items() if cid in predictions}

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
