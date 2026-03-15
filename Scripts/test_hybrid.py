#!/usr/bin/env python3
"""
Test hybrid pipeline: Flash for appointment_booked + client_type, GPT-5 for treatment_type + reason_not_booked.
Saves detailed output with reasoning for error analysis.
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
logger = logging.getLogger("hybrid")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

sys.path.insert(0, script_dir)
from validate_prompt_engineering import (
    V9_APPOINTMENT_BOOKED_PROMPT, V9_CLIENT_TYPE_PROMPT,
    V9_TREATMENT_TYPE_PROMPT, V9_REASON_NOT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA, V9_CLIENT_TYPE_SCHEMA,
    V9_TREATMENT_TYPE_SCHEMA, V9_REASON_NOT_BOOKED_SCHEMA,
    _make_batched_prompt, _make_batched_schema,
)

_semaphore = threading.Semaphore(15)


def call_batch(client, model, calls_batch, prompt, step_name, schema=None, use_temp=True):
    payload = json.dumps([{"call_id": c["id"], "transcript": c["transcript"]} for c in calls_batch])
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": payload}]
    kwargs = {"model": model, "messages": messages, "response_format": schema or {"type": "json_object"}}
    if use_temp:
        kwargs["temperature"] = 0.0

    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)
            result = json.loads(resp.choices[0].message.content)
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


def run_field(client, model, calls, prompt, batch_size, step_name, schema=None, use_temp=True):
    batched_prompt = _make_batched_prompt(prompt)
    batched_schema = _make_batched_schema(schema) if schema else None
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
    logger.info(f"  {step_name}: {len(calls)} calls in {len(batches)} batches of {batch_size} ({model})")

    results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_batch, client, model, batch, batched_prompt, step_name, batched_schema, use_temp): i
            for i, batch in enumerate(batches)
        }
        for f in as_completed(futures):
            results.update(f.result())
    return results


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data = load_data()
    call_ids = sorted(data.keys())[:args.max_calls]
    calls = [{"id": cid, "transcript": data[cid]["transcript"]} for cid in call_ids]

    # Gemini Flash client
    flash_client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        max_retries=3,
    )
    # OpenAI GPT-5 client
    gpt5_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_retries=3,
    )

    logger.info(f"Hybrid pipeline: Flash(bs=15) + GPT-5(bs=5) on {len(calls)} calls")

    # Phase 1: Flash appointment_booked + client_type + GPT-5 treatment_type (parallel)
    logger.info("Phase 1: appointment_booked(Flash) + client_type(Flash) + treatment_type(GPT-5)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        appt_f = pool.submit(run_field, flash_client, "gemini-2.5-flash", calls,
                             V9_APPOINTMENT_BOOKED_PROMPT, 15, "appointment_booked",
                             V9_APPOINTMENT_BOOKED_SCHEMA)
        client_f = pool.submit(run_field, flash_client, "gemini-2.5-flash", calls,
                               V9_CLIENT_TYPE_PROMPT, 15, "client_type",
                               V9_CLIENT_TYPE_SCHEMA)
        treat_f = pool.submit(run_field, gpt5_client, "gpt-5", calls,
                              V9_TREATMENT_TYPE_PROMPT, 5, "treatment_type",
                              V9_TREATMENT_TYPE_SCHEMA, use_temp=False)

        appt_results = appt_f.result()
        client_results = client_f.result()
        treat_results = treat_f.result()

    logger.info(f"Phase 1 complete: appt={len(appt_results)}, client={len(client_results)}, treat={len(treat_results)}")

    # Phase 2: GPT-5 reason_not_booked (depends on appointment_booked)
    logger.info("Phase 2: reason_not_booked(GPT-5, depends on appointment_booked)...")
    reason_results = {}
    calls_needing_reason = []

    for c in calls:
        cid = c["id"]
        appt_answer = appt_results.get(cid, {}).get("answer", "")
        if appt_answer == "No":
            calls_needing_reason.append(c)
        else:
            reason_results[cid] = {"call_id": cid, "reasoning": f"appointment_booked={appt_answer}, skipping", "answer": None}

    if calls_needing_reason:
        logger.info(f"  reason_not_booked: {len(calls_needing_reason)}/{len(calls)} calls need LLM (appt=No)")
        prompt = V9_REASON_NOT_BOOKED_PROMPT.replace("{appointment_booked}", "No")
        llm_results = run_field(gpt5_client, "gpt-5", calls_needing_reason,
                                prompt, 5, "reason_not_booked",
                                V9_REASON_NOT_BOOKED_SCHEMA, use_temp=False)
        reason_results.update(llm_results)

    logger.info(f"Phase 2 complete: reason={len(reason_results)}")

    # Assemble + Score
    fields = ["appointment_booked", "client_type", "treatment_type", "reason_not_booked"]
    field_results_map = {
        "appointment_booked": appt_results,
        "client_type": client_results,
        "treatment_type": treat_results,
        "reason_not_booked": reason_results,
    }

    print(f"\n{'='*70}")
    print(f"  HYBRID PIPELINE: Flash(bs=15) + GPT-5(bs=5) | n={len(calls)}")
    print(f"{'='*70}")

    all_mismatches = {}
    for field in fields:
        results = field_results_map[field]
        correct = total = 0
        mismatches = []
        for cid in call_ids:
            gold = data[cid]["labels"].get(field, "").strip()
            pred = (results.get(cid, {}).get("answer") or "").strip()
            if not gold or gold.lower() in ("null", "none"): gold = ""
            if not pred or pred.lower() in ("null", "none"): pred = ""
            if field == "reason_not_booked" and not gold and not pred: continue
            total += 1
            if gold == pred:
                correct += 1
            else:
                reasoning = results.get(cid, {}).get("reasoning", "")
                mismatches.append({"call_id": cid, "gold": gold, "pred": pred, "reasoning": reasoning[:200]})

        acc = correct / total if total > 0 else 0
        target = {"appointment_booked": 0.90, "client_type": 0.90, "treatment_type": 0.80, "reason_not_booked": 0.85}[field]
        status = "PASS" if acc >= target else "FAIL"
        print(f"\n  {field}: {acc:.1%} ({correct}/{total}) [target: {target:.0%}] [{status}]")
        for m in mismatches[:10]:
            print(f"    {m['call_id'][:20]}...: gold={m['gold']!r} pred={m['pred']!r}")
            if m['reasoning']:
                print(f"      reasoning: {m['reasoning'][:120]}...")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        all_mismatches[field] = mismatches

    print(f"\n{'='*70}")

    # Treatment type confusion summary
    from collections import Counter
    tt = all_mismatches.get("treatment_type", [])
    if tt:
        confusion = Counter()
        for m in tt:
            gp = m["gold"].split(" –")[0].split(" \u2013")[0]
            pp = m["pred"].split(" –")[0].split(" \u2013")[0]
            if gp == pp:
                confusion["SAME-FAMILY (parent/sub)"] += 1
            else:
                confusion[f"{gp} -> {pp}"] += 1
        print("\n  treatment_type error breakdown:")
        for pattern, count in confusion.most_common(15):
            print(f"    {pattern}: {count}x")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "field_results": {
                    field: {cid: field_results_map[field].get(cid, {}) for cid in call_ids}
                    for field in fields
                },
                "mismatches": {field: all_mismatches[field] for field in fields},
            }, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
