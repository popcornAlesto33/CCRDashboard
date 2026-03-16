#!/usr/bin/env python3
"""
Test appointment_booked prompts in isolation with Flash.
Runs just this one field — fast and cheap.
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
logger = logging.getLogger("test_appt")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

_semaphore = threading.Semaphore(15)

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "appointment_booked_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["Yes", "No", "Inconclusive"]},
                "confidence": {"type": "integer"}
            },
            "required": ["reasoning", "answer", "confidence"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# PROMPT VARIANTS TO TEST
# ============================================================

PROMPTS = {
    "v9.3": """You are a veterinary call transcript analyst. Determine whether an appointment was booked in this call.

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

Also rate your confidence from 0 to 100 (0 = pure guess, 100 = completely certain).

Return JSON: {"reasoning": "...", "answer": "Yes" | "No" | "Inconclusive", "confidence": 0-100}
Return JSON ONLY.""".strip(),

    "minimal": """You are a veterinary call transcript analyst. Determine whether an appointment was booked in this call.

Answer Yes, No, or Inconclusive.

- Yes: appointment confirmed or caller agreed to come in
- No: caller left without an appointment (includes: declined, schedule full, callback list, got info and left)
- Inconclusive: call wasn't about booking (admin, checking results, voicemail)

Return JSON: {"reasoning": "brief explanation", "answer": "Yes" | "No" | "Inconclusive"}
Return JSON ONLY.""".strip(),

    "guided": """You are a veterinary call transcript analyst. Determine whether an appointment was booked in this call.

## Yes
- A specific date/time is confirmed
- Caller agreed to come in (including emergency walk-ins without exact times)
- Existing appointment rescheduled to a new time

## No
- Caller chose not to book (said "I'll think about it", hung up after getting info)
- Schedule was full — caller left without an appointment, even if clinic said they'd check/call back
- Caller was put on a cancellation list (no confirmed appointment)
- Call was purely informational (pricing, hours, medication advice) and ended without scheduling
- Caller explicitly declined

## Inconclusive
- Call went to voicemail or automated system — no real conversation
- The call was administrative (checking results, updating records) — booking was never discussed
- Inter-clinic consultation where no patient appointment was the purpose
- Clinic will call back about a medical decision (doctor reviewing test results)

Key: if the caller TRIED to book but couldn't, that's No. Inconclusive is for calls where booking was never the purpose.

Return JSON: {"reasoning": "brief explanation", "answer": "Yes" | "No" | "Inconclusive"}
Return JSON ONLY.""".strip(),
}


def call_llm(client, model, call_id, transcript, prompt, schema):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ]
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                response_format=schema,
            )
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
    parser.add_argument("--offset", type=int, default=0, help="Skip first N calls")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--prompt", default="all", choices=list(PROMPTS.keys()) + ["all"])
    args = parser.parse_args()

    data = load_data()
    call_ids = sorted(data.keys())[args.offset:args.offset + args.max_calls]

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        max_retries=3,
    )

    prompts_to_test = PROMPTS if args.prompt == "all" else {args.prompt: PROMPTS[args.prompt]}

    for name, prompt in prompts_to_test.items():
        logger.info(f"Testing prompt '{name}' on {len(call_ids)} calls with {args.model}")

        results = {}
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {
                pool.submit(call_llm, client, args.model, cid, data[cid]["transcript"], prompt, SCHEMA): cid
                for cid in call_ids
            }
            for f in as_completed(futures):
                cid = futures[f]
                results[cid] = f.result()

        # Score
        correct = total = 0
        mismatches = []
        no_to_inc = inc_to_no = yes_errors = 0
        correct_confidences = []
        wrong_confidences = []

        for cid in call_ids:
            gold = data[cid]["labels"].get("appointment_booked", "").strip()
            pred = results[cid].get("answer", "")
            conf = results[cid].get("confidence", -1)
            if not gold: continue
            total += 1
            if gold == pred:
                correct += 1
                correct_confidences.append(conf)
            else:
                mismatches.append({"gold": gold, "pred": pred, "call_id": cid, "confidence": conf})
                wrong_confidences.append(conf)
                if gold == "No" and pred == "Inconclusive": no_to_inc += 1
                elif gold == "Inconclusive" and pred == "No": inc_to_no += 1
                elif gold == "Yes" or pred == "Yes": yes_errors += 1

        acc = correct / total if total > 0 else 0
        print(f"\n{'='*70}")
        print(f"  PROMPT: {name} | {args.model} | n={total}")
        print(f"  Accuracy: {acc:.1%} ({correct}/{total})")
        print(f"  Errors: No→Inc={no_to_inc}, Inc→No={inc_to_no}, Yes-related={yes_errors}")
        print(f"{'='*70}")

        # Confidence calibration
        if correct_confidences and wrong_confidences:
            avg_correct = sum(correct_confidences) / len(correct_confidences)
            avg_wrong = sum(wrong_confidences) / len(wrong_confidences)
            print(f"\n  CONFIDENCE CALIBRATION:")
            print(f"    Avg confidence (correct): {avg_correct:.1f}")
            print(f"    Avg confidence (wrong):   {avg_wrong:.1f}")
            print(f"    Gap: {avg_correct - avg_wrong:.1f}pp")

            # Bucket analysis
            buckets = [(0, 50), (50, 70), (70, 85), (85, 100)]
            print(f"\n    {'Confidence':>12s}  {'Total':>6s}  {'Correct':>8s}  {'Accuracy':>9s}")
            for lo, hi in buckets:
                bucket_correct = sum(1 for c in correct_confidences if lo <= c < hi)
                bucket_wrong = sum(1 for c in wrong_confidences if lo <= c < hi)
                bucket_total = bucket_correct + bucket_wrong
                bucket_acc = bucket_correct / bucket_total if bucket_total > 0 else 0
                print(f"    {lo:>3d}-{hi:<3d}        {bucket_total:>5d}   {bucket_correct:>7d}   {bucket_acc:>8.1%}")
            # 100 bucket
            bucket_correct = sum(1 for c in correct_confidences if c == 100)
            bucket_wrong = sum(1 for c in wrong_confidences if c == 100)
            bucket_total = bucket_correct + bucket_wrong
            bucket_acc = bucket_correct / bucket_total if bucket_total > 0 else 0
            print(f"    {'100':>7s}        {bucket_total:>5d}   {bucket_correct:>7d}   {bucket_acc:>8.1%}")

        print(f"\n  ERRORS (sorted by confidence, lowest first):")
        for m in sorted(mismatches, key=lambda x: x["confidence"])[:20]:
            print(f"    conf={m['confidence']:3d}  gold={m['gold']:13s} pred={m['pred']:13s} | {m['call_id'][:20]}...")
        if len(mismatches) > 20:
            print(f"    ... and {len(mismatches) - 20} more")


if __name__ == "__main__":
    main()
