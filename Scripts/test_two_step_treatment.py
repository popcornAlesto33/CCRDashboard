#!/usr/bin/env python3
"""
Test two-step Flash treatment_type classification:
  Step 1: Classify into parent category (11 choices)
  Step 2: Given parent, pick sub-category or stay at parent (2-5 choices)
"""
import os
import sys
import csv
import json
import threading
import time
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("two_step_treatment")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# ============================================================
# STEP 1 PROMPT: Parent classification (11 choices)
# ============================================================

STEP1_PARENT_PROMPT = """You are a veterinary call transcript analyst. Determine the BROAD category of veterinary service discussed in this call.

Choose ONE:

1. Preventive Care — wellness visits, vaccinations, parasite prevention, flea/tick products, annual exams, routine bloodwork. DEFAULT for rescheduling/admin calls when the service type is unclear.
2. Urgent Care / Sick Pet — sick or injured animal needing attention. Symptoms: vomiting, limping, coughing, urinary issues, eye problems, general "not doing well." Medication advice calls for sick pets go here too.
3. Surgical Services — spays/neuters, soft tissue surgery, orthopedic, dental cleanings/extractions. Use when a SURGERY is the primary reason.
4. Diagnostic Services — ONLY when the call is specifically and solely about a diagnostic procedure (X-ray, ultrasound, ECG). If bloodwork is part of a wellness/annual visit, use Preventive Care instead.
5. Emergency & Critical Care — life-threatening or critical situations: trauma, poisoning, seizures, burns, inability to walk, severe bleeding, critical symptoms. Also when caller is directed to an emergency hospital. Convulsions or collapse = Emergency, not Urgent Care.
6. Dermatology — skin, coat, EAR, or allergy issues as the PRIMARY complaint. Ear infections, itching, rashes, Apoquel/Cytopoint. Even if the pet is "sick" — if the main issue is skin/ears/allergies, use Dermatology.
7. Retail — food orders, prescription REFILLS for existing medications. NOT new prevention products (those are Preventive Care).
8. End of Life Care — euthanasia discussions or appointments. Convulsing pet where caller mentions bringing them "back" for end-of-life = End of Life Care, not Emergency.
9. N/A (missed call) — voicemail or automated message with NO real conversation. Short greetings that cut off = N/A.
10. Other — LAST RESORT. Wrong number calls with zero medical content. Do NOT use for rescheduling/admin calls.

## Examples

Rescheduling appointment, no medical detail mentioned → Preventive Care (default for admin calls)
Caller asking about cost of exam and bloodwork for new pet → Preventive Care (wellness visit)
Cat vomiting and in pain → Urgent Care / Sick Pet
Dog ran into porcupine, quills everywhere → Emergency & Critical Care
Caller needs spay for her dog → Surgical Services
Dog has recurring ear infection → Dermatology
Caller needs medication refill → Retail
Pet convulsing, bringing them in to be put down → End of Life Care
Voicemail greeting, no caller interaction → N/A (missed call)

Return JSON: {"reasoning": "...", "answer": "<category name>"}
Return JSON ONLY.""".strip()

STEP1_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "parent_category_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {
                    "type": "string",
                    "enum": [
                        "Preventive Care",
                        "Urgent Care / Sick Pet",
                        "Surgical Services",
                        "Diagnostic Services",
                        "Emergency & Critical Care",
                        "Dermatology",
                        "Retail",
                        "End of Life Care",
                        "N/A (missed call)",
                        "Other",
                    ]
                }
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# STEP 2 PROMPTS: Sub-category classification (per parent)
# ============================================================

SUB_CATEGORIES = {
    "Preventive Care": [
        "Preventive Care",
        "Preventive Care \u2013 Vaccinations",
        "Preventive Care \u2013 Parasite Prevention",
        "Preventive Care \u2013 Annual Exams",
        "Preventive Care \u2013 Wellness Screening (Bloodwork, Urinalysis, Fecals)",
    ],
    "Urgent Care / Sick Pet": [
        "Urgent Care / Sick Pet",
        "Urgent Care \u2013 Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)",
        "Urgent Care \u2013 Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)",
        "Urgent Care \u2013 Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)",
    ],
    "Surgical Services": [
        "Surgical Services",
        "Surgical Services \u2013 Spays and Neuters",
        "Surgical Services \u2013 Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)",
        "Surgical Services \u2013 Orthopedic Surgeries (ACL Repairs, Fracture Repair \u2014 Sometimes Referred Out)",
        "Surgical Services \u2013 Emergency Surgeries (Pyometra, C-Sections, GDV)",
        "Surgical Services \u2013 Dental Care (Cleanings, Extractions)",
    ],
    "Diagnostic Services": [
        "Diagnostic Services",
        "Diagnostic Services \u2013 X-Rays (Digital Radiography)",
        "Diagnostic Services \u2013 Ultrasound",
        "Diagnostic Services \u2013 In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)",
        "Diagnostic Services \u2013 ECG or Blood Pressure Monitoring",
    ],
    "Emergency & Critical Care": [
        "Emergency & Critical Care",
        "Emergency & Critical Care \u2013 Stabilization (Trauma, Poisoning, Seizures)",
        "Emergency & Critical Care \u2013 Overnight Hospitalization",
        "Emergency & Critical Care \u2013 Fluid Therapy, Oxygen Therapy, Intensive Monitoring",
        "Emergency & Critical Care \u2013 Referred to an Emergency Hospital",
    ],
    "Dermatology": [
        "Dermatology",
        "Dermatology \u2013 Allergies",
        "Dermatology \u2013 Ear Infections",
    ],
    "Retail": [
        "Retail",
        "Retail \u2013 Food Orders",
        "Retail \u2013 Prescriptions",
    ],
    "End of Life Care": [
        "End of Life Care",
        "End of Life Care \u2013 In-Home Euthanasia",
        "End of Life Care \u2013 In-Clinic Euthanasia",
    ],
}

STEP2_GUIDANCE = {
    "Preventive Care": """Tips:
- "Vaccinations" ONLY if vaccines/shots are the SOLE purpose. If vaccines are mentioned alongside a general checkup, use parent or Annual Exams.
- "Annual Exams" if caller explicitly says "annual", "yearly", or "wellness" checkup.
- "Wellness Screening" for routine bloodwork, urinalysis, fecals as part of wellness.
- "Parasite Prevention" for flea/tick/heartworm prevention plans.
- Use parent "Preventive Care" if multiple services discussed or purpose is general.""",

    "Urgent Care / Sick Pet": """Tips:
- Stay at parent "Urgent Care / Sick Pet" when the pet has general symptoms (limping, not eating, "not doing well") and no specific diagnosis or workup is discussed.
- Use "Diagnosis and Treatment of Illnesses" only when specific illnesses are named or the symptoms clearly point to a specific condition (vomiting + diabetes, infections).
- Use "Chronic Disease Management" for ongoing/recurring conditions (arthritis, thyroid, long-term allergies).
- Use "Internal Medicine Workups" when specific tests are discussed (bloodwork, imaging, specialist referral).
- When in doubt, stay at parent — it is a valid answer.""",

    "Emergency & Critical Care": """Tips:
- Stay at parent "Emergency & Critical Care" when it's clearly an emergency but no specific intervention is described.
- Use "Stabilization" for active trauma, poisoning, or seizures requiring immediate intervention.
- Use "Referred to an Emergency Hospital" ONLY when the caller is being directed to a DIFFERENT emergency facility.
- When in doubt, stay at parent — it is a valid answer.""",

    "Surgical Services": """Tips:
- Use specific sub only when the surgery type is clearly identified.
- "Dental Care" for cleanings and extractions.
- When in doubt, stay at parent.""",

    "Diagnostic Services": """Tips:
- Use specific sub only when the exact diagnostic is named.
- Stay at parent for general diagnostic visits.""",
}

def make_step2_prompt(parent: str) -> str:
    subs = SUB_CATEGORIES.get(parent)
    if not subs:
        return None  # No sub-categories (N/A, Other)

    options = "\n".join(f"- {s}" for s in subs)
    guidance = STEP2_GUIDANCE.get(parent, "")
    return f"""You are a veterinary call transcript analyst. This call has been classified as "{parent}".

Determine the most specific sub-category, OR stay at the parent level if there isn't enough detail. The parent category is ALWAYS a valid answer — use it when the call is general or ambiguous.

Options:
{options}

{guidance}

Return JSON: {{"reasoning": "...", "answer": "<exact category string>"}}
Return JSON ONLY.""".strip()


def make_step2_schema(parent: str):
    subs = SUB_CATEGORIES.get(parent)
    if not subs:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sub_category_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string", "enum": subs}
                },
                "required": ["reasoning", "answer"],
                "additionalProperties": False
            }
        }
    }


# ============================================================
# EXECUTION
# ============================================================

_semaphore = threading.Semaphore(15)


def call_llm(client, model, call_id, transcript, prompt, step_name, schema=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ]
    for attempt in range(3):
        _semaphore.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=schema or {"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step_name} {call_id} attempt {attempt+1} failed: {e}")
            else:
                logger.error(f"  {step_name} {call_id} failed: {e}")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data = load_data()
    call_ids = sorted(data.keys())[:args.max_calls]
    logger.info(f"Testing two-step treatment_type on {len(call_ids)} calls with {args.model}")

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=3,
    )

    # Step 1: Parent classification (all calls in parallel)
    logger.info("Step 1: Parent classification...")
    step1_results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(call_llm, client, args.model, cid, data[cid]["transcript"],
                        STEP1_PARENT_PROMPT, "step1", STEP1_SCHEMA): cid
            for cid in call_ids
        }
        for f in as_completed(futures):
            cid = futures[f]
            step1_results[cid] = f.result()

    # Count parents
    parent_counts = {}
    for cid, r in step1_results.items():
        p = r.get("answer", "error")
        parent_counts[p] = parent_counts.get(p, 0) + 1
    logger.info(f"Step 1 complete. Parent distribution: {parent_counts}")

    # Step 2: Sub-category classification (only for parents with subs)
    logger.info("Step 2: Sub-category classification...")
    final_results = {}
    step2_calls = []

    for cid in call_ids:
        parent = step1_results[cid].get("answer", "")
        if parent in SUB_CATEGORIES:
            step2_calls.append((cid, parent))
        else:
            # No sub-categories — final answer is the parent
            final_results[cid] = parent

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {}
        for cid, parent in step2_calls:
            prompt = make_step2_prompt(parent)
            schema = make_step2_schema(parent)
            futures[pool.submit(call_llm, client, args.model, cid, data[cid]["transcript"],
                                prompt, "step2", schema)] = cid

        for f in as_completed(futures):
            cid = futures[f]
            result = f.result()
            final_results[cid] = result.get("answer", step1_results[cid].get("answer", ""))

    # Score
    correct = 0
    total = 0
    mismatches = []
    for cid in call_ids:
        gold = data[cid]["labels"].get("treatment_type", "").strip()
        pred = final_results.get(cid, "").strip()
        if not gold:
            continue
        total += 1
        if gold == pred:
            correct += 1
        else:
            mismatches.append({"call_id": cid, "gold": gold, "predicted": pred})

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"  TWO-STEP TREATMENT_TYPE RESULTS ({args.model})")
    print(f"{'='*70}")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total}) [target: 80%]")
    print(f"  Mismatches ({len(mismatches)}):")
    for m in sorted(mismatches, key=lambda x: x["call_id"]):
        print(f"    {m['call_id'][:24]}...: gold={m['gold']!r}")
        print(f"    {'':24s}     pred={m['predicted']!r}")
    print(f"{'='*70}")

    # Confusion summary
    from collections import Counter
    confusion = Counter()
    for m in mismatches:
        g_parent = m["gold"].split(" –")[0].split(" \u2013")[0]
        p_parent = m["predicted"].split(" –")[0].split(" \u2013")[0]
        if g_parent == p_parent:
            confusion["SAME-FAMILY (parent/sub)"] += 1
        else:
            confusion[f"{g_parent} -> {p_parent}"] += 1
    print("\n  Error breakdown:")
    for pattern, count in confusion.most_common():
        print(f"    {pattern}: {count}x")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "accuracy": accuracy, "correct": correct, "total": total,
                "mismatches": mismatches,
                "step1_results": {k: v for k, v in step1_results.items()},
                "final_results": final_results,
            }, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
