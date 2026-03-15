#!/usr/bin/env python3
"""
Two-step treatment_type: Flash picks parent, GPT-5 decides sub-category.
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
logger = logging.getLogger("two_step_hybrid")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

_semaphore = threading.Semaphore(15)

# ============================================================
# STEP 1: Flash picks parent category (11 choices)
# ============================================================

STEP1_PROMPT = """You are a veterinary call transcript analyst. Determine the BROAD category of veterinary service discussed in this call.

Choose ONE:

1. Preventive Care — wellness visits, vaccinations, parasite prevention, annual exams, routine bloodwork, flea/tick products. DEFAULT for rescheduling/admin calls when service type is unclear.
2. Urgent Care / Sick Pet — sick or injured animal needing attention. Symptoms: vomiting, limping, coughing, urinary issues, eye problems, "not doing well." Medication advice for sick pets goes here too.
3. Surgical Services — spays/neuters, soft tissue surgery, orthopedic, dental cleanings/extractions. Use when SURGERY is the primary reason.
4. Diagnostic Services — ONLY when the call is specifically and solely about a diagnostic procedure (X-ray, ultrasound, ECG). Routine bloodwork as part of wellness = Preventive Care.
5. Emergency & Critical Care — life-threatening or critical: trauma, poisoning, seizures, burns, inability to walk, severe bleeding, critical symptoms. Also when caller directed to an emergency hospital.
6. Dermatology — skin, coat, EAR, or allergy issues as PRIMARY complaint. Ear infections, itching, rashes, Apoquel/Cytopoint.
7. Retail — food orders, prescription REFILLS for existing medications.
8. End of Life Care — euthanasia discussions or appointments.
9. N/A (missed call) — voicemail or automated message with NO real conversation.
10. Other — LAST RESORT. Do NOT use for rescheduling/admin calls.

You will receive multiple transcripts as a JSON array. Classify EACH one independently.

Return JSON: {"results": [{"call_id": "...", "reasoning": "brief reason", "answer": "<category name>"}, ...]}
Return JSON ONLY.""".strip()

STEP1_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "parent_category_batch",
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
                            "answer": {"type": "string", "enum": [
                                "Preventive Care", "Urgent Care / Sick Pet",
                                "Surgical Services", "Diagnostic Services",
                                "Emergency & Critical Care", "Dermatology",
                                "Retail", "End of Life Care",
                                "N/A (missed call)", "Other",
                            ]}
                        },
                        "required": ["call_id", "reasoning", "answer"],
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
# STEP 2: GPT-5 decides sub-category (per parent)
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
    "Urgent Care / Sick Pet": """Decide between staying at PARENT or picking a sub-category.

Use the PARENT "Urgent Care / Sick Pet" when:
- Only vague/general symptoms are described (limping, not eating, bleeding, "not doing well")
- No specific condition or diagnosis is identified
- The call is about general advice for a sick pet

Use a SUB-CATEGORY when the transcript identifies a SPECIFIC condition or diagnostic direction:
- "Diagnosis and Treatment of Illnesses" — specific illness symptoms named (vomiting + fever = possible infection, diabetes symptoms, named infections)
- "Chronic Disease Management" — ongoing/recurring condition (arthritis, thyroid disease, long-term allergies, Cushing's)
- "Internal Medicine Workups" — specific tests/imaging discussed (bloodwork to investigate, X-rays ordered, specialist referral)""",

    "Preventive Care": """Decide between staying at PARENT or picking a sub-category.

Use the PARENT "Preventive Care" when:
- Multiple preventive services are discussed (checkup + vaccines + bloodwork)
- It's a general wellness visit or new pet checkup
- Admin/rescheduling with no specific service mentioned

Use a SUB-CATEGORY when one specific service is the PRIMARY purpose:
- "Vaccinations" — caller's main reason is vaccines/shots (even if checkup is also mentioned)
- "Annual Exams" — caller explicitly says "annual", "yearly", or "wellness" exam
- "Parasite Prevention" — flea/tick/heartworm prevention is the main topic
- "Wellness Screening" — routine bloodwork, urinalysis, or fecals as primary purpose""",

    "Emergency & Critical Care": """Decide between staying at PARENT or picking a sub-category.

Use the PARENT when it's clearly an emergency but no specific intervention is described.

Use a SUB-CATEGORY when the specific emergency type is clear:
- "Stabilization" — active trauma (hit by car, porcupine), poisoning, seizures
- "Overnight Hospitalization" — pet needs to stay overnight
- "Fluid/Oxygen Therapy" — specific supportive care discussed
- "Referred to Emergency Hospital" — caller is being DIRECTED to a different emergency facility""",
}


def make_step2_prompt(parent):
    subs = SUB_CATEGORIES.get(parent)
    if not subs:
        return None

    options = "\n".join(f"- {s}" for s in subs)
    guidance = STEP2_GUIDANCE.get(parent, "Stay at parent unless a specific sub-category clearly fits.")

    return f"""You are a veterinary call transcript analyst. This call has been classified as "{parent}".

Now determine: is there enough information to pick a specific sub-category, or should it stay at the parent level? The PARENT is always a valid answer — use it when the transcript is general or ambiguous.

{guidance}

Options:
{options}

You will receive multiple transcripts as a JSON array. Classify EACH one independently.

Return JSON: {{"results": [{{"call_id": "...", "reasoning": "...", "answer": "<exact category string>"}}, ...]}}
Return JSON ONLY.""".strip()


def make_step2_schema(parent):
    subs = SUB_CATEGORIES.get(parent)
    if not subs:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sub_category_batch",
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
                                "answer": {"type": "string", "enum": subs},
                            },
                            "required": ["call_id", "reasoning", "answer"],
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
# EXECUTION
# ============================================================

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
            return {str(item.get("call_id", "")): item for item in result.get("results", [])}
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"  {step_name} batch attempt {attempt+1} failed: {e}")
            else:
                logger.error(f"  {step_name} batch failed: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            _semaphore.release()


def run_batched(client, model, calls, prompt, batch_size, step_name, schema=None, use_temp=True):
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
    logger.info(f"  {step_name}: {len(calls)} calls in {len(batches)} batches of {batch_size} ({model})")
    results = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(call_batch, client, model, b, prompt, step_name, schema, use_temp): i for i, b in enumerate(batches)}
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

    flash_client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        max_retries=3,
    )
    gpt5_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_retries=3,
    )

    # Step 1: Flash picks parent (bs=15)
    logger.info("Step 1: Flash parent classification...")
    step1_results = run_batched(flash_client, "gemini-2.5-flash", calls, STEP1_PROMPT, 15, "step1_parent", STEP1_SCHEMA)

    from collections import Counter
    parents = Counter(r.get("answer", "") for r in step1_results.values())
    logger.info(f"Step 1 complete. Parents: {dict(parents)}")

    # Step 2: GPT-5 picks sub-category (bs=5), grouped by parent
    logger.info("Step 2: GPT-5 sub-category classification...")
    final_results = {}

    # Group calls by parent
    parent_groups = {}
    for c in calls:
        cid = c["id"]
        parent = step1_results.get(cid, {}).get("answer", "Other")
        if parent in SUB_CATEGORIES:
            parent_groups.setdefault(parent, []).append(c)
        else:
            # No subs (N/A, Other, etc) — final answer is the parent
            final_results[cid] = {"call_id": cid, "answer": parent, "reasoning": "No sub-categories for this parent."}

    # Run Step 2 for each parent group in parallel
    with ThreadPoolExecutor(max_workers=len(parent_groups)) as pool:
        futures = {}
        for parent, group_calls in parent_groups.items():
            prompt = make_step2_prompt(parent)
            schema = make_step2_schema(parent)
            futures[pool.submit(
                run_batched, gpt5_client, "gpt-5", group_calls, prompt, 5,
                f"step2_{parent[:10]}", schema, False
            )] = parent

        for f in as_completed(futures):
            parent = futures[f]
            sub_results = f.result()
            final_results.update(sub_results)

    logger.info(f"Step 2 complete. {len(final_results)} results.")

    # Score
    from collections import Counter
    correct = total = 0
    mismatches = []
    for cid in call_ids:
        gold = data[cid]["labels"].get("treatment_type", "").strip()
        pred = final_results.get(cid, {}).get("answer", "")
        if not gold: continue
        total += 1
        if gold == pred:
            correct += 1
        else:
            mismatches.append({
                "call_id": cid, "gold": gold, "pred": pred,
                "step1": step1_results.get(cid, {}).get("answer", ""),
                "step2_reasoning": final_results.get(cid, {}).get("reasoning", "")[:150],
            })

    acc = correct / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"  TWO-STEP HYBRID: Flash(bs=15) parent → GPT-5(bs=5) sub | n={total}")
    print(f"  Accuracy: {acc:.1%} ({correct}/{total}) [target: 80%]")
    print(f"{'='*70}")

    # Error breakdown
    confusion = Counter()
    for m in mismatches:
        gp = m["gold"].split(" –")[0].split(" \u2013")[0]
        pp = m["pred"].split(" –")[0].split(" \u2013")[0]
        if gp == pp or (gp == "Urgent Care / Sick Pet" and pp.startswith("Urgent Care")) or (pp == "Urgent Care / Sick Pet" and gp.startswith("Urgent Care")):
            confusion["SAME-FAMILY"] += 1
        else:
            confusion[f"{gp} -> {pp}"] += 1
    print("\n  Error breakdown:")
    for pattern, count in confusion.most_common(15):
        print(f"    {pattern}: {count}x")

    print(f"\n  Mismatches ({len(mismatches)}):")
    for m in mismatches[:15]:
        print(f"    {m['call_id'][:20]}...: step1={m['step1'][:25]:25s} gold={m['gold'][:45]:45s} pred={m['pred'][:45]}")
    if len(mismatches) > 15:
        print(f"    ... and {len(mismatches) - 15} more")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": total,
                       "step1_results": {k: v for k, v in step1_results.items()},
                       "final_results": {k: v for k, v in final_results.items()},
                       "mismatches": mismatches}, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
