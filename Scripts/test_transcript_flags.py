#!/usr/bin/env python3
"""
Test whether transcript quality flags correlate with prediction errors.
Runs on existing v9.3 Flash results — no LLM calls needed.
"""
import os, csv, json, re
from collections import Counter

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    return transcripts, labels


def flag_transcript(transcript: str) -> list:
    """Return list of quality flags for a transcript."""
    flags = []

    # Very short
    turns = len(re.findall(r'(Agent:|Caller:)', transcript))
    if turns < 3 or len(transcript) < 100:
        flags.append("very_short")

    # Voicemail / automated
    voicemail_patterns = [
        r'office is currently closed',
        r'leave a message',
        r'leave your (name|number|message)',
        r'after the (tone|beep)',
        r'our (hours|office hours) are',
        r'we are (currently )?(closed|unavailable)',
        r'press \d',
        r'you have reached',
    ]
    # Only agent speaks (no "Caller:" tag) or very few caller words
    caller_parts = re.findall(r'Caller:\s*(.*?)(?=Agent:|$)', transcript, re.DOTALL)
    caller_words = sum(len(part.split()) for part in caller_parts)

    if caller_words < 5:
        flags.append("voicemail")
    elif any(re.search(p, transcript, re.IGNORECASE) for p in voicemail_patterns):
        if turns < 4:
            flags.append("voicemail")

    # Garbled / heavily redacted
    redacted_tags = len(re.findall(r'\[(MEDICAL_CONDITION|DRUG|MEDICAL_PROCESS)\]', transcript))
    total_words = len(transcript.split())
    if total_words > 0 and redacted_tags / total_words > 0.05:
        flags.append("garbled")

    # Wrong number
    wrong_number_patterns = [
        r'wrong number',
        r'called the wrong',
        r'sorry.{0,20}wrong',
        r'is this the .{5,30}\? .{0,20}(nope|no,? this is)',
    ]
    if any(re.search(p, transcript, re.IGNORECASE) for p in wrong_number_patterns):
        if len(transcript) < 500:
            flags.append("wrong_number")

    # Very long (potential multi-topic call — harder to classify)
    if len(transcript) > 5000:
        flags.append("long_complex")

    return flags


def main():
    transcripts, labels = load_data()

    # Load a results file if available
    results_files = [
        ("v9.3 Flash n=100 (1-100)", "results_v9_3_flash_n100.json", 0, 100),
        ("v9.3 Flash n=100 (two-step)", "results_two_step_flash_v3_n100.json", 0, 100),
    ]

    # First: just flag all transcripts and show distribution
    call_ids = sorted(set(transcripts.keys()) & set(labels.keys()))[:200]

    print(f"{'='*70}")
    print(f"  TRANSCRIPT FLAG ANALYSIS (n={len(call_ids)})")
    print(f"{'='*70}")

    flag_counts = Counter()
    flagged_calls = {}
    unflagged = 0

    for cid in call_ids:
        flags = flag_transcript(transcripts[cid])
        flagged_calls[cid] = flags
        if flags:
            for f in flags:
                flag_counts[f] += 1
        else:
            unflagged += 1

    total_flagged = sum(1 for f in flagged_calls.values() if f)
    print(f"\n  Flagged: {total_flagged}/{len(call_ids)} ({total_flagged/len(call_ids):.1%})")
    print(f"  Unflagged: {unflagged}/{len(call_ids)} ({unflagged/len(call_ids):.1%})")
    print(f"\n  Flag distribution:")
    for flag, count in flag_counts.most_common():
        print(f"    {flag:20s}: {count:3d} ({count/len(call_ids):.1%})")

    # Now check correlation with errors using the v9.3 Flash full pipeline results
    print(f"\n{'='*70}")
    print(f"  CORRELATION: FLAGS vs ERRORS")
    print(f"{'='*70}")

    # Use the full pipeline Flash results if available
    for results_name, results_file, offset, count in results_files:
        results_path = os.path.join(project_dir, results_file)
        if not os.path.exists(results_path):
            continue

        with open(results_path) as f:
            data = json.load(f)

        predictions = data.get("predictions", data.get("final_results", {}))
        test_ids = sorted(set(transcripts.keys()) & set(labels.keys()))[offset:offset+count]

        print(f"\n  Results: {results_name}")

        fields = ["appointment_booked", "client_type", "treatment_type", "reason_not_booked"]

        for field in fields:
            flagged_correct = 0
            flagged_wrong = 0
            unflagged_correct = 0
            unflagged_wrong = 0

            for cid in test_ids:
                if cid not in predictions:
                    continue
                gold = labels[cid].get(field, "").strip()
                pred = (predictions[cid].get(field) or "").strip()

                if not gold or gold.lower() in ("null", "none"):
                    gold = ""
                if not pred or pred.lower() in ("null", "none"):
                    pred = ""

                if field == "reason_not_booked" and not gold and not pred:
                    continue

                flags = flagged_calls.get(cid, [])
                is_correct = (gold == pred)

                if flags:
                    if is_correct:
                        flagged_correct += 1
                    else:
                        flagged_wrong += 1
                else:
                    if is_correct:
                        unflagged_correct += 1
                    else:
                        unflagged_wrong += 1

            flagged_total = flagged_correct + flagged_wrong
            unflagged_total = unflagged_correct + unflagged_wrong
            flagged_acc = flagged_correct / flagged_total if flagged_total > 0 else 0
            unflagged_acc = unflagged_correct / unflagged_total if unflagged_total > 0 else 0

            print(f"\n    {field}:")
            print(f"      Flagged calls:   {flagged_acc:.1%} accuracy ({flagged_correct}/{flagged_total})")
            print(f"      Unflagged calls: {unflagged_acc:.1%} accuracy ({unflagged_correct}/{unflagged_total})")
            if flagged_total > 0 and unflagged_total > 0:
                print(f"      Gap: {unflagged_acc - flagged_acc:.1%} (unflagged is better)")

    # Show specific flagged calls with their errors
    print(f"\n{'='*70}")
    print(f"  FLAGGED CALLS DETAIL")
    print(f"{'='*70}")

    for cid in call_ids:
        flags = flagged_calls[cid]
        if not flags:
            continue
        t = transcripts[cid]
        print(f"\n  {cid[:24]}... flags={flags}")
        print(f"    Transcript ({len(t)} chars): {t[:150]}...")
        print(f"    Gold: appt={labels[cid].get('appointment_booked','')}, client={labels[cid].get('client_type','')}, treat={labels[cid].get('treatment_type','')[:40]}")


if __name__ == "__main__":
    main()
