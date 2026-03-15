#!/usr/bin/env python3
"""
Generate a CSV of flagged transcripts for human review.
Runs flag_transcript() on all calls and outputs flagged ones.
No LLM calls — pure transcript analysis.

Output: CallData/flagged_calls.csv
"""
import os, sys, csv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from validate_prompt_engineering import flag_transcript


def main():
    # Load transcripts
    transcripts = {}
    with open(os.path.join(project_dir, "CallData", "Transcript_details.csv"), "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] != "NULL" and row[1].strip():
                transcripts[row[0]] = row[1]

    # Load labels (if available — might not exist for new calls)
    labels = {}
    labels_path = os.path.join(project_dir, "CallData", "VetCare_CallInsight_Labels - labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                labels[row["id"]] = row

    # Flag all transcripts
    flagged_rows = []
    total = 0
    for cid in sorted(transcripts.keys()):
        total += 1
        t = transcripts[cid]
        flags = flag_transcript(t)
        if not flags:
            continue

        # Build row
        label = labels.get(cid, {})
        flagged_rows.append({
            "call_id": cid,
            "flags": "|".join(flags),
            "flag_count": len(flags),
            "transcript_chars": len(t),
            "transcript_preview": t[:300].replace("\n", " "),
            # Include gold labels if available (for review context)
            "gold_appointment_booked": label.get("appointment_booked", ""),
            "gold_client_type": label.get("client_type", ""),
            "gold_treatment_type": label.get("treatment_type", ""),
            "gold_reason_not_booked": label.get("reason_not_booked", ""),
            # Review columns (empty — for human to fill in)
            "review_status": "",
            "review_notes": "",
        })

    # Write CSV
    output_path = os.path.join(project_dir, "CallData", "flagged_calls.csv")
    fieldnames = [
        "call_id", "flags", "flag_count", "transcript_chars", "transcript_preview",
        "gold_appointment_booked", "gold_client_type", "gold_treatment_type", "gold_reason_not_booked",
        "review_status", "review_notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    print(f"Total calls scanned: {total}")
    print(f"Flagged calls: {len(flagged_rows)} ({len(flagged_rows)/total:.1%})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
