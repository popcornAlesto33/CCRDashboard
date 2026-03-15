#!/usr/bin/env python3
"""
Systematic gold label audit using transcript analysis + model predictions.
Flags calls where gold labels are likely wrong or inconsistent.
No LLM calls — pure data analysis on existing results + transcript patterns.
"""
import os, csv, re, json
from collections import Counter, defaultdict

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


def analyze_transcript(t):
    """Extract features from a transcript."""
    turns = len(re.findall(r'(Agent:|Caller:)', t))
    caller_parts = re.findall(r'Caller:\s*(.*?)(?=Agent:|$)', t, re.DOTALL)
    caller_words = sum(len(p.split()) for p in caller_parts)
    total_words = len(t.split())
    redacted = len(re.findall(r'\[(MEDICAL_CONDITION|DRUG|MEDICAL_PROCESS)\]', t))

    medical_terms = re.findall(r'(sick|vomit|limp|pain|bleed|itch|sneez|cough|ear.{0,5}infect|allergy|diarrhea|seiz|surgery|spay|neuter|dental|vaccin|shot|bloodwork|x-ray|ultrasound|emergency|euthan|lump|swollen|limping|not eating|not drinking|discharge)', t, re.IGNORECASE)

    features = {
        'turns': turns,
        'caller_words': caller_words,
        'total_words': total_words,
        'chars': len(t),
        'n_medical': len(medical_terms),
        'medical_terms': list(set(m.lower() for m in medical_terms)),
        'is_voicemail': caller_words < 5,
        'is_very_short': turns < 3 or len(t) < 100,
        'is_wrong_number': bool(re.search(r'wrong number|called the wrong', t, re.IGNORECASE)),
        'is_reschedule': bool(re.search(r'reschedule|move.{1,15}appointment|cancel.{1,15}appointment|change.{1,15}appointment', t, re.IGNORECASE)),
        'is_admin': bool(re.search(r'reschedule|cancel|pick.?up|results|records|transfer|file', t, re.IGNORECASE)),
        'mentions_vaccine': bool(re.search(r'vaccin|shot|booster|rabies|distemper|parvo', t, re.IGNORECASE)),
        'mentions_annual': bool(re.search(r'annual|yearly|year.{0,5}(checkup|exam|visit)', t, re.IGNORECASE)),
        'mentions_bloodwork': bool(re.search(r'blood.?work|blood test|lab.?work|urinalysis|fecal', t, re.IGNORECASE)),
        'mentions_surgery': bool(re.search(r'surg|spay|neuter|dental clean|extract|lump remov', t, re.IGNORECASE)),
        'mentions_emergency': bool(re.search(r'emergency|poison|seiz|trauma|can.t walk|collaps', t, re.IGNORECASE)),
    }
    return features


def main():
    transcripts, labels = load_data()
    call_ids = sorted(set(transcripts.keys()) & set(labels.keys()))

    print(f"Auditing {len(call_ids)} calls")
    print(f"{'='*80}")

    issues = []

    for cid in call_ids:
        t = transcripts[cid]
        l = labels[cid]
        f = analyze_transcript(t)

        tt = l.get('treatment_type', '').strip()
        ab = l.get('appointment_booked', '').strip()
        ct = l.get('client_type', '').strip()
        rnb = l.get('reason_not_booked', '').strip()
        tt_parent = tt.split(' –')[0].split(' \u2013')[0]
        is_sub = ' – ' in tt or ' \u2013 ' in tt

        # ============================================================
        # AUDIT RULE 1: Voicemail/very short with specific gold labels
        # ============================================================
        if f['is_voicemail'] or f['is_very_short']:
            if ab == 'Yes' or (tt != 'N/A (missed call)' and tt != 'Other' and f['is_voicemail']):
                issues.append({
                    'cid': cid, 'rule': 'VOICEMAIL_WITH_SPECIFIC_LABELS',
                    'detail': f'ab={ab}, tt={tt}, turns={f["turns"]}, caller_words={f["caller_words"]}',
                    'severity': 'HIGH',
                })

        # ============================================================
        # AUDIT RULE 2: Wrong number with medical treatment type
        # ============================================================
        if f['is_wrong_number']:
            if tt not in ('Other', 'N/A (missed call)'):
                issues.append({
                    'cid': cid, 'rule': 'WRONG_NUMBER_WITH_TREATMENT',
                    'detail': f'tt={tt}',
                    'severity': 'HIGH',
                })

        # ============================================================
        # AUDIT RULE 3: Admin/rescheduling with specific sub-category
        # ============================================================
        if f['is_reschedule'] and f['n_medical'] == 0 and is_sub:
            issues.append({
                'cid': cid, 'rule': 'ADMIN_WITH_SPECIFIC_SUB',
                'detail': f'tt={tt}, is_reschedule=True, n_medical=0',
                'severity': 'MEDIUM',
            })

        # ============================================================
        # AUDIT RULE 4: No medical content but specific treatment type
        # ============================================================
        if f['n_medical'] == 0 and f['turns'] >= 4 and is_sub and not f['is_reschedule']:
            issues.append({
                'cid': cid, 'rule': 'NO_MEDICAL_WITH_SPECIFIC_SUB',
                'detail': f'tt={tt}, n_medical=0, turns={f["turns"]}',
                'severity': 'MEDIUM',
            })

        # ============================================================
        # AUDIT RULE 5: Preventive Care sub but no matching content
        # ============================================================
        if tt_parent == 'Preventive Care' and is_sub:
            if 'Vaccinations' in tt and not f['mentions_vaccine']:
                issues.append({
                    'cid': cid, 'rule': 'VACCINATION_LABEL_NO_VACCINE_MENTION',
                    'detail': f'tt={tt}, medical_terms={f["medical_terms"][:5]}',
                    'severity': 'LOW',
                })
            if 'Annual Exams' in tt and not f['mentions_annual']:
                issues.append({
                    'cid': cid, 'rule': 'ANNUAL_EXAM_LABEL_NO_ANNUAL_MENTION',
                    'detail': f'tt={tt}',
                    'severity': 'LOW',
                })
            if 'Wellness Screening' in tt and not f['mentions_bloodwork']:
                issues.append({
                    'cid': cid, 'rule': 'WELLNESS_SCREENING_NO_BLOODWORK_MENTION',
                    'detail': f'tt={tt}',
                    'severity': 'LOW',
                })

        # ============================================================
        # AUDIT RULE 6: appointment_booked=Yes but reason_not_booked populated
        # ============================================================
        if ab == 'Yes' and rnb and rnb.lower() not in ('null', 'none', ''):
            issues.append({
                'cid': cid, 'rule': 'YES_WITH_REASON',
                'detail': f'ab=Yes, rnb={rnb}',
                'severity': 'HIGH',
            })

        # ============================================================
        # AUDIT RULE 7: appointment_booked=No but no reason
        # ============================================================
        if ab == 'No' and (not rnb or rnb.lower() in ('null', 'none', '')):
            issues.append({
                'cid': cid, 'rule': 'NO_WITHOUT_REASON',
                'detail': f'ab=No, rnb=(empty)',
                'severity': 'MEDIUM',
            })

        # ============================================================
        # AUDIT RULE 8: Emergency treatment but no emergency keywords
        # ============================================================
        if tt_parent == 'Emergency & Critical Care' and not f['mentions_emergency']:
            if f['turns'] >= 4:  # Not a voicemail
                issues.append({
                    'cid': cid, 'rule': 'EMERGENCY_LABEL_NO_EMERGENCY_KEYWORDS',
                    'detail': f'tt={tt}, medical_terms={f["medical_terms"][:5]}',
                    'severity': 'LOW',
                })

        # ============================================================
        # AUDIT RULE 9: Surgical treatment but no surgery keywords
        # ============================================================
        if tt_parent == 'Surgical Services' and not f['mentions_surgery']:
            if f['turns'] >= 4:
                issues.append({
                    'cid': cid, 'rule': 'SURGERY_LABEL_NO_SURGERY_KEYWORDS',
                    'detail': f'tt={tt}, medical_terms={f["medical_terms"][:5]}',
                    'severity': 'LOW',
                })

    # ============================================================
    # REPORT
    # ============================================================

    print(f"\nTotal issues found: {len(issues)}")
    print()

    by_severity = defaultdict(list)
    for issue in issues:
        by_severity[issue['severity']].append(issue)

    by_rule = Counter(i['rule'] for i in issues)
    print("Issues by rule:")
    for rule, count in by_rule.most_common():
        print(f"  {rule}: {count}")
    print()

    for severity in ['HIGH', 'MEDIUM', 'LOW']:
        sev_issues = by_severity[severity]
        if not sev_issues:
            continue
        print(f"\n{'='*80}")
        print(f"  {severity} SEVERITY ({len(sev_issues)} issues)")
        print(f"{'='*80}")

        for issue in sev_issues:
            cid = issue['cid']
            t = transcripts[cid]
            l = labels[cid]
            print(f"\n  {cid}")
            print(f"    Rule: {issue['rule']}")
            print(f"    Detail: {issue['detail']}")
            print(f"    Gold: ab={l.get('appointment_booked','')}, ct={l.get('client_type','')}")
            print(f"           tt={l.get('treatment_type','')}")
            print(f"           rnb={l.get('reason_not_booked','')}")
            print(f"    Transcript ({len(t)} chars): {t[:150]}...")

    # Summary for CSV export
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {len(by_severity['HIGH'])} HIGH, {len(by_severity['MEDIUM'])} MEDIUM, {len(by_severity['LOW'])} LOW")
    print(f"  Unique calls flagged: {len(set(i['cid'] for i in issues))}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
