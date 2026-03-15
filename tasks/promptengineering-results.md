# Prompt Engineering Results: CCR Dashboard Call Classification

**Date:** 2026-03-14 to 2026-03-15
**Dataset:** 510 labeled veterinary clinic call transcripts
**All accuracy numbers:** n=200 unless noted

---

## Final Architecture Comparison

| Option | Description | appt_booked | client_type | treat_type | reason | Avg Accuracy | Cost (60K) | Per Call |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Orig** | gpt-4o-mini single-call, original prompts, no strict schema, bs=8 | 67.9% | 71.4% | 38.8% | 5.4% | **45.9%** | **$34** | $0.0006 |
| **A0** | Flash single-call, v9 prompts + strict schema, bs=15 | 79.4% | 87.4% | 61.3% | 38.0% | **66.5%** | **$34** | $0.0006 |
| **A0b** | Flash single-call, v9 v2 prompts (cross-field reasoning + improved reason), bs=15 | 79.5% | 91.0% | 62.0% | 42.3% | **68.7%** | **$34** | $0.0006 |
| **A1** | Flash 2-call: appt separate + 3 fields combined, bs=15 | 85.0% | 91.5% | 55.5% | 38.6% | **67.7%** | **$68** | $0.0011 |
| **A1b** | Flash 2-call: appt separate + 3 fields with appt context passed, bs=15 | 86.0% | 93.0% | 57.5% | 47.7% | **71.1%** | **$68** | $0.0011 |
| **A** | Flash 4-call field-decomposed, bs=15 | 85.0% | 91.5% | 53.0% | 40.6% | **67.5%** | **$115** | $0.0019 |
| **B** | Flash bs=15 appt+client. GPT-4o bs=5 treat+reason | 85.0% | 91.5% | 62.3% | 40.5% | **69.8%** | **$838** | $0.0140 |
| **D** | Flash bs=15 appt+client. Gemini Pro bs=5 treat+reason | 85.0% | 91.5% | 66.1% | 42.3% | **71.2%** | **$534** | $0.0089 |
| **C** | Flash bs=15 appt+client. GPT-5 bs=5 treat+reason | 85.0% | 91.5% | 68.5% | 45.9% | **72.7%** | **$2,984** | $0.0497 |

---

## Key Findings

### 1. Prompt engineering delivered +20.6pp at zero additional cost (Orig → A0)

The single biggest improvement came from better prompts + strict schema + model upgrade (gpt-4o-mini → Flash), all at the same $34/60K cost. This was purely prompt engineering value — no architecture changes needed.

### 2. Field decomposition helps appointment_booked but hurts treatment_type

| Architecture | appointment_booked | treatment_type |
|:---|:---:|:---:|
| Single-call (A0b) | 79.5% | **62.0%** |
| 2-call with context (A1b) | **86.0%** | 57.5% |
| 4-call decomposed (A) | **85.0%** | 53.0% |

appointment_booked benefits from a dedicated call (+6.5pp). treatment_type benefits from cross-field context — when the model considers all fields together, it reasons better about the service type. Fully decomposing into 4 calls removes this context and hurts treatment_type by 9pp.

### 3. Passing context between calls recovers most of the loss

A1b passes the appointment_booked result into the 3-field call. This recovered reason_not_booked (+9pp over A1) and partially recovered treatment_type (+2pp over A1). Cross-field context matters.

### 4. More expensive models help treatment_type, not much else

| Model for treat+reason | treatment_type | reason | Extra cost over Flash |
|:---|:---:|:---:|:---:|
| Flash | 53-62% | 38-42% | $0 |
| GPT-4o | 62.3% | 40.5% | +$770 |
| Gemini Pro | 66.1% | 42.3% | +$466 |
| GPT-5 | 68.5% | 45.9% | +$2,916 |

The reasoning models (Pro, GPT-5) improve treatment_type by 4-16pp over Flash. But reason_not_booked gains are modest (+2-8pp). appointment_booked and client_type don't benefit from stronger models at all — Flash is sufficient.

### 5. Flash is batch-resistant, other models are not

| Model | bs=1 | bs=5 | bs=10 | bs=15 | bs=20 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Flash (treat) | 50.5% | 60.0%* | 52.0% | 53.0% | 53.5% |
| GPT-4o-mini (treat) | 62.0% | 48.0% | — | — | — |
| GPT-4o (treat) | 62.0% | 59.0% | — | — | — |
| GPT-5 (treat) | 64.0% | 64.0% | 60.0% | — | — |

*n=50 only

Flash maintains accuracy at high batch sizes (up to bs=15), making it ideal for production throughput. GPT-4o-mini degrades badly with batching (-14pp). GPT-5 is stable at bs=5 but degrades at bs=10.

### 6. reason_not_booked accuracy is bottlenecked by appointment_booked (cascade)

~30% of reason_not_booked errors cascade from wrong appointment_booked decisions. When Flash misclassifies a "No" call as "Inconclusive", reason gets skipped (null) instead of being classified. Fixing appointment_booked accuracy is the biggest lever for improving reason.

### 7. treatment_type has a structural ceiling at ~65% due to gold label issues

97 calls (19% of dataset) have specific sub-category gold labels but no detectable medical keywords in the transcript. These represent calls where the gold label was likely assigned from CRM/appointment system data, not transcript content. The model cannot derive these labels from the transcript alone.

### 8. "Less rules, more trust" outperforms rule-heavy prompts

The v8 pipeline had ~3000 tokens of rules, decision trees, and edge cases. Stripping these and using minimal guidance with calibrating examples improved treatment_type from 52% to 70% (Pro). Every rule that fixed one error pattern broke another. The model's native judgment is often better than engineered rules.

### 9. Two-step classification (parent → sub) didn't help

Tested with Flash, GPT-5, and Flash→GPT-5 hybrid. All performed worse than single-step because errors compound across steps. The model reasons better about parent AND sub simultaneously than in sequence.

### 10. LLM confidence scores are useless for this task

Model self-reported 97-100 confidence for both correct and incorrect predictions. Zero signal for identifying which predictions to review. Transcript quality flags (keyword analysis) are more effective for routing human review.

---

## Recommended Production Architectures

### Best value: Option A1b ($68 for 60K, 71.1% avg)

```
Call 1: Flash bs=15 → appointment_booked only
Call 2: Flash bs=15 → client_type + treatment_type + reason_not_booked
        (receives appointment_booked result as context)
```

- 2 API calls per batch of 15 transcripts
- 60K transcripts: ~8,000 API calls total on Gemini
- No rate limit issues
- Best average accuracy of any Flash-only option

### Best accuracy: Option C ($2,984 for 60K, 72.7% avg)

```
Flash bs=15 → appointment_booked + client_type (parallel)
GPT-5 bs=5  → treatment_type + reason_not_booked (parallel)
```

- 4 fields across 2 providers
- 60K transcripts: ~8,000 Gemini + ~16,000 OpenAI calls
- +1.6pp over A1b for 44x the cost

### Best cost: Option A0b ($34 for 60K, 68.7% avg)

```
Flash bs=15 → all 4 fields in a single call
```

- 1 API call per batch of 15 transcripts
- 60K transcripts: ~4,000 API calls total
- Same cost as the original script with +22.8pp accuracy
- Weakness: appointment_booked at 79.5% (vs 86% in A1b)

---

## Transcript Flagging for Human Review

16.5% of calls (84/510) flagged based on transcript quality signals. Flagged calls are 2-3x more likely to have prediction errors.

| Flag | Count | Description |
|:---|:---:|:---|
| no_medical_content | 71 (13.9%) | No medical keywords — model guessing treatment_type |
| garbled | 5 (1.0%) | Heavy redaction tags |
| very_short | 5 (1.0%) | <3 turns or <100 chars |
| admin_no_medical | 4 (0.8%) | Rescheduling with no medical context |
| wrong_number | 2 (0.4%) | Caller dialed wrong clinic |
| voicemail | 1 (0.2%) | No caller interaction |

Production workflow: model classifies all calls → flagged calls go to human review queue → unflagged calls auto-accepted.

---

## Gold Label Audit

### Applied fixes (2026-03-14): 8 field-level changes across 5 calls

Voicemail/automated messages, wrong number calls, rescheduling calls with CRM-derived labels that can't be inferred from transcript.

### Systematic audit (2026-03-15): 181 issues across 510 calls

- 4 HIGH: wrong numbers with treatment types, appointment=Yes + reason populated
- 119 MEDIUM: 97 sub-category labels with no medical keywords, 21 No without reason
- 58 LOW: label/keyword mismatches

---

## What Would Move the Needle Further

| Lever | Expected Impact | Effort | Cost |
|:---|:---:|:---:|:---:|
| Gold label cleanup (97 flagged calls) | +3-5pp treatment | Medium | Free |
| Taxonomy simplification (merge confusing subs) | +5-10pp treatment | High | Free |
| Fine-tune Flash on 510 gold labels | +10-15pp all fields | High | ~$50 |
| More gold labeled data (1000+ examples) | +5-10pp long-term | High | Labor |
| Try Claude Sonnet for treatment_type | +2-5pp treatment | Low | ~$800/60K |

---

## Files

| File | Purpose |
|:---|:---|
| `Scripts/validate_prompt_engineering.py` | V9 pipeline: prompts, schemas, batching, flagging |
| `Scripts/test_hybrid.py` | End-to-end Flash+GPT-5 hybrid test |
| `Scripts/test_batching.py` | Batch size comparison tool |
| `Scripts/test_appointment_booked.py` | Per-field prompt variant testing |
| `Scripts/test_two_step_treatment.py` | Two-step treatment_type testing (Flash) |
| `Scripts/test_two_step_hybrid.py` | Flash→GPT-5 two-step testing |
| `Scripts/audit_gold_labels.py` | Systematic gold label audit |
| `Scripts/generate_flagged_calls.py` | Generate flagged calls CSV for human review |
| `CallData/flagged_calls.csv` | 84 flagged calls ready for human review |
| `tasks/gold_label_audit.md` | Gold label audit trail with decisions |
| `tasks/gold_label_review_97.md` | Detailed review file for 97 flagged calls |
| `docs/superpowers/specs/2026-03-14-prompt-v9-pipeline-redesign.md` | Original design spec |
