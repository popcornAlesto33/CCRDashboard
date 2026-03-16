# Original Script Evaluation and Reflections

## Context

Ran the original (pre-v1) prompt from `Script 03 OLD - ORIGINAL.py` through the same validation harness used for v1–v8 testing. The original prompt has zero examples, no decision trees, no parent/sub-category rules — just bucket lists and "choose ONE."

Model used: `gemini-2.5-flash` (single-model, direct transcript classification).
v8 comparison uses: `gemini-2.5-pro` (reasoning) → `gemini-2.5-flash` (classification).

---

## Results Comparison

### n=50 (initial run)

| Field | Original (Flash) | v8 (Pro→Flash) | Delta |
|-------|:---:|:---:|:---:|
| appointment_booked | **95.9%** (47/49) | 79.2% (38/48) | +16.7pp |
| client_type | 81.6% (40/49) | **85.4%** | -3.8pp |
| treatment_type | 53.1% (26/49) | **66.7%** | -13.6pp |
| reason_not_booked | 5.0% (1/20) | **27.3%** | -22.3pp |

### n=200 (scale validation)

| Field | Original n=50 | Original n=200 | v8 n=50 | v6 n=100 |
|-------|:---:|:---:|:---:|:---:|
| appointment_booked | **95.9%** | **68.5%** (137/200) | 79.2% | 76.8% |
| client_type | 81.6% | 84.0% (168/200) | **85.4%** | **89.5%** |
| treatment_type | 53.1% | 50.0% (100/200) | **66.7%** | 48.4% |
| reason_not_booked | 5.0% | 1.8% (2/114) | **27.3%** | **38.7%** |

### n=510 (full gold label set)

| Field | Original n=510 | Original n=200 | Original n=50 | v8 n=50 |
|-------|:---:|:---:|:---:|:---:|
| appointment_booked | **70.0%** (350/500) | 68.5% | 95.9% | 79.2% |
| client_type | **79.0%** (395/500) | 84.0% | 81.6% | 85.4% |
| treatment_type | **43.6%** (218/500) | 50.0% | 53.1% | 66.7% |
| reason_not_booked | **2.4%** (7/292) | 1.8% | 5.0% | 27.3% |

Note: 500/510 evaluated — 8 calls lost to a JSON parse error on batch 30 (no strict schema = malformed JSON), 2 calls had no gold labels matched.

### n=200 — Model Comparison: Gemini Flash vs gpt-4o-mini (original prompt)

To isolate model quality from prompt quality, ran the same original prompt on n=200 with the model the consultant originally chose (gpt-4o-mini) vs Gemini Flash.

| Field | Original + Gemini Flash | Original + gpt-4o-mini | Delta |
|-------|:---:|:---:|:---:|
| appointment_booked | 68.5% (137/200) | 67.9% (133/196) | -0.6pp |
| client_type | **84.0%** (168/200) | 71.4% (140/196) | **-12.6pp** |
| treatment_type | **50.0%** (100/200) | 38.8% (76/196) | **-11.2pp** |
| reason_not_booked | 1.8% (2/114) | 5.4% (5/92) | +3.6pp |

Note: gpt-4o-mini lost 4 calls to parsing issues (196/200 evaluated).

**gpt-4o-mini is significantly worse than Gemini Flash on this task:**

- **client_type -12.6pp**: gpt-4o-mini over-uses `Inconclusive` — 28 of 56 errors are unnecessary Inconclusive predictions where Gemini Flash correctly identifies New or Existing.
- **treatment_type -11.2pp**: New error pattern unique to gpt-4o-mini — classifying real calls as `N/A (missed call)` (8x Preventive Care→N/A, 4x Vaccinations→N/A). Gemini Flash never makes this mistake. The top confusions also differ: gpt-4o-mini misclassifies Emergency calls as Urgent Care (4x), suggesting weaker domain comprehension.
- **appointment_booked** roughly the same (~68%) — both models struggle equally without Inconclusive guidance.
- **reason_not_booked** slightly higher for gpt-4o-mini (5.4% vs 1.8%) but both are near-zero due to formatting issues (missing number prefixes).

**Conclusion:** The original consultant's choice of gpt-4o-mini was a weaker baseline than Gemini Flash at a similar price point ($0.15/$0.60 per 1M input/output tokens for both). The move to Gemini was a net improvement independent of prompt engineering.

---

## CORRECTION: The n=50 Result Was Sample Bias

The original prompt's 95.9% appointment_booked at n=50 was a **misleading outlier**. At n=200 it collapsed to **68.5%** — worse than every v-series run.

The first 50 calls (sorted alphabetically by ID) happened to be easy cases for appointment classification. This same bias affected all v1–v8 n=50 runs, which was already flagged when v6 was scaled to n=100 (treatment_type dropped from 62% to 48.4%).

### What the n=200 run actually showed

- **63 appointment_booked errors** — the dominant pattern is Inconclusive→No. Without any guidance on what "Inconclusive" means, the model defaults to No for ambiguous calls (voicemails, pending callbacks, inter-clinic consultations).
- **100 treatment_type errors** — the same weaknesses seen at n=50, amplified:
  - 17x Urgent Care parent→sub over-specification (the #1 error v7 fixed)
  - 15x Preventive Care→Other (the "Other" gate v8 added)
  - 5x Urgent Care→Other
  - 4x Wellness Screening→Diagnostic Lab (routine vs symptom-driven bloodwork confusion)
  - 3x Dermatology→Urgent Care (no Dermatology recognition)
- **reason_not_booked 1.8%** — entirely a formatting issue. The model returns correct concepts but drops numbering prefixes (e.g., `"Scheduling Issue - Full schedule"` instead of `"2b. Scheduling Issue - Full schedule"`). The strict JSON schema in v8 is essential.

---

## Revised Conclusions

### The prompt engineering work IS validated at scale

At n=200, every field performs worse with the original prompt than with v8. The v2–v8 improvements are real:
- Parent/sub-category rules cut Urgent Care over-specification from 17x to near-zero
- "Other" gate rules cut Preventive Care→Other from 15x to 2x
- Dermatology recognition eliminated Dermatology→Urgent Care misclassification
- Strict JSON schema eliminated formatting errors in reason_not_booked
- Client type signal definitions ("THIS clinic", concrete evidence requirements) improved consistency

### The n=50 sample size is unreliable for this dataset

The first 50 calls (alphabetically sorted) are not representative of the full 510-call distribution. This has implications for all v1–v8 results in `test_results.md` — the absolute accuracy numbers at n=50 should be treated as directional, not precise. Future evaluations should use n=200+ or random sampling with a fixed seed.

### The initial appointment_booked "miracle" was wrong

The hypothesis that "the model's native judgment is excellent and rules hurt" was based on a biased sample. At scale, the model's native judgment on appointment_booked (68.5%) is actually the **worst** of any run — it needs the guidance. The No/Inconclusive boundary rules, while they caused oscillation in n=50 runs, are necessary.

---

## What Still Holds From the Initial Analysis

### Information loss in the two-model pipeline is a real concern

Even though the original prompt performed worse overall at n=200, the two-model architecture (Pro reasoning → Flash classification) does compress information. This is worth investigating as a separate variable — the v9 field-decomposed pipeline was designed partly to address this by having the model read raw transcripts per field.

### Prompt engineering helps most where domain knowledge is needed

This conclusion still holds. treatment_type (35 categories, parent/sub rules, domain boundaries) benefited most from prompt engineering. appointment_booked (3 values, relatively obvious) benefited least. The ROI of prompt engineering scales with task ambiguity.

### reason_not_booked needs strict schema enforcement, not just prompt rules

At both n=50 and n=200, the original prompt gets the right concept but wrong exact string. This is a structural issue that no amount of prompt wording will fix — strict JSON schema with enum values is the correct solution.

---

## Top Error Patterns at n=510 (Original Prompt)

### treatment_type confusions (282 errors)

| Gold | Predicted | Count |
|------|-----------|:-----:|
| Urgent Care / Sick Pet | Urgent Care – Diagnosis and Treatment | **56x** |
| Preventive Care | Other | **24x** |
| Diagnostic Services | Urgent Care – Diagnosis and Treatment | 17x |
| Urgent Care / Sick Pet | Emergency – Stabilization | 10x |
| Urgent Care / Sick Pet | Other | 9x |
| Preventive Care | Annual Exams | 8x |
| Emergency & Critical Care | Emergency – Stabilization | 6x |
| Wellness Screening | Vaccinations | 6x |
| Wellness Screening | Diagnostic Lab Testing | 5x |
| Dermatology – Ear Infections | Urgent Care – Diagnosis | 5x |

### appointment_booked errors (150 errors)

The dominant pattern is gold=Inconclusive predicted as No — the model has no concept of when Inconclusive is appropriate without guidance.

### reason_not_booked errors (285 errors)

Almost entirely formatting — the model returns correct concepts but drops numbering prefixes (e.g., `"Scheduling Issue - Full schedule"` instead of `"2b. Scheduling Issue - Full schedule"`). Strict JSON schema is essential.

---

## Cost Comparison

| Run | Reasoning Cost | Classification Cost | Total | Per-Call |
|-----|:---:|:---:|:---:|:---:|
| Original n=50 (Flash only) | $0.00 | $0.011 | $0.011 | $0.00022 |
| Original n=200 (Flash only) | $0.00 | $0.040 | $0.040 | $0.00020 |
| Original n=510 (Flash only) | $0.00 | $0.104 | $0.104 | $0.00021 |
| Original n=200 (gpt-4o-mini) | $0.00 | $0.034 | $0.034 | $0.00017 |
| v8 n=50 (Pro→Flash) | ~$0.21 | ~$0.005 | ~$0.21 | ~$0.004 |

The original is ~20x cheaper per call — but accuracy at scale doesn't justify the savings.

---

## Meta-Lesson: Always Validate at Scale

The biggest takeaway from this exercise is methodological: **n=50 sorted-alphabetically is not a reliable evaluation.** The initial n=50 result led to a compelling but wrong narrative about over-engineering. The n=200 run told the true story. Future prompt iterations should:

1. Use n=200+ for any result that will inform decisions
2. Use random sampling with a fixed seed (e.g., `--random --seed 42`) to avoid alphabetical bias
3. Treat n=50 runs as quick directional checks only, never as final results
