# Design Principles — Call Transcript Classification

This document captures all design decisions made for the prompt engineering pipeline. These principles are the ground truth for how the model should behave. When in doubt, the human-labeled data is authoritative.

---

## 1. Human Labels Are Ground Truth

The 514 expert-labeled transcripts are the definitive reference. If the model disagrees with a human label, the model is wrong. All validation uses **exact match scoring** — the model must produce the same label the human chose, not a "better" one.

**Why:** The labelers are domain experts who understand veterinary call workflows. Their judgment on ambiguous calls reflects real-world classification needs.

---

## 2. Parent-Level Categories Are Intentional

Human labelers used parent-level categories (e.g., "Urgent Care / Sick Pet" instead of "Urgent Care – Diagnosis and Treatment of Illnesses") **43% of the time** (221 of 514 records). This is not laziness or error — it reflects that the transcript did not contain enough detail to sub-classify.

**Rule:** Use the parent category when the transcript does not contain enough detail to determine a specific sub-category. Only select a sub-category when the transcript explicitly mentions or clearly implies the specific service.

**Examples:**

| Scenario | Correct Label | Why |
|----------|--------------|-----|
| Caller says "my dog is sick, can I bring him in?" | Urgent Care / Sick Pet | No detail on what illness or treatment |
| Caller says "my dog has been vomiting for two days" | Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections) | Specific symptom mentioned that maps to a sub-category |
| Caller asks about "getting my cat's shots" | Preventive Care – Vaccinations | Explicitly about vaccinations |
| Caller asks about "a checkup for my new puppy" | Preventive Care | Could be vaccines, exam, or wellness — not enough to sub-classify |
| Caller asks about "surgery" generally | Surgical Services | No detail on what type of surgery |
| Caller asks about "getting my dog's teeth cleaned" | Surgical Services – Dental Care (Cleanings, Extractions) | Specific service mentioned |

**Distribution of parent-level usage in labeled data:**
- Urgent Care / Sick Pet: 105 records (20.4% of all labels)
- Preventive Care: 54 records (10.5%)
- Diagnostic Services: 29 records (5.6%)
- Emergency & Critical Care: 10 records (1.9%)
- Surgical Services: 10 records (1.9%)
- End of Life Care: 7 records (1.4%)
- Dermatology: 5 records (1.0%)
- Retail: 1 record (0.2%)

---

## 3. Reason Not Booked — Populate for "No" AND "Inconclusive"

Human labelers populated `reason_not_booked` for 14 calls where `appointment_booked` was "Inconclusive". This makes sense: even when the outcome is unclear, the reason the appointment wasn't definitively booked is still observable.

**Rule:**
- `appointment_booked = "Yes"` → `reason_not_booked = null`
- `appointment_booked = "No"` → `reason_not_booked = <select one>`
- `appointment_booked = "Inconclusive"` → `reason_not_booked = <select one if a reason is apparent, null if not>`

**Example:** A caller asks about dental cleaning prices, says "let me think about it," and hangs up without committing. The appointment isn't definitively not-booked (they might call back), but the reason is clearly price objection.
- `appointment_booked`: Inconclusive
- `reason_not_booked`: 1a. Caller Procrastination - Price Objection / Shopping / Request for Quote

---

## 4. Client Type — Binary in Practice

In 514 labeled examples, human labelers **never used "Inconclusive"** for `client_type`. Every call was classified as either New (212, 41%) or Existing (302, 59%).

**Rule:**
- **Existing**: Caller is familiar with the hospital, references past visits, has a file on record, or is recognized by the agent.
- **New**: Caller asks introductory questions (hours, location, whether they accept new patients), or the agent asks "have you been here before?" and the answer is no.
- **Inconclusive**: Keep as a valid option, but only use when there is genuinely zero signal in the transcript. This should be rare.

---

## 5. Two-Model Architecture

Classification is split into two API calls:

1. **Reasoning step** (GPT-4o / GPT-5): Reads the raw transcript and produces a structured summary of what happened in the call — who called, what they wanted, whether an appointment was made, names mentioned, and any observable reasons.

2. **Classification step** (GPT-4o-mini): Takes the reasoning summary and maps it to the correct buckets using structured outputs with strict enum constraints.

**Why:** Transcript comprehension is the hard part (messy audio, ambiguity, nuance). Classification from a clean summary is mechanical. Splitting these lets us use a stronger model where it matters and a cheaper model where it doesn't.

**Batch sizes:**
- Reasoning: Start at 4, tune up to 8 if quality holds. Full transcripts vary in length, so monitor context window pressure.
- Classification: Batch 8. Inputs are short, uniform reasoning summaries.

---

## 6. Exact Match Scoring

During validation, the model's output must **exactly match** the human label. There is no partial credit.

- If the human labeled "Urgent Care / Sick Pet" and the model outputs "Urgent Care – Diagnosis and Treatment of Illnesses," that is **incorrect**, even though the sub-category is within the parent.
- If the human labeled "1. Caller Procrastination" and the model outputs "1a. Caller Procrastination - Price Objection," that is **incorrect**.

**Why:** The labeled data reflects the appropriate level of specificity for each call. Over-specifying is as wrong as under-specifying.

**Target accuracy:**
- `appointment_booked`: 90%+
- `client_type`: 90%+
- `treatment_type`: 80%+ (33 categories, many rare)
- `reason_not_booked`: 85%+ (when applicable)

---

## 7. Unused Categories Are Still Valid

Six `reason_not_booked` categories have zero usage in the 514 labeled examples:
- 5. Meant to call low cost / free service provider
- 7. File Transferred
- 8. Medication/food order
- 9. Client/appt query (non-medical)
- 10. Missed call
- 11. No transcription

**Rule:** Keep these in the prompt and in the structured output enum. They may appear in future data. But do not use them in few-shot examples (no reference examples available).

---

## 8. Labeler Distribution

| Labeler | Records | % of Total |
|---------|---------|-----------|
| Adelaide | 240 | 46.7% |
| Dani | 104 | 20.2% |
| Justin | 79 | 15.4% |
| Mik | 74 | 14.4% |
| Caitlin | 4 | 0.8% |
| (missing) | 13 | 2.5% |

**Note:** Adelaide labeled nearly half the dataset. If there are systematic biases in her labeling approach, they will dominate the training signal. Inter-labeler agreement has not been formally measured. Future work should spot-check overlapping labels across labelers.

---

## 9. Data Gaps

- **Truncated transcripts in CSV**: The calldata CSV only has ~30 character transcript previews. Full transcripts must be pulled from `dbo.CallRailAPI` in the SQL database for few-shot selection and validation.
- **Patient name missing 41% of the time**: `stated_patient_name` is null in 210 of 514 records. This is expected — not all calls mention a pet by name.
- **Agent name missing 32% of the time**: `agent_name` is null in 165 records. Some agents don't introduce themselves.
- **Hospital name missing 8% of the time**: `stated_hospital_name` is null in 40 records.

---

## 10. Version Management

Use the existing `analysis_version` column to run new prompt versions side-by-side:
- Current production: `prod_v1`
- New two-model pipeline: `prod_v2`
- Compare results before cutting over

This infrastructure already exists in Script 03 — no code changes needed for A/B testing.
