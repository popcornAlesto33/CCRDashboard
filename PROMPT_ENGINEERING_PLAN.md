# Prompt Engineering Improvement Plan - Script 03

## Current State

Script `Scripts/03_CallRail_Transcripts_Analyze_Buckets.py` uses a single system prompt to classify veterinary call transcripts via OpenAI (`gpt-4o-mini`). The prompt lists bucket labels but has no examples, no reasoning step, no disambiguation rules, and uses freeform JSON output.

## Architecture: Two-Model Pipeline

Split the current single-call approach into two distinct steps:

### Step 1: Reasoning (Stronger Model)
- **Model**: GPT-4o or GPT-5
- **Purpose**: Read raw transcript, produce a structured reasoning summary of what happened in the call
- **Batch size**: Start at 4, validate quality, then try 8. Tunable parameter.
- **Output**: Semi-structured reasoning per call (e.g., was an appointment made, caller intent, names mentioned, service discussed)
- **Why**: Transcript comprehension is the hard part — nuance, ambiguity, messy audio. A stronger model gives each transcript proper attention.

### Step 2: Classification (Faster Model + Structured Outputs)
- **Model**: GPT-4o-mini
- **Purpose**: Take the reasoning summary from Step 1 and slot it into the correct buckets
- **Batch size**: 8 (inputs are short, uniform reasoning summaries — easy to batch)
- **Output**: Strict JSON with enum-constrained fields via OpenAI structured outputs
- **Why**: Classification from a clean summary is mechanical. Mini handles it well, and structured outputs eliminate format errors.

### Benefits of the Split
- Iterate on reasoning and classification independently
- Debug which step caused a misclassification (store both reasoning and final output)
- Stronger model cost is amortized across batched transcripts (system prompt paid once per batch)
- Classification step is dirt cheap at scale

### Risks to Monitor
- **Context window pressure**: Long transcripts batched together may push toward limits — quality degrades near ceiling
- **Positional bias**: Middle items in a batch may get slightly less attention
- **Error propagation**: If reasoning misreads the transcript, the classifier can't recover — store reasoning output for auditing
- **Batch failures**: A single bad transcript can cause the whole batch to fail — all-or-nothing retry

## Data-Driven Findings (514 Labeled Examples)

Analysis of 514 expert-labeled call transcripts reveals the following patterns that inform all subsequent prompt decisions:

### Finding 1: Parent-Level Category Usage (43%)
43% of labels use parent-level categories (e.g., "Urgent Care / Sick Pet") instead of sub-categories. Experts intentionally used parent-level when transcripts lacked sufficient detail to determine a specific sub-category. The prompt must teach this pattern — the model must match the exact level of specificity, not over-specify.

### Finding 2: Client Type Distribution (Zero "Inconclusive")
All 514 examples are labeled as New (41%) or Existing (59%). No expert used "Inconclusive" for client type. Keep Inconclusive as an option but add guidance: default to Existing when the caller appears familiar with the hospital.

### Finding 3: Reason_not_booked Rule Change
14 expert examples populate `reason_not_booked` when `appointment_booked = "Inconclusive"`. Updated rule: populate when appointment_booked is "No" **or** "Inconclusive" (when a reason is apparent). Only set null when appointment_booked is "Yes".

### Finding 4: Unused Reason_not_booked Categories
6 categories have zero usage in 514 examples: "Meant to call low cost provider", "File Transferred", "Medication/food order", "Client/appt query", "Missed call", "No transcription". Keep in prompt with definitions — they may appear in future data.

### Finding 5: Treatment Type Distribution Skew
Top 3 treatment types cover 46% of all labels. Many sub-categories have only 1-4 examples. Few-shot example selection must cover both high-frequency and rare categories to avoid bias.

### Finding 6: Labeler Distribution
5 labelers contributed, with Adelaide responsible for 47% of labels. Inter-labeler consistency is unknown. Validation should watch for labeler-specific patterns.

## Planned Improvements

### 1. Few-shot examples (5-8 in reasoning prompt)
Select 5-8 examples from the 514 labeled transcripts covering:
- **High-frequency categories**: Urgent Care / Sick Pet (parent-level), Preventive Care – Vaccinations, Retail – Prescriptions
- **Parent vs. sub-category**: One case where parent-level is correct + one case where sub-category is correct (see Finding 1)
- **Disambiguation**: Emergency vs. Urgent Care, Dental (Surgical vs. Preventive), Price shopping (1a) vs. generic procrastination (1)
- **Edge cases**: End of Life, appointment_booked = "Inconclusive" with reason_not_booked populated
- **Note**: Full transcripts from `dbo.CallRailAPI` required for actual selection (calldata CSV has truncated transcripts ~30 chars)

### 2. Structured outputs with strict enums (classification prompt)
Replace `response_format={"type": "json_object"}` with a full JSON schema:
- `appointment_booked`: enum `["Yes", "No", "Inconclusive"]`
- `client_type`: enum `["New", "Existing", "Inconclusive"]`
- `treatment_type`: enum of all 33 values observed in labeled data + any additional categories from the prompt taxonomy (union coverage)
- `reason_not_booked`: enum of all 16 observed values + the 6 unused categories + null (union of labeled data + original prompt buckets)
- Eliminates typos, invented categories, partial labels, and malformed responses

### 3. Disambiguation rules for overlapping buckets
Explicit decision rules derived from the 514 labeled examples:

**Parent-level vs. sub-category (critical — affects 43% of labels):**
- Use the parent category (e.g., "Urgent Care / Sick Pet") when the transcript does not contain enough detail to determine a specific sub-category
- Only select a sub-category when the transcript explicitly mentions or clearly implies the specific service
- This is the single most important disambiguation rule

**Category boundary rules:**
- Urgent Care vs. Emergency & Critical Care
- Preventive Care vs. Surgical Services (dental, wellness)
- "Caller Procrastination - Price Objection" vs. "Meant to call low cost provider"

**Client type disambiguation:**
- Default to "Existing" when the caller appears familiar with the hospital
- Default to "New" when the caller asks introductory questions (hours, location, whether they accept new patients)
- Only use "Inconclusive" when there is genuinely no signal (zero expert examples used Inconclusive)

**Reason_not_booked rules:**
- Populate when `appointment_booked` is "No" or "Inconclusive" (when a reason is apparent)
- Only set null when `appointment_booked` is "Yes"

### 4. Bucket label definitions
Add 1-line definitions for buckets where the label alone is ambiguous (e.g., what counts as "Caller Procrastination" vs. a genuine scheduling conflict).

### 5. Transcript quality handling
Explicit rules for short, garbled, or non-English transcripts (e.g., "fewer than 3 exchanges = Inconclusive for all fields").

## Removed from Plan

- **~~Two-pass treatment type classification~~**: Unnecessary. With few-shot examples + structured output enums, the model handles 30+ options in a single pass. Adds complexity (two API calls, extra error handling) for marginal gain.
- **~~Batch size reduction to 1-2~~**: Not justified. The real accuracy wins come from prompt improvements, not batch size. The two-model architecture handles this better — reasoning batches are tunable, classification batches stay at 8.

## Next Steps

### What I need to provide
100-1000 expert-labeled transcript examples (human-verified, quality-checked) with the following fields per example:
- Transcript text (or call ID)
- `appointment_booked`
- `client_type`
- `treatment_type`
- `reason_not_booked`
- `stated_hospital_name`, `stated_patient_name`, `agent_name` (if available)

Format: CSV, JSON, or spreadsheet export all work.

### What happens once examples are provided
1. **Analyze the labeled examples** to identify edge cases, boundary patterns, and disambiguation rules
2. **Select 5-8 few-shot examples** for the reasoning prompt covering the trickiest scenarios
3. **Define the reasoning output format** — semi-structured summary that feeds the classifier
4. **Write the reasoning prompt** (Step 1) with examples, disambiguation rules, and quality handling
5. **Build the classification prompt + JSON schema** (Step 2) with strict enums for all bucket values
6. **Validate** by running the new two-step pipeline against the labeled examples and comparing to human labels
7. **Report accuracy per field** with defined success criteria and iterate on misclassifications

### Validation Methodology
- 514 labeled examples as gold standard
- Reserve 8 for few-shot, validate against remaining ~506
- **Exact match scoring** — model must match human label exactly, including parent-level vs. sub-category
- Target accuracy:
  - `appointment_booked`: 90%+
  - `client_type`: 90%+
  - `treatment_type`: 80%+ (33 categories, many rare)
  - `reason_not_booked`: 85%+ (when applicable)
- Use `analysis_version = "prod_v2"` for side-by-side comparison with existing results
- Review disagreements between human labels and model output — determine if label or model is correct

### Data Gap
Calldata CSV has truncated transcripts (~30 chars). Full transcripts needed from `dbo.CallRailAPI` for:
- Few-shot example selection
- Validation runs
- Transcript quality pattern analysis

### Priority order (highest impact first)
1. Two-model architecture (reasoning + classification split)
2. Few-shot examples in reasoning prompt
3. Structured outputs with strict enums in classification prompt
4. Disambiguation rules + bucket definitions
5. Transcript quality handling
