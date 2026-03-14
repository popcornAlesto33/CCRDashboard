# Client Handover — Script 03 Prompt Engineering Update

## What Changed

Script `Scripts/03_CallRail_Transcripts_Analyze_Buckets.py` was rewritten from a single-model, single-prompt approach to a **two-model pipeline** with improved prompts. No other scripts were modified.

### Before (v1)
- Single model (gpt-4o-mini) does everything in one call
- No few-shot examples — prompt only listed bucket names
- No disambiguation rules — model guessed on ambiguous cases
- Freeform JSON output — model could invent categories or make typos
- Hardcoded to OpenAI
- `response_format={"type": "json_object"}` (no schema enforcement)

### After (v2)
- **Two-step pipeline**: a stronger model reasons about the transcript, then a faster model classifies from that reasoning
- **8 few-shot examples** in the reasoning prompt covering the trickiest classification patterns
- **Disambiguation rules** for the most common confusions (parent vs sub-category, emergency vs urgent care, price shopping vs general procrastination)
- **Strict JSON schema** with enum constraints — the model can only output valid bucket values
- **LLM-agnostic** — works with any OpenAI-compatible API (Gemini, OpenAI, Azure, etc.)
- **Updated reason_not_booked rule** — now populated for both "No" and "Inconclusive" (was only "No")

### New Files
| File | Purpose |
|------|---------|
| `Scripts/validate_prompt_engineering.py` | Runs the pipeline against 510 labeled examples and reports accuracy per field |
| `CallData/few_shot_examples.json` | The 8 curated few-shot examples with full transcripts and labels |

---

## .env Changes for Script 03

### Old .env variables (no longer used by script 03)
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### New .env variables (required)
```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
REASONING_MODEL=gemini-2.5-pro
CLASSIFICATION_MODEL=gemini-2.5-flash
```

> **Note:** The other scripts (01, 02, 04, 05) do not use these variables. They are unchanged.

---

## LLM-Agnostic Design

The script uses the OpenAI Python SDK pointed at any compatible API via `LLM_BASE_URL`. To switch providers, change three .env values:

### Gemini (current)
```
LLM_API_KEY=AIza...
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
REASONING_MODEL=gemini-2.5-pro
CLASSIFICATION_MODEL=gemini-2.5-flash
```

### OpenAI
```
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
REASONING_MODEL=gpt-4o
CLASSIFICATION_MODEL=gpt-4o-mini
```

### Azure OpenAI
```
LLM_API_KEY=your_azure_key
LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
REASONING_MODEL=your-gpt4o-deployment
CLASSIFICATION_MODEL=your-gpt4o-mini-deployment
```

No code changes needed — just update the .env and restart.

---

## How the Two-Model Pipeline Works

```
Transcript ──> [Step 1: Reasoning Model] ──> Structured reasoning summary
                                                        │
                                                        v
                                    [Step 2: Classification Model] ──> JSON with bucket values
```

**Step 1 (Reasoning)** — The stronger model reads the raw transcript and produces a reasoning summary: was an appointment made, what service was discussed, who is the client, why wasn't it booked, and what names were mentioned. This is the hard part — transcript comprehension with messy audio, ambiguity, and nuance.

**Step 2 (Classification)** — The faster model takes the clean reasoning summary and slots it into the correct bucket values. This is mechanical. The strict JSON schema with enum constraints means it can only output valid values.

**Why split it?** You can iterate on reasoning and classification independently. If a call is misclassified, you can check the reasoning output to see whether the reasoning model misunderstood the transcript or the classification model picked the wrong bucket.

### Batch sizes
- Reasoning: default 4 calls per batch (transcripts are long, need attention)
- Classification: default 8 per batch (reasoning summaries are short and uniform)

### Single-model fallback
Pass `--single-model` to use only the classification model with the improved prompt (no reasoning step). Useful for cost comparison or if the reasoning model is unavailable.

---

## Key Observations from the Data Analysis

These findings drove the prompt design decisions:

1. **43% of labels use parent-level categories** (e.g., "Urgent Care / Sick Pet" instead of a sub-category). Experts intentionally use the parent level when the transcript lacks detail. The prompt teaches this pattern explicitly.

2. **Zero "Inconclusive" client type in 514 expert labels.** All are New (41%) or Existing (59%). The prompt guides the model to default to Existing when the caller seems familiar with the hospital.

3. **14 expert labels populate reason_not_booked when appointment_booked is "Inconclusive."** The old rule only populated it for "No". The new rule: populate for "No" or "Inconclusive" when a reason is apparent.

4. **Top 3 treatment types cover 46% of labels.** Many sub-categories have only 1-4 examples. The few-shot examples cover both high-frequency and rare categories.

5. **6 reason_not_booked categories have zero usage** in 514 examples (e.g., "File Transferred", "Missed call"). Kept in the schema since they may appear in future data.

---

## Running Validation

The validation script runs the pipeline against the 510 labeled examples from `CallData/` and reports accuracy per field.

```bash
# Quick test (50 calls)
python Scripts/validate_prompt_engineering.py --max-calls 50

# Full validation (all 510)
python Scripts/validate_prompt_engineering.py

# Save detailed results with per-call predictions and reasoning
python Scripts/validate_prompt_engineering.py --output results.json

# Test single-model mode
python Scripts/validate_prompt_engineering.py --single-model
```

### Accuracy targets
| Field | Target |
|-------|--------|
| appointment_booked | 90%+ |
| client_type | 90%+ |
| treatment_type | 80%+ |
| reason_not_booked | 85%+ |

The output shows mismatches and a confusion matrix for treatment_type to guide prompt iteration.

---

## Analysis Version

The default `analysis_version` is now `prod_v2` (was `prod_v1`). This allows side-by-side comparison in the database between old and new pipeline results.
