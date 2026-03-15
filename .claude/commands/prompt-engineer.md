---
name: prompt-engineer
description: Autonomous prompt engineering optimization loop with OPRO-style iteration, error analysis, gold label auditing, and architecture search.
---

# /prompt-engineer

You are an autonomous prompt engineering optimizer. You will systematically improve classification prompts through measured iteration — not guesswork.

## Setup

1. Look for `prompt-engineer.yaml` in the project root. If not found, ask the user for the config path.
2. Read the config and validate all required sections exist: `data`, `fields`, `models`, `budget`.
3. Set the working directory to the project root and ensure `Scripts/prompt_engineer/` exists with `runner.py`, `error_analysis.py`, `gold_label_audit.py`.

## State Management

Maintain state in `{output_dir}/state.json` with:
```json
{
  "current_phase": 0,
  "completed_phases": [],
  "field_status": {"field_name": {"best_accuracy": 0, "best_prompt_version": "v0", "rounds": 0}},
  "total_cost_usd": 0,
  "holdout_ids": [],
  "timestamp": "..."
}
```

Read state.json at the start of every phase to support resume. Write state.json after every phase completes.

---

## Phase 0: Init

**Entry:** Config file exists and is valid.
**Actions:**
1. Read the config YAML.
2. Verify data files exist at the configured paths.
3. Verify at least one model's API key env var is set (run: `python -c "import os; print(os.getenv('KEY_NAME'))"` for each).
4. Create output directory.
5. Generate holdout split IDs and save to state.
6. Log: project name, number of items, number of fields, available models.

**Exit:** state.json created with `current_phase: 0` in `completed_phases`.

---

## Phase 1: Baseline

**Entry:** Phase 0 complete.
**Actions:**
For each field, using the cheapest available model:

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase baseline \
    --field $FIELD_NAME \
    --model $CHEAPEST_MODEL \
    --batch-size $DEFAULT_BS \
    --max-calls $BASELINE_N \
    --output $OUTPUT_DIR/baseline_${FIELD_NAME}.json \
    --seed 42
```

Run independent fields (same priority, no dependencies) in **parallel subagents**. Run dependent fields sequentially after their dependencies complete.

After all fields complete:
1. Read each baseline result JSON.
2. Create a summary table: field | model | accuracy | target | status | cost.
3. Update state.json with baseline accuracies.
4. **Check budget** after each runner invocation. Stop if exceeded.

**Exit:** All fields have baseline scores. Summary table printed.

---

## Phase 2: Error Analysis

**Entry:** Phase 1 complete.
**Actions:**
For each field:

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase evaluate \
    --field $FIELD_NAME \
    --results $OUTPUT_DIR/baseline_${FIELD_NAME}.json
```

Then read the generated `error_report_${FIELD_NAME}.md` and summarize:
1. Error category breakdown (cascade, parent_sub, format, gold_suspect, genuine).
2. Top confusion pairs.
3. Fields ordered by "improvability" — genuine errors / total errors ratio.

This ordering determines which field to optimize first in Phase 4.

**Exit:** Error reports generated. Fields ranked by improvability.

---

## Phase 3: Gold Label Audit

**Entry:** Phase 2 complete.
**Actions:**

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase audit \
    --output $OUTPUT_DIR/flagged_items.csv
```

Read the generated `gold_label_audit.md`. Present to user:
1. Summary table of flagged items by severity.
2. All HIGH severity items (impossible states).
3. Sample of MEDIUM items (5-10 examples).
4. Ask: "Review `$OUTPUT_DIR/flagged_items.csv` and tell me which labels to update. I'll wait."

**⚠️ PAUSE HERE — wait for user response before continuing.**

If user provides corrected labels:
1. Note which IDs were corrected and what changed.
2. These corrections will be applied in future phases (pass corrected values to scoring).

**Exit:** Gold label review complete (user approved or skipped).

---

## Phase 4: OPRO Loop

**Entry:** Phase 3 complete. Fields ranked by improvability from Phase 2.
**Actions:**

Process fields in improvability order (most improvable first). For each field:

### Round N:

1. **Gather context for the optimizer:**
   - Current best prompt for this field
   - Current accuracy and target
   - Top 10 error patterns with examples (from error analysis)
   - Model reasoning on misclassified examples
   - History of previous prompt versions and their scores
   - Hardcoded learnings (see below)

2. **Generate candidate prompts** using the best available model (Claude/GPT-5/Gemini Pro):

   Use this optimizer meta-prompt to generate 2-3 candidates:

   ```
   You are an expert prompt engineer optimizing a classification prompt.

   CURRENT PROMPT:
   {current_prompt}

   CURRENT ACCURACY: {accuracy}% on {n} samples
   TARGET ACCURACY: {target}%

   ERROR PATTERNS:
   {opro_error_context}

   PREVIOUS ATTEMPTS (prompt version → accuracy):
   {history}

   HARD CONSTRAINTS — these are empirically validated findings:
   1. PREFER REMOVING RULES over adding them. Simpler prompts often outperform complex ones.
   2. PREFER EXAMPLES over rules. Show the model what correct classification looks like.
   3. DO NOT add confidence scores — they are useless (97-100 for both correct/incorrect).
   4. DO NOT add chain-of-thought unless the field genuinely needs reasoning.
   5. n=50 is unreliable — always confirm improvements at n=100+.
   6. Batch sizes affect accuracy differently per model. Don't assume bs=1 is best.
   7. Gold labels may be wrong — if >30% of errors look like gold label issues, flag for audit.
   8. Cascade errors from upstream fields set a ceiling — fixing the upstream field first may be more efficient.
   9. For hierarchical fields: the parent/sub boundary is the main source of errors.

   TASK: Propose 2-3 candidate prompt modifications. For each:
   - Describe the change in one sentence
   - Provide the full modified prompt
   - Estimate confidence (low/medium/high) that this improves accuracy
   - Explain your reasoning

   Prefer minimal changes. The best change is often removing a confusing rule,
   not adding a new one.
   ```

3. **Test each candidate** at iteration sample size:

   ```bash
   cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
       --config prompt-engineer.yaml \
       --phase baseline \
       --field $FIELD_NAME \
       --model $MODEL \
       --batch-size $BS \
       --max-calls $ITERATION_N \
       --output $OUTPUT_DIR/opro_${FIELD}_r${ROUND}_c${CANDIDATE}.json \
       --prompt $OUTPUT_DIR/candidate_prompt_${CANDIDATE}.txt \
       --seed $ROUND_SEED
   ```

   Use different seeds per round to avoid overfitting to a specific sample.

4. **Pick winner:** Highest accuracy among candidates. If winner beats current best by >= 1pp, adopt it.

5. **Confirm at larger N** (if gain is marginal, 1-3pp):
   Re-test winner at confirmation sample size to verify the gain is real.

6. **Check convergence:**
   - Record result in convergence detector.
   - If should_stop returns True, move to next field.
   - If budget exceeded, stop all optimization.

7. **Update state.json** after each round.

### After all fields optimized:

Print summary: field | baseline | optimized | delta | rounds | cost.

**Exit:** All fields either converged, hit target, or budget exhausted.

---

## Phase 5: Architecture Search

**Entry:** Phase 4 complete.
**Actions:**

Use **parallel subagents** to test simultaneously:

### 5a. Batch Size Sweep
For each field, test configured batch sizes with the current best prompt:

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase baseline \
    --field $FIELD_NAME \
    --model $MODEL \
    --batch-size $BS \
    --max-calls $ITERATION_N \
    --output $OUTPUT_DIR/arch_bs${BS}_${FIELD}.json
```

### 5b. Model Comparison
For each field, test across model tiers:

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase baseline \
    --field $FIELD_NAME \
    --model $MODEL_NAME \
    --batch-size $MODEL_DEFAULT_BS \
    --max-calls $ITERATION_N \
    --output $OUTPUT_DIR/arch_${MODEL}_${FIELD}.json
```

### 5c. Produce Architecture Recommendation

Create a comparison table:

| Field | Model | Batch Size | Accuracy | Cost/call | Monthly Cost (60K) |
|-------|-------|-----------|----------|-----------|-------------------|

Recommend the best architecture balancing accuracy and cost.

**Exit:** Architecture recommendation with cost projections.

---

## Phase 6: Final Validation

**Entry:** Phase 5 complete.
**Actions:**

Using the best prompt + model + batch size for each field, run on the holdout set:

```bash
cd $PROJECT_ROOT && python Scripts/prompt_engineer/runner.py \
    --config prompt-engineer.yaml \
    --phase baseline \
    --field $FIELD_NAME \
    --model $BEST_MODEL \
    --batch-size $BEST_BS \
    --max-calls 999999 \
    --output $OUTPUT_DIR/final_${FIELD}.json \
    --seed 99
```

(max-calls=999999 means "all items"; the holdout set filtering happens via seed/split logic)

Report final accuracies on holdout with confidence intervals (use binomial proportion CI).

**Exit:** Final validation results on unseen data.

---

## Phase 7: Report

**Entry:** Phase 6 complete.
**Actions:**

Generate `{output_dir}/REPORT.md` with these sections:

```markdown
# Prompt Engineering Report — {project_name}
Date: {date}

## Executive Summary
- Fields optimized: N
- Average accuracy improvement: baseline → final
- Total cost: $X.XX
- Recommended architecture: {model + batch size per field}

## Per-Field Results
### {field_name}
- Baseline: X% → Final: Y% (Δ +Z pp)
- Model: {model}, Batch size: {bs}
- Rounds: N, Converged: yes/no
- Best prompt: (inline or path)

## Architecture Comparison
{table from Phase 5}

## Error Analysis
{summary from Phase 2, updated with final numbers}

## Gold Label Issues
{summary from Phase 3}

## Cost Breakdown
{from BudgetTracker}

## Key Findings
{notable patterns, surprises, recommendations}

## Recommended Next Steps
{what to try next if accuracy targets not met}
```

**Exit:** REPORT.md written.

---

## Phase 8: Migration Proposal

**Entry:** Phase 7 complete.
**Actions:**

If the project has existing classification code (check for scripts matching `*Analyze*`, `*classify*`, `*transcripts*` in the project), propose a migration plan:

1. Identify which files need prompt updates.
2. Show the diff between current and optimized prompts.
3. Show the model/batch-size configuration changes.
4. Estimate cost impact (current vs. proposed).

Present to user: "Here's the migration plan. Shall I apply these changes?"

**⚠️ PAUSE HERE — wait for user approval before modifying any production code.**

If approved, apply changes to the identified files and commit with message:
`prompt engineering: apply optimized prompts from /prompt-engineer run`

**Exit:** Changes applied (or user declined).

---

## Budget Protocol

After EVERY `runner.py` invocation:
1. Read `{output_dir}/costs.json`
2. Check total against `budget.max_total_usd`
3. If exceeded: STOP immediately, report current results, skip remaining phases
4. If at warning threshold: log warning, continue but note in output

---

## Error Handling

- If `runner.py` fails: read stderr, diagnose, fix config or prompt, retry once.
- If API key missing: tell user which env var to set, stop.
- If data files not found: tell user exact path expected, stop.
- If accuracy drops after OPRO round: revert to previous best prompt, log the regression.
- If all candidates in a round are worse: try one "radical" change (completely restructure prompt), then accept convergence if that also fails.
