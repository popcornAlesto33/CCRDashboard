# CCR Dashboard - CallRail Data Pipeline

## Project Overview

Data pipeline for processing veterinary clinic phone call data from CallRail. The pipeline extracts call data, syncs labels from Google Sheets, analyzes transcripts with AI, anonymizes PII, and exports training data for fine-tuning.

## Script Pipeline (Numbered Execution Order)

| Script | Purpose | Source | Destination |
|--------|---------|--------|-------------|
| `01_UpdateCallsTranscripts` | ETL - fetch calls & transcripts | CallRail API | `dbo.CallRailAPI` |
| `02_Sync_AppSheetToTranscriptLabels` | Sync human labels | Google Sheets | `dbo.CallRailAPI_TranscriptLabels` |
| `03_Transcripts_Analyze_Buckets` | AI classification | `CallRailAPI` | `dbo.CallRailAPI_TranscriptAnalysis` |
| `04_AnonymizeCallRailTranscripts` | PII masking | `CallRailAPI` | `dbo.CallRailAPI_TranscriptAnonymized` |
| `05_ExportTrainingJSONL` | Fine-tuning export | `vw_CallRail_TrainingDataset` | `.jsonl` files |

## Environment Variables

```
# CallRail
CALLRAIL_API_KEY
CALLRAIL_ACCOUNT (or CallRail_Account, CALLRAIL_ACCOUNT_ID)

# SQL Server
SQLSERVER_SERVER (or SQL_HOST)
SQLSERVER_PORT (default: 1433)
SQLSERVER_DATABASE (or SQL_DATABASE)
SQLSERVER_UID (or SQL_USERNAME)
SQLSERVER_PWD (or SQL_PASSWORD)

# OpenAI (for script 03)
OPENAI_API_KEY
OPENAI_MODEL (default: gpt-4o-mini)

# Google Sheets (for script 02)
SPREADSHEET_ID
```

## Key Classification Buckets (Script 03)

The AI analysis classifies calls into:
- **appointment_booked**: Yes / No / Inconclusive
- **client_type**: New / Existing / Inconclusive
- **treatment_type**: 30+ veterinary service categories
- **reason_not_booked**: 15+ categories (only when not booked)
- **Extracted fields**: hospital name, patient name, agent name

## Improvement Tracking

### Performance Targets
- [ ] Optimize batch sizes for OpenAI calls
- [ ] Review SQL query efficiency
- [ ] Consider connection pooling

### Prompt Engineering (Script 03)
- [ ] Review bucket definitions for clarity
- [ ] Evaluate edge case handling
- [ ] Consider few-shot examples in prompt
- [ ] Review temperature and response format settings

### Code Quality
- [ ] Consistent error handling patterns
- [ ] Unified logging approach
- [ ] Configuration consolidation

## Working Principles

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

## Version Control

- **Git remote**: https://github.com/popcornAlesto33/CCRDashboard.git
- **Commit every prompt version**: After each prompt engineering iteration (v1, v2, etc.), commit the prompt changes and test results with a descriptive message before moving to the next version
- **Commit message format**: `prompt v{N}: {brief description of changes}` (e.g., `prompt v8: remove Inconclusive anchor, add decision tree`)
- **Push after commit**: Always push to origin after committing

## Notes

- Scripts written by external consultant
- Changes should be tactical and traceable (no structural overhauls for now)
- Focus areas: performance, prompt engineering
