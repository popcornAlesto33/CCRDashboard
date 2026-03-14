# CallRail Data Pipeline - Script Documentation

## High-Level Summary

This pipeline processes phone call data from your veterinary clinics. It works like an assembly line:

```
Phone Calls → Get Data → Add Labels → AI Analysis → Remove Personal Info → Training Data
   (CallRail)     ↓           ↓            ↓               ↓                  ↓
              Script 01   Script 02    Script 03       Script 04          Script 05
```

**What each script does in plain terms:**

| Script | One-Line Summary |
|--------|------------------|
| 01 | Downloads call records and transcripts from CallRail |
| 02 | Imports human-reviewed labels from Google Sheets |
| 03 | Uses AI to automatically classify what each call was about |
| 04 | Removes personal information (names, phone numbers) from transcripts |
| 05 | Packages labeled data to train a custom AI model |

---

## Script 01: Update Calls & Transcripts

**File:** `01_CallRail_UpdateCallsTranscripts.py`

### What It Does

This script connects to CallRail (your call tracking service) and downloads all the call information into your database. Think of it as copying records from one filing cabinet (CallRail) to another (your database).

### Information Collected

For each phone call, the script saves:
- **Call details**: When the call happened, how long it lasted, was it answered
- **Phone numbers**: The customer's number, tracking number, and business number
- **Transcript**: The written version of what was said during the call
- **Marketing info**: How the caller found you (Google, website, referral, etc.)
- **Sentiment**: Whether the call seemed positive or negative

### How It Works

1. Connects to CallRail using your account credentials
2. Asks for all calls within a date range (default: last 90 days)
3. Downloads the calls in batches of 100 at a time
4. Cleans up the data (formats phone numbers consistently, organizes marketing info)
5. Saves everything to your SQL database
6. If a call already exists, it updates the record instead of creating a duplicate

### When It Runs

Typically scheduled to run daily (e.g., every night) to keep your database up to date with the latest calls.

---

## Script 02: Sync Labels from Google Sheets

**File:** `02_CallRail_Sync_AppSheetToTranscriptLabels.py`

### What It Does

Your team reviews calls and adds labels in a Google Sheet (like "appointment booked" or "new client"). This script copies those labels from the Google Sheet into your database so they can be used alongside the call data.

### Why This Matters

Human-reviewed labels are the "ground truth" for training AI. When a person listens to a call and marks it as "appointment booked = Yes", that becomes a verified example the AI can learn from.

### How It Works

1. Connects to the Google Sheet using a service account (like a robot with permission to read the sheet)
2. Downloads all rows from the "labels" worksheet
3. Clears the old labels in the database
4. Inserts all the fresh labels from the sheet
5. Skips any rows that don't have a call ID (incomplete entries)
6. If the same call ID appears twice in the sheet, keeps only the last one

### Important Notes

- This is a **full replacement** - every time it runs, it completely refreshes the labels
- Changes in the Google Sheet won't appear in reports until this script runs
- Labels include: hospital name, appointment status, client type, treatment type, agent name, etc.

---

## Script 03: AI Transcript Analysis

**File:** `03_CallRail_Transcripts_Analyze_Buckets.py`

### What It Does

This is the "brain" of the pipeline. It reads call transcripts and uses AI (OpenAI/ChatGPT) to automatically figure out:
- Was an appointment booked?
- Is this a new or existing client?
- What type of treatment were they calling about?
- If no appointment was booked, why not?

### Why This Matters

Manually reviewing thousands of calls is impossible. This script can analyze hundreds of calls per hour, giving you insights into:
- How many calls convert to appointments
- What services people are calling about most
- Why callers don't book (price concerns? scheduling issues? services not offered?)

### The Categories It Uses

**Appointment Booked:**
- Yes / No / Inconclusive

**Client Type:**
- New / Existing / Inconclusive

**Treatment Type (30+ categories):**
- Preventive Care (vaccinations, annual exams, wellness screening)
- Urgent Care (sick pet, chronic disease management)
- Surgical Services (spay/neuter, dental, soft tissue, orthopedic)
- Diagnostic Services (x-rays, ultrasound, lab work)
- Emergency & Critical Care
- Dermatology (allergies, ear infections)
- Retail (food orders, prescriptions)
- End of Life Care
- And more...

**Reasons Not Booked (if applicable):**
- Caller procrastination (price shopping, checking with partner)
- Scheduling issues (no same-day, full schedule, not open evenings/weekends)
- Service not offered (grooming, exotics, large animals)
- Emergency care not offered
- Meant to call different hospital
- And more...

### How It Works

1. Finds calls that have transcripts but haven't been analyzed yet
2. Groups calls into batches (default: 8 calls per batch)
3. Sends each batch to OpenAI with instructions on how to classify
4. Receives the AI's analysis back as structured data
5. Saves the results to the database
6. Processes multiple batches at the same time for speed (default: 4 simultaneous)

### Configuration Options

- `--max-calls`: Limit how many calls to process in one run
- `--batch-size`: How many calls to send to AI at once
- `--dry-run`: Preview what would be processed without actually doing it

---

## Script 04: Anonymize Transcripts

**File:** `04_CallRail_AnonymizeCallRailTranscripts.py`

### What It Does

Before using transcripts for training AI or sharing reports, this script removes personal information to protect customer privacy. It replaces sensitive data with placeholder tags.

### What Gets Masked

| Original | Replaced With |
|----------|---------------|
| `555-123-4567` | `[PHONE]` |
| `john@email.com` | `[EMAIL]` |
| `https://website.com` | `[URL]` |
| `A1B 2C3` (postal code) | `[POSTAL_CODE]` |
| `4111-1111-1111-1111` | `[CARD]` |
| `Sunrise Animal Hospital` | `[HOSPITAL]` |
| `"Hi, this is Sarah"` | `"Hi, this is [PERSON]"` |

### Why This Matters

- **Privacy compliance**: Protects customer personal information
- **Safe for training**: Anonymized transcripts can be used to train AI without exposing real customer data
- **Shareable**: Reports and examples can be shared without privacy concerns

### How It Works

1. Finds transcripts that haven't been anonymized yet (or need re-processing)
2. Applies pattern matching to find and replace sensitive information
3. Creates a fingerprint (hash) of the original for tracking
4. Saves the anonymized version to a separate table
5. Marks the version used (so transcripts can be re-processed if patterns improve)

---

## Script 05: Export Training Data

**File:** `05_CallRail_ExportTrainingJSONL.py`

### What It Does

Takes your human-labeled, anonymized call data and packages it into files that can be used to train a custom AI model. This creates a "fine-tuned" version of ChatGPT that's specialized for your veterinary call analysis.

### Why Fine-Tune?

A custom-trained model:
- Understands your specific terminology and services
- Is more accurate for your use case
- Costs less per analysis (smaller, faster model)
- Can work offline or with different AI providers

### Output Files

The script creates two files:

1. **`callrail_training.jsonl`** (80% of data)
   - Used to teach the AI

2. **`callrail_validation.jsonl`** (20% of data)
   - Used to test if the AI learned correctly

### File Format

Each line contains one example:
```
{
  "messages": [
    {"role": "system", "content": "Instructions for the AI..."},
    {"role": "user", "content": "Transcript of the call..."},
    {"role": "assistant", "content": "Correct labels for this call..."}
  ]
}
```

### How It Works

1. Queries a database view that joins anonymized transcripts with human labels
2. Converts each labeled call into the training format
3. Randomly shuffles all examples
4. Splits into 80% training / 20% validation
5. Writes both files

### Next Steps After Export

The generated files can be uploaded to OpenAI's fine-tuning interface to create a custom model. This is typically done manually through the OpenAI dashboard or API.

---

## Running the Scripts

### Prerequisites

All scripts need:
- Python 3.8+
- Database connection details in a `.env` file
- Appropriate API keys (CallRail, OpenAI, Google)

### Typical Schedule

| Script | Frequency | Purpose |
|--------|-----------|---------|
| 01 | Daily (night) | Keep call data fresh |
| 02 | Daily or on-demand | Sync latest human labels |
| 03 | Daily (after 01) | Analyze new transcripts |
| 04 | Weekly or before training | Prepare data for training |
| 05 | On-demand | When ready to fine-tune |

### Logs

Each script creates log files in the Scripts folder:
- `callrail_etl.log` (Script 01)
- `anonymize_callrail_transcripts.log` (Script 04)

Script 03 logs to console by default, but can be configured to write to a file.

---

## Questions?

For technical details, see the code comments in each script or the `claude.md` file in the project root.
