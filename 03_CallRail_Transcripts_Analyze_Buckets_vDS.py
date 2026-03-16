# analyze_callrail_transcripts_buckets_nightly.py
import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# Load .env (GEMINI_API_KEY, SQLSERVER_*, etc.)
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pyodbc
except ImportError:
    pyodbc = None
from openai import OpenAI

# ============================================================
# DEFAULTS (override via CLI flags or env vars)
# ============================================================

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# If you do nothing, the script will KEEP this stable version (nightly-safe).
DEFAULT_ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "prod_vDS")

# Only used when you explicitly enable auto-versioning
DEFAULT_ANALYSIS_PREFIX = os.getenv("ANALYSIS_PREFIX", "v3_fast_bucketed_run_")

DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
DEFAULT_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "0"))  # 0 = unlimited

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")  # optional, e.g. C:\Logs\vetcare_transcript_analysis.log

TABLE_CALLS = os.getenv("TABLE_CALLS", "CallRailAPI")
TABLE_ANALYSIS = os.getenv("TABLE_ANALYSIS", "CallRailAPI_TranscriptAnalysis")

COL_CALL_ID = os.getenv("COL_CALL_ID", "id")
COL_TRANSCRIPT = os.getenv("COL_TRANSCRIPT", "transcription")  # RAW transcript column on CallRailAPI

# ============================================================
# A1b PROMPTS & SCHEMAS
# ============================================================

# --- Call 1: appointment_booked ---

V9_APPOINTMENT_BOOKED_PROMPT = """You are a veterinary call transcript analyst. Determine whether an appointment was booked in this call.

## Categories

- **Yes**: An appointment was confirmed (date/time set, caller agreed to come in, or existing appointment adjusted)
- **No**: The caller did not end up with an appointment — includes: chose not to book, schedule was full, was put on a cancellation list, or clinic offered to check and call back
- **Inconclusive**: The call was not about booking (administrative, checking results, voicemail) so appointment status doesn't apply

Key distinction: if the caller TRIED to book but couldn't (schedule full, no same-day, put on callback list), that is **No** — they left the call without an appointment. Inconclusive is for calls where booking was never the purpose.

## Examples

### Yes — urgent booking
Transcript: "Agent: Can you come in at 9am? Caller: Yes, that works."
Answer: Yes

### No — got info and left
Transcript: "Caller: How much for an exam? Agent: $122. Caller: Okay, thanks. Bye."
Answer: No

### No — schedule full, clinic will check and call back
Transcript: "Agent: We're booked up, but let me take your number and see if we can squeeze you in. Caller: Okay, it's 555-1234."
Answer: No

### Inconclusive — admin call, not about booking
Transcript: "Caller: I'm calling to check on Max's blood test results. Agent: Results look normal."
Answer: Inconclusive

## Output Format

Return JSON with your reasoning and answer:
{"reasoning": "...", "answer": "Yes" | "No" | "Inconclusive"}

Return JSON ONLY.""".strip()

V9_APPOINTMENT_BOOKED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "appointment_booked_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["Yes", "No", "Inconclusive"]}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

# --- Enum lists for strict schema validation ---

TREATMENT_TYPE_ENUMS = [
    "Preventive Care",
    "Preventive Care \u2013 Vaccinations",
    "Preventive Care \u2013 Parasite Prevention",
    "Preventive Care \u2013 Annual Exams",
    "Preventive Care \u2013 Wellness Screening (Bloodwork, Urinalysis, Fecals)",
    "Urgent Care / Sick Pet",
    "Urgent Care \u2013 Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)",
    "Urgent Care \u2013 Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)",
    "Urgent Care \u2013 Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)",
    "Surgical Services",
    "Surgical Services \u2013 Spays and Neuters",
    "Surgical Services \u2013 Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)",
    "Surgical Services \u2013 Orthopedic Surgeries (ACL Repairs, Fracture Repair \u2014 Sometimes Referred Out)",
    "Surgical Services \u2013 Emergency Surgeries (Pyometra, C-Sections, GDV)",
    "Surgical Services \u2013 Dental Care (Cleanings, Extractions)",
    "Diagnostic Services",
    "Diagnostic Services \u2013 X-Rays (Digital Radiography)",
    "Diagnostic Services \u2013 Ultrasound",
    "Diagnostic Services \u2013 In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)",
    "Diagnostic Services \u2013 ECG or Blood Pressure Monitoring",
    "Emergency & Critical Care",
    "Emergency & Critical Care \u2013 Stabilization (Trauma, Poisoning, Seizures)",
    "Emergency & Critical Care \u2013 Overnight Hospitalization",
    "Emergency & Critical Care \u2013 Fluid Therapy, Oxygen Therapy, Intensive Monitoring",
    "Emergency & Critical Care \u2013 Referred to an Emergency Hospital",
    "Dermatology",
    "Dermatology \u2013 Allergies",
    "Dermatology \u2013 Ear Infections",
    "Retail",
    "Retail \u2013 Food Orders",
    "Retail \u2013 Prescriptions",
    "End of Life Care",
    "End of Life Care \u2013 In-Home Euthanasia",
    "End of Life Care \u2013 In-Clinic Euthanasia",
    "N/A (missed call)",
    "Other",
]

REASON_NOT_BOOKED_ENUMS = [
    "1. Caller Procrastination",
    "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote",
    "1b. Caller Procrastination - Need to check with partner",
    "1c. Caller Procrastination - Getting information for someone else",
    "2. Scheduling Issue",
    "2a. Scheduling Issue - Walk ins not available / no same day appt",
    "2b. Scheduling Issue - Full schedule",
    "2c. Scheduling Issue - Not open / no availability on evenings",
    "2d. Scheduling Issue - Not open / no availability on weekends",
    "3. Service/treatment not offered",
    "3a. Service/treatment not offered - Grooming",
    "3b. Service/treatment not offered - Pet Adoption",
    "3c. Service/treatment not offered - Exotics",
    "3d. Service/treatment not offered - Farm / Large Animals",
    "3e. Service/treatment not offered - Birds",
    "3f. Service/treatment not offered - Reptiles",
    "3g. Service/treatment not offered - Pocket Pets",
    "4. Meant to call competitor hospital",
    "5. Meant to call low cost / free service provider",
    "6. Emergency care not offered",
    "7. File Transferred",
    "8. Medication/food order",
    "9. Client/appt query (non-medical)",
    "10. Missed call",
    "11. No transcription",
    None,
]

# --- Call 2: client_type + treatment_type + reason_not_booked + names ---

A1B_COMBO_PROMPT = """You are a veterinary medicine expert and call transcript analyst. You have deep knowledge of veterinary services, common symptoms, diagnostic pathways, and how clinics triage patients. Use this clinical expertise alongside the transcript to classify each field below.

The appointment_booked decision has already been made: {appointment_booked}

Classify the following fields from this call transcript.

---

# FIELD 1: client_type

Determine whether the caller is a new or existing client at THIS specific clinic. "Existing" means the CALLER (not the pet) has been a client at this clinic before.

## Signals

**Existing:** Agent finds their file, pet already in system, caller references past visits here, knows doctor names
**New:** Asks "do you accept new patients?", unfamiliar with pricing/location, agent creates new file, mentions having a vet elsewhere

Casual/friendly tone alone does not indicate Existing — require concrete evidence.
Inconclusive should be extremely rare.

## Examples

### Existing — agent finds file
Transcript: "Agent: What's the name? Caller: Luna. Agent: I see Luna here, last visit was March."
Answer: Existing

### New — agent creates file
Transcript: "Caller: I want to book shots for my dog. Agent: Let me grab your number to open a file."
Answer: New

### New — no record found, agent creates file
Transcript: "Caller: I'd like to bring my cat in for a checkup. Agent: Sure, what's your last name? ... I'm not finding you in our system. Let me set up a new file."
Answer: New

---

# FIELD 2: treatment_type

Determine what veterinary service was discussed. Choose EXACTLY ONE category from the list below. Match the level of specificity that best fits — use a sub-category when the call is clearly about that specific service, use the parent when the call is general or covers multiple services.

## Key Guidelines

- Classify based on the PRIMARY reason for the call
- If a sick pet has specific symptoms (vomiting, ear infection, limping), use the relevant sub-category — you don't need a named procedure
- Emergency & Critical Care requires actual emergency-level situations (trauma, poisoning, critical symptoms) — not just calling an emergency hospital
- Routine bloodwork (annual, wellness, pre-op) = Preventive Care – Wellness Screening, NOT Diagnostic Services. Only use Diagnostic when the bloodwork is investigating a specific problem.
- Admin/rescheduling calls: if the transcript mentions WHAT the appointment is for (surgery, sick visit, etc.), classify by THAT service — only default to Preventive Care when the service type is truly unknown
- "Other" is a LAST RESORT. Before using Other, check:
  - Rescheduling/cancelling calls → classify by the underlying service if mentioned, else Preventive Care
  - Nail trims, microchip scans, general checkups → Preventive Care
  - Medication ordering for an existing condition → Retail – Prescriptions
  - Service not offered (exotics, grooming) → still use the relevant category (e.g., "Surgical Services" if they asked about a procedure)
  - If the caller mentioned ANY medical concern → classify by that concern

## Examples

### Sick cat with specific symptoms → sub-category
Transcript: "Caller: Luna's been vomiting foam and seems in pain, shallow breathing. Agent: Let's get her in at 9am."
Answer: Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)
Why: Specific symptoms (vomiting, pain) clearly map to diagnosis/treatment of illness.

### Dog limping, no further detail → parent
Transcript: "Caller: My dog's been limping. Agent: Let's get him in tomorrow. Caller: Okay, thanks."
Answer: Urgent Care / Sick Pet
Why: Limping could be many things — not enough to pick a specific sub-category.

### Ear problem → Dermatology sub
Transcript: "Caller: His ear is all black inside. Agent: We should take a look at that."
Answer: Dermatology – Ear Infections
Why: Ear issue is the primary complaint, maps directly to the ear infections sub-category.

### Solely vaccinations → Preventive Care sub
Transcript: "Caller: I want to book shots for my dog. Agent: We have tomorrow at 12."
Answer: Preventive Care – Vaccinations
Why: Sole stated purpose is vaccinations, nothing else discussed.

### Annual checkup → Annual Exams sub
Transcript: "Caller: I need to schedule a yearly appointment for my cat. Agent: We have an opening on Thursday."
Answer: Preventive Care – Annual Exams
Why: Caller explicitly requests annual/yearly appointment.

### Vaccines + checkup → Vaccinations (primary purpose wins)
Transcript: "Caller: I need to bring my dog in for his booster shot. Agent: We can also do a quick checkup while he's here."
Answer: Preventive Care – Vaccinations
Why: Primary purpose is vaccinations. The checkup is secondary — classify by what the caller called about.

### Prescription refill → Retail
Transcript: "Caller: Henry's running out of his medication, can I get a refill?"
Answer: Retail – Prescriptions
Why: Existing prescription refill, not a new medical concern.

### Porcupine quills, emergency → Emergency
Transcript: "Caller: Dog ran into a porcupine, quills everywhere. Agent: Come in right away, it'll be $1020."
Answer: Emergency & Critical Care – Stabilization (Trauma, Poisoning, Seizures)
Why: Acute trauma requiring emergency intervention.

### Multiple services including surgery → prioritize surgery
Transcript: "Caller: I need to book a spay and vaccines for my dog."
Answer: Surgical Services – Spays and Neuters
Why: When multiple services are discussed, prioritize the most significant medical procedure.

### Rescheduling call, no medical detail → Preventive Care (default)
Transcript: "Caller: I need to move my appointment to next week. Agent: Sure, how about Tuesday at 3?"
Answer: Preventive Care
Why: Admin/rescheduling call with no medical content discussed. Default to Preventive Care, NOT Other.

### Routine bloodwork → Wellness Screening, NOT Diagnostic
Transcript: "Caller: I'd like to schedule bloodwork for my senior dog, he's due for his annual check. Agent: Sure, we can do that Thursday."
Answer: Preventive Care – Wellness Screening (Bloodwork, Urinalysis, Fecals)
Why: Routine/annual bloodwork is Wellness Screening under Preventive Care, not Diagnostic Services.

### Surgery rescheduling → classify by the surgery, not as Preventive Care
Transcript: "Caller: I need to reschedule Bella's spay from next week. Agent: How about the 15th?"
Answer: Surgical Services – Spays and Neuters
Why: Even though this is a rescheduling call, the underlying service (spay) is mentioned — classify by that.

### Elderly dog declining → End of Life Care, not Urgent Care
Transcript: "Caller: My 16-year-old lab isn't eating and can barely walk anymore. We think it might be time. Agent: I'm so sorry. Would you like to discuss options?"
Answer: End of Life Care
Why: Context (very old pet, "might be time") indicates end-of-life discussion, not just a sick pet visit.

## Categories

Preventive Care
Preventive Care – Vaccinations
Preventive Care – Parasite Prevention
Preventive Care – Annual Exams
Preventive Care – Wellness Screening (Bloodwork, Urinalysis, Fecals)
Urgent Care / Sick Pet
Urgent Care – Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)
Urgent Care – Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)
Urgent Care – Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)
Surgical Services
Surgical Services – Spays and Neuters
Surgical Services – Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)
Surgical Services – Orthopedic Surgeries (ACL Repairs, Fracture Repair — Sometimes Referred Out)
Surgical Services – Emergency Surgeries (Pyometra, C-Sections, GDV)
Surgical Services – Dental Care (Cleanings, Extractions)
Diagnostic Services
Diagnostic Services – X-Rays (Digital Radiography)
Diagnostic Services – Ultrasound
Diagnostic Services – In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)
Diagnostic Services – ECG or Blood Pressure Monitoring
Emergency & Critical Care
Emergency & Critical Care – Stabilization (Trauma, Poisoning, Seizures)
Emergency & Critical Care – Overnight Hospitalization
Emergency & Critical Care – Fluid Therapy, Oxygen Therapy, Intensive Monitoring
Emergency & Critical Care – Referred to an Emergency Hospital
Dermatology
Dermatology – Allergies
Dermatology – Ear Infections
Retail
Retail – Food Orders
Retail – Prescriptions
End of Life Care
End of Life Care – In-Home Euthanasia
End of Life Care – In-Clinic Euthanasia
N/A (missed call)
Other

---

# FIELD 3: reason_not_booked

Determine why an appointment was NOT booked.

## Rules

- If appointment_booked is "Yes": answer is null.
- If appointment_booked is "Inconclusive": answer is null (unless there's an explicit barrier like "we're fully booked").
- If appointment_booked is "No": choose the most specific matching category below.

## Key Distinctions

- **Parent vs sub-category:** Use the PARENT category (e.g., "1. Caller Procrastination" or "2. Scheduling Issue") unless the sub-category is a clear, unambiguous match. When in doubt, use the parent.
- **Price Objection (1a):** Use when the caller's PRIMARY reason for not booking was the price. The caller asked about cost and that was the deciding factor. If price was discussed but something ELSE was the real barrier (schedule full, service not offered, caller just procrastinating), use the real barrier instead.
- "I'll think about it" with NO price discussion → 1 (Procrastination)
- Caller cancels and says they'll reschedule later → 1 (Procrastination), NOT 9 (Client/appt query)
- Caller asked about a service the clinic doesn't offer → 3 (Service not offered), even if price was also discussed
- Wants same-day, told none available → 2a
- Schedule full for days/weeks → 2b
- If scheduling was the issue but you're unsure between 2a/2b/2c/2d → use parent "2. Scheduling Issue"
- Caller seeking a specific service style the clinic doesn't offer (holistic, outdoor euthanasia, exotics) → 3 (Service not offered)

## Examples

### Pure price shopping → 1a
Transcript: "Caller: How much for an exam? Agent: $122. Caller: Wow, okay. Thanks. Bye."
Answer: "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"
Why: Caller's sole purpose was pricing. Price was the barrier.

### Price asked BUT scheduling was the real barrier → Scheduling
Transcript: "Caller: How much for a spay? Agent: $350. Caller: Okay, when's the soonest? Agent: June 17th. Caller: That's too far out."
Answer: "2. Scheduling Issue"
Why: Caller accepted the price and asked to book. The scheduling gap was the real barrier.

### Caller cancels and will reschedule → Procrastination
Transcript: "Caller: I need to cancel Thursday's appointment, something came up. I'll call back next week."
Answer: "1. Caller Procrastination"
Why: Caller is postponing, not objecting to price or encountering a scheduling barrier.

### Service not offered → even if price discussed
Transcript: "Caller: Do you do holistic treatments? Agent: No. Caller: How much for a regular visit? Agent: $122. Caller: Okay, thanks."
Answer: "3. Service/treatment not offered"
Why: Primary reason: clinic doesn't offer what caller wanted. Price question was secondary.

### Same-day not available → 2a specifically
Transcript: "Caller: Can I get my puppy in today? He's been coughing. Agent: We're pretty booked, let me check... I'll take your number and call you back."
Answer: "2a. Scheduling Issue - Walk ins not available / no same day appt"
Why: Caller wanted same-day. Use 2a when the caller specifically needed today/now and was told none available.

### Caller canceled + no availability → still Procrastination
Transcript: "Caller: I need to cancel my appointment. Is there anything this week? Agent: We're pretty booked this week. Caller: Okay, I'll call back later."
Answer: "1. Caller Procrastination"
Why: The caller initiated the cancellation. Even though the schedule is full, the root cause is the caller's decision to cancel — not a scheduling barrier.

## Categories

- "1. Caller Procrastination"
- "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote"
- "1b. Caller Procrastination - Need to check with partner"
- "1c. Caller Procrastination - Getting information for someone else"
- "2. Scheduling Issue"
- "2a. Scheduling Issue - Walk ins not available / no same day appt"
- "2b. Scheduling Issue - Full schedule"
- "2c. Scheduling Issue - Not open / no availability on evenings"
- "2d. Scheduling Issue - Not open / no availability on weekends"
- "3. Service/treatment not offered"
- "3a. Service/treatment not offered - Grooming"
- "3b. Service/treatment not offered - Pet Adoption"
- "3c. Service/treatment not offered - Exotics"
- "3d. Service/treatment not offered - Farm / Large Animals"
- "3e. Service/treatment not offered - Birds"
- "3f. Service/treatment not offered - Reptiles"
- "3g. Service/treatment not offered - Pocket Pets"
- "4. Meant to call competitor hospital"
- "5. Meant to call low cost / free service provider"
- "6. Emergency care not offered"
- "7. File Transferred"
- "8. Medication/food order"
- "9. Client/appt query (non-medical)"
- "10. Missed call"
- "11. No transcription"

---

# FIELD 4: Name Extraction

- stated_hospital_name: Extract the hospital or clinic name if spoken during the call. null if not mentioned.
- stated_patient_name: Extract the pet's name if mentioned. null if unclear or not mentioned.
- agent_name: Extract the staff member's name if they introduce themselves or are named. null if not spoken.

---

# Output Format

Return JSON with your reasoning and all field answers:
{"reasoning": "...", "client_type": "New"|"Existing"|"Inconclusive", "treatment_type": "<exact category from list above>", "reason_not_booked": "<exact category or null>", "stated_hospital_name": "<string or null>", "stated_patient_name": "<string or null>", "agent_name": "<string or null>"}

Return JSON ONLY.""".strip()

A1B_COMBO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "a1b_combo_3field",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "client_type": {"type": "string", "enum": ["New", "Existing", "Inconclusive"]},
                "treatment_type": {"type": "string", "enum": TREATMENT_TYPE_ENUMS},
                "reason_not_booked": {"type": ["string", "null"], "enum": REASON_NOT_BOOKED_ENUMS},
                "stated_hospital_name": {"type": ["string", "null"]},
                "stated_patient_name": {"type": ["string", "null"]},
                "agent_name": {"type": ["string", "null"]},
            },
            "required": ["reasoning", "client_type", "treatment_type", "reason_not_booked",
                         "stated_hospital_name", "stated_patient_name", "agent_name"],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("callrail_analysis")


logger = setup_logging()

# ============================================================
# CLI ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Analyze CallRail transcripts with Gemini and bucket outputs.")
    p.add_argument("--version", help="Explicit analysis_version to use (stable nightly).")
    p.add_argument("--auto-version", action="store_true",
                   help="Auto-increment analysis_version using prefix + DB max().")
    p.add_argument("--prefix", help="Prefix used when --auto-version is set.")
    p.add_argument("--max-calls", type=int, help="Max calls to process this run (0=unlimited).")
    p.add_argument("--batch-size", type=int, help="Calls per LLM request.")
    p.add_argument("--max-concurrent", type=int, help="Parallel LLM requests.")
    p.add_argument("--dry-run", action="store_true", help="Fetch and log work, but do not call LLM or write to SQL.")
    return p.parse_args()

# ============================================================
# LLM CLIENT (Gemini via OpenAI-compatible endpoint)
# ============================================================

def get_llm_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or LLM_API_KEY / OPENAI_API_KEY) not found in environment or .env")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=3)

# ============================================================
# DB CONNECTION & VERSIONING
# ============================================================

def get_db_connection():
    driver = os.getenv("SQLSERVER_DRIVER", "{ODBC Driver 17 for SQL Server}")
    server = os.getenv("SQLSERVER_SERVER")
    database = os.getenv("SQLSERVER_DATABASE")
    uid = os.getenv("SQLSERVER_UID")
    pwd = os.getenv("SQLSERVER_PWD")

    if not all([server, database, uid, pwd]):
        raise RuntimeError(
            "Missing SQLSERVER_* env vars. Need SQLSERVER_SERVER, SQLSERVER_DATABASE, "
            "SQLSERVER_UID, SQLSERVER_PWD."
        )

    conn_str = (
        f"DRIVER={driver};"
        f"SERVER=tcp:{server},1433;"
        f"DATABASE={database};"
        f"UID={uid};PWD={pwd};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    logger.info(f"Connecting to SQL Server at {server}...")
    return pyodbc.connect(conn_str)


def get_next_analysis_version(conn, prefix: str) -> str:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT MAX(analysis_version)
        FROM {TABLE_ANALYSIS}
        WHERE analysis_version LIKE ?
        """,
        f"{prefix}%",
    )
    row = cur.fetchone()
    last = row[0] if row else None

    if not last:
        return f"{prefix}0001"

    try:
        suffix = last.split("_")[-1]
        num = int(suffix)
        return f"{prefix}{num + 1:04d}"
    except Exception:
        return f"{prefix}0001"

# ============================================================
# FETCH WORK (ONLY IDs NOT IN ANALYSIS TABLE)
# ============================================================

def fetch_unanalyzed_calls(conn, limit: int) -> List[Dict[str, Any]]:
    """
    Fetch calls that have a RAW transcript on CallRailAPI and no existing
    row yet in CallRailAPI_TranscriptAnalysis (PK is id).
    """
    sql = f"""
    SELECT TOP (?) c.{COL_CALL_ID}, c.{COL_TRANSCRIPT}
    FROM {TABLE_CALLS} c
    LEFT JOIN {TABLE_ANALYSIS} a ON a.{COL_CALL_ID} = c.{COL_CALL_ID}
    WHERE a.{COL_CALL_ID} IS NULL
      AND c.{COL_TRANSCRIPT} IS NOT NULL
      AND LEN(c.{COL_TRANSCRIPT}) > 0
    ORDER BY c.start_time
    """
    cur = conn.cursor()
    cur.execute(sql, (limit,))
    rows = cur.fetchall()
    return [{"id": r[0], "transcript": r[1]} for r in rows]

# ============================================================
# A1b 2-CALL PIPELINE (replaces single call_openai_batch)
# ============================================================

def _build_batched_schema(schema):
    """Wrap a single-item schema into a batched {results: [...]} schema."""
    single_props = schema["json_schema"]["schema"]["properties"].copy()
    single_req = list(schema["json_schema"]["schema"]["required"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema["json_schema"]["name"] + "_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {**single_props, "call_id": {"type": "string"}},
                            "required": single_req + ["call_id"],
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            }
        }
    }


def _make_batched_prompt(prompt):
    """Convert a single-transcript prompt to batch mode."""
    return prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently \u2014 do not let one transcript '
        'influence your classification of another.\n\n'
        'Return JSON: {"results": [{"call_id": "...", ...}, ...]}\n'
        'Return JSON ONLY.'
    )


def _call_api_batch(client, model, payload, system_prompt, schema):
    """Make a single batched API call with retry."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=schema,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 1.0)
                logger.warning(f"API batch attempt {attempt + 1} failed: {e}")
            else:
                raise


def call_llm_batch(client: OpenAI, model: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A1b 2-call pipeline for a batch of calls:
      Call 1: appointment_booked (all calls)
      Call 2: client_type + treatment_type + reason_not_booked + names
              (grouped by appointment answer for context)

    Returns {"calls": [...]} matching the shape expected by convert_results_to_rows().
    """
    # Build payload for Call 1
    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls
    ])

    # --- Call 1: appointment_booked ---
    appt_prompt = _make_batched_prompt(V9_APPOINTMENT_BOOKED_PROMPT)
    appt_schema = _build_batched_schema(V9_APPOINTMENT_BOOKED_SCHEMA)
    appt_result = _call_api_batch(client, model, payload, appt_prompt, appt_schema)
    appt_by_id = {
        str(r["call_id"]): r["answer"]
        for r in appt_result.get("results", [])
    }

    # Group calls by appointment answer
    groups = {"Yes": [], "No": [], "Inconclusive": []}
    for c in calls:
        answer = appt_by_id.get(str(c["id"]), "Inconclusive")
        groups[answer].append(c)

    logger.debug(
        f"Appt split: Yes={len(groups['Yes'])}, No={len(groups['No'])}, "
        f"Inconclusive={len(groups['Inconclusive'])}"
    )

    # --- Call 2: combo (3 fields + names) per appointment group ---
    combo_by_id = {}
    for appt_answer, group_calls in groups.items():
        if not group_calls:
            continue

        prompt = A1B_COMBO_PROMPT.replace("{appointment_booked}", appt_answer)
        combo_prompt = _make_batched_prompt(prompt)
        combo_schema = _build_batched_schema(A1B_COMBO_SCHEMA)

        group_payload = json.dumps([
            {"call_id": c["id"], "transcript": c["transcript"]}
            for c in group_calls
        ])

        combo_result = _call_api_batch(client, model, group_payload, combo_prompt, combo_schema)
        for r in combo_result.get("results", []):
            combo_by_id[str(r["call_id"])] = r

    # --- Merge Call 1 + Call 2 ---
    merged_calls = []
    for c in calls:
        cid = str(c["id"])
        appt = appt_by_id.get(cid, "Inconclusive")
        combo = combo_by_id.get(cid, {})
        reason = combo.get("reason_not_booked")

        # Cross-field consistency
        if appt != "No":
            reason = None

        merged_calls.append({
            "call_id": cid,
            "appointment_booked": appt,
            "client_type": combo.get("client_type"),
            "treatment_type": combo.get("treatment_type"),
            "reason_not_booked": reason,
            "stated_hospital_name": combo.get("stated_hospital_name"),
            "stated_patient_name": combo.get("stated_patient_name"),
            "agent_name": combo.get("agent_name"),
        })

    return {"calls": merged_calls}

# ============================================================
# NULL NORMALIZATION + RESULT CONVERSION
# ============================================================

def normalize_null(v):
    if v is None:
        return None
    if isinstance(v, str):
        x = v.strip()
        if x == "" or x.lower() in ("null", "none", "n/a"):
            return None
    return v


def convert_results_to_rows(
    batch_calls: List[Dict[str, Any]],
    result_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    calls_results = result_json.get("calls", [])
    if not isinstance(calls_results, list):
        calls_results = []

    by_id: Dict[str, Any] = {}
    for item in calls_results:
        cid = item.get("call_id")
        if cid:
            by_id[str(cid)] = item

    rows: List[Dict[str, Any]] = []
    for call in batch_calls:
        cid = call["id"]
        item = by_id.get(str(cid), {})

        appointment_booked = normalize_null(item.get("appointment_booked")) or "Inconclusive"
        client_type = normalize_null(item.get("client_type")) or "Inconclusive"
        treatment_type = normalize_null(item.get("treatment_type")) or "Inconclusive"
        stated_hospital_name = normalize_null(item.get("stated_hospital_name"))
        stated_patient_name = normalize_null(item.get("stated_patient_name"))
        agent_name = normalize_null(item.get("agent_name"))
        reason_not_booked = normalize_null(item.get("reason_not_booked"))

        if appointment_booked != "No":
            reason_not_booked = None

        rows.append(
            {
                "id": cid,
                "appointment_booked": appointment_booked,
                "client_type": client_type,
                "treatment_type": treatment_type,
                "stated_hospital_name": stated_hospital_name,
                "stated_patient_name": stated_patient_name,
                "agent_name": agent_name,
                "reason_not_booked": reason_not_booked,
            }
        )
    return rows

# ============================================================
# UPSERT
# ============================================================

def upsert_analysis(conn, rows: List[Dict[str, Any]], analysis_version: str) -> None:
    if not rows:
        return

    cur = conn.cursor()
    now = datetime.now(timezone.utc)

    update_sql = f"""
    UPDATE {TABLE_ANALYSIS}
    SET
        stated_hospital_name = ?,
        appointment_booked   = ?,
        client_type          = ?,
        agent_name           = ?,
        reason_not_booked    = ?,
        treatment_type       = ?,
        analyzed_at          = ?,
        analysis_version     = ?,
        stated_patient_name  = ?
    WHERE id = ?
    """

    insert_sql = f"""
    INSERT INTO {TABLE_ANALYSIS} (
        id,
        stated_hospital_name,
        appointment_booked,
        client_type,
        agent_name,
        reason_not_booked,
        treatment_type,
        analyzed_at,
        analysis_version,
        stated_patient_name
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for r in rows:
        cur.execute(
            update_sql,
            (
                r.get("stated_hospital_name"),
                r["appointment_booked"],
                r["client_type"],
                r.get("agent_name"),
                r.get("reason_not_booked"),
                r["treatment_type"],
                now,
                analysis_version,
                r.get("stated_patient_name"),
                r["id"],
            ),
        )

        if cur.rowcount == 0:
            cur.execute(
                insert_sql,
                (
                    r["id"],
                    r.get("stated_hospital_name"),
                    r["appointment_booked"],
                    r["client_type"],
                    r.get("agent_name"),
                    r.get("reason_not_booked"),
                    r["treatment_type"],
                    now,
                    analysis_version,
                    r.get("stated_patient_name"),
                ),
            )

    conn.commit()

# ============================================================
# MAIN PIPELINE
# ============================================================

def process_backlog(
    analysis_version: str,
    model: str,
    batch_size: int,
    max_concurrent: int,
    max_calls_per_run: int,
    dry_run: bool,
):
    conn = get_db_connection()
    client = None if dry_run else get_llm_client()

    processed = 0
    started = time.time()

    try:
        while True:
            if max_calls_per_run and processed >= max_calls_per_run:
                logger.info(f"Reached max_calls_per_run={max_calls_per_run}")
                break

            # Fetch a bigger chunk so the thread pool stays busy
            to_fetch = batch_size * max_concurrent * 10
            if max_calls_per_run:
                remaining = max_calls_per_run - processed
                if remaining <= 0:
                    break
                to_fetch = min(to_fetch, remaining)

            calls = fetch_unanalyzed_calls(conn, to_fetch)
            if not calls:
                logger.info("No more calls to process.")
                break

            logger.info(f"Fetched {len(calls)} unanalyzed calls (analysis_version={analysis_version})")

            if dry_run:
                # Just show what would happen
                sample_ids = [c["id"] for c in calls[: min(10, len(calls))]]
                logger.info(f"[DRY RUN] Would process {len(calls)} calls. Sample IDs: {sample_ids}")
                break

            # Chunk into batches
            batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_batch = {
                    executor.submit(call_llm_batch, client, model, batch): batch
                    for batch in batches
                }

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result_json = future.result()
                    except Exception as e:
                        logger.error(f"LLM call failed for batch (skipping {len(batch)} calls): {e}")
                        continue

                    rows = convert_results_to_rows(batch, result_json)
                    upsert_analysis(conn, rows, analysis_version)

                    processed += len(rows)
                    logger.info(f"Processed total={processed} calls")

    finally:
        conn.close()
        elapsed = time.time() - started
        logger.info(f"DONE. processed={processed}, elapsed_sec={elapsed:.1f}")


def main():
    args = parse_args()

    # Decide analysis_version:
    # 1) --version wins
    # 2) else if --auto-version, generate next from DB with prefix
    # 3) else pinned default (nightly-safe)
    version = args.version
    auto_version = bool(args.auto_version)

    prefix = args.prefix or DEFAULT_ANALYSIS_PREFIX
    model = DEFAULT_MODEL
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    max_concurrent = args.max_concurrent or DEFAULT_MAX_CONCURRENT_REQUESTS
    max_calls = args.max_calls if args.max_calls is not None else DEFAULT_MAX_CALLS_PER_RUN
    dry_run = bool(args.dry_run)

    conn = None
    try:
        if not version and auto_version:
            conn = get_db_connection()
            version = get_next_analysis_version(conn, prefix)
            logger.info(f"✅ Auto-generated analysis_version = {version}")
        elif not version:
            version = DEFAULT_ANALYSIS_VERSION
            logger.info(f"✅ Using pinned analysis_version = {version}")
        else:
            logger.info(f"✅ Using explicit analysis_version = {version}")

    finally:
        if conn:
            conn.close()

    logger.info(
        f"Starting transcript bucket analysis | version={version} | model={model} | "
        f"batch_size={batch_size} | max_concurrent={max_concurrent} | max_calls={max_calls} | dry_run={dry_run}"
    )

    process_backlog(
        analysis_version=version,
        model=model,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        max_calls_per_run=max_calls,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
