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
# Load .env (OPENAI_API_KEY, SQLSERVER_*, etc.)
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

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# If you do nothing, the script will KEEP this stable version (nightly-safe).
DEFAULT_ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "prod_v1")

# Only used when you explicitly enable auto-versioning
DEFAULT_ANALYSIS_PREFIX = os.getenv("ANALYSIS_PREFIX", "v3_fast_bucketed_run_")

DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
DEFAULT_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "0"))  # 0 = unlimited

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")  # optional, e.g. C:\Logs\vetcare_transcript_analysis.log

TABLE_CALLS = os.getenv("TABLE_CALLS", "CallRailAPI")
TABLE_ANALYSIS = os.getenv("TABLE_ANALYSIS", "CallRailAPI_TranscriptAnalysis")

COL_CALL_ID = os.getenv("COL_CALL_ID", "id")
COL_TRANSCRIPT = os.getenv("COL_TRANSCRIPT", "transcription")  # RAW transcript column on CallRailAPI

# ============================================================
# SYSTEM PROMPT WITH YOUR BUCKETS
# ============================================================

SYSTEM_PROMPT = """
You are a meticulous veterinary call analyst.

You will receive JSON input with:
{
  "analysis_version": "<string>",
  "calls": [
    {
      "call_id": "CAL123",
      "transcript": "FULL RAW CALL TRANSCRIPT HERE"
    },
    ...
  ]
}

For EACH call, extract and classify the following fields using ONLY the buckets below.
Never invent your own categories. Always choose exactly one bucket per field unless instructed to return null.

--------------------------------------------------
APPOINTMENT BOOKED (choose ONE)
--------------------------------------------------
Yes
No
Inconclusive

--------------------------------------------------
CLIENT TYPE (choose ONE)
--------------------------------------------------
New
Existing
Inconclusive

--------------------------------------------------
TREATMENT TYPE (choose ONE)
--------------------------------------------------

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

--------------------------------------------------
REASONS NOT BOOKED (ONLY IF appointment_booked = "No")
Choose ONE of these buckets:
------------------------------------------------------

1. Caller Procrastination
1a. Caller Procrastination - Price Objection / Shopping / Request for Quote
1b. Caller Procrastination - Need to check with partner
1c. Caller Procrastination - Getting information for someone else

2. Scheduling Issue
2a. Scheduling Issue - Walk ins not available / no same day appt
2b. Scheduling Issue - Full schedule
2c. Scheduling Issue - Not open / no availability on evenings
2d. Scheduling Issue - Not open / no availability on weekends

3. Service/treatment not offered
3a. Service/treatment not offered - Grooming
3b. Service/treatment not offered - Pet Adoption
3c. Service/treatment not offered - Exotics
3d. Service/treatment not offered - Farm / Large Animals
3e. Service/treatment not offered - Birds
3f. Service/treatment not offered - Reptiles
3g. Service/treatment not offered - Pocket Pets

4. Meant to call competitor hospital
5. Meant to call low cost / free service provider
6. Emergency care not offered
7. File Transferred
8. Medication/food order
9. Client/appt query (non-medical)
10. Missed call
11. No transcription

RULES FOR reason_not_booked:
- Only populate if appointment_booked = "No".
- If appointment_booked = "Yes", set reason_not_booked = null.
- If appointment_booked = "Inconclusive", set reason_not_booked = null.
- Always return the FULL bucket label exactly as written above.

--------------------------------------------------
EXTRACTION FIELDS
--------------------------------------------------

stated_hospital_name:
- Extract the hospital name if spoken.
- If no hospital name is clearly spoken, return null.

stated_patient_name:
- Extract the pet's name if mentioned.
- If unclear, return null.

agent_name:
- Extract the staff member's name if they introduce themselves or are named.
- If no agent name is spoken, return null.

--------------------------------------------------
OUTPUT FORMAT (JSON ONLY)
--------------------------------------------------

Return ONLY JSON in the following shape:

{
  "analysis_version": "<copy input analysis_version>",
  "calls": [
    {
      "call_id": "CAL123",
      "appointment_booked": "Yes" | "No" | "Inconclusive",
      "client_type": "New" | "Existing" | "Inconclusive",
      "treatment_type": "<ONE Treatment Type bucket exactly as listed>",
      "stated_hospital_name": "<string or null>",
      "stated_patient_name": "<string or null>",
      "agent_name": "<string or null>",
      "reason_not_booked": "<ONE Reason Not Booked bucket or null>"
    },
    ...
  ]
}

Do NOT add any commentary or explanation. Return JSON ONLY.
""".strip()

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
    p = argparse.ArgumentParser(description="Analyze CallRail transcripts with OpenAI and bucket outputs.")
    p.add_argument("--version", help="Explicit analysis_version to use (stable nightly).")
    p.add_argument("--auto-version", action="store_true",
                   help="Auto-increment analysis_version using prefix + DB max().")
    p.add_argument("--prefix", help="Prefix used when --auto-version is set.")
    p.add_argument("--max-calls", type=int, help="Max calls to process this run (0=unlimited).")
    p.add_argument("--batch-size", type=int, help="Calls per OpenAI request.")
    p.add_argument("--max-concurrent", type=int, help="Parallel OpenAI requests.")
    p.add_argument("--dry-run", action="store_true", help="Fetch and log work, but do not call OpenAI or write to SQL.")
    return p.parse_args()

# ============================================================
# OPENAI CLIENT
# ============================================================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
    return OpenAI(api_key=api_key, max_retries=3)

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
# OPENAI BATCH CALL
# ============================================================

def call_openai_batch(client: OpenAI, model: str, calls: List[Dict[str, Any]], analysis_version: str) -> Dict[str, Any]:
    payload = {
        "analysis_version": analysis_version,
        "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in calls],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

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
    client = None if dry_run else get_openai_client()

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
                    executor.submit(call_openai_batch, client, model, batch, analysis_version): batch
                    for batch in batches
                }

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result_json = future.result()
                    except Exception as e:
                        logger.error(f"OpenAI call failed for batch (skipping {len(batch)} calls): {e}")
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
    model = DEFAULT_OPENAI_MODEL
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
