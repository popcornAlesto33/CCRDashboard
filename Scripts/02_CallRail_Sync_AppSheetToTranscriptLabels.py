#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import pyodbc
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# ==========================================
# ENV / CONFIG
# ==========================================

# Resolve paths relative to this script (SQL Agent-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')

if not os.path.exists(ENV_PATH):
    raise RuntimeError(f'.env not found at: {ENV_PATH}')

load_dotenv(ENV_PATH)  # always load the .env next to the script

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
if not SPREADSHEET_ID:
    raise RuntimeError("SPREADSHEET_ID is missing or empty. Check your .env file.")

WORKSHEET_NAME = "labels"

# Path to service account key: ./keys/molten-tendril-480221-m9-9565a7b17470.json
SERVICE_ACCOUNT_FILE = os.path.join(
    BASE_DIR,
    "keys",
    "molten-tendril-480221-m9-9565a7b17470.json"
)

SQL_TABLE = "dbo.CallRailAPI_TranscriptLabels"
USE_TRUNCATE = True  # full replace pattern

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ==========================================
# SQL CONNECTION (MATCHES TRANSCRIPT SCRIPT)
# ==========================================

def _build_sql_conn_str():
    server   = os.getenv("SQLSERVER_SERVER") or os.getenv("SQL_HOST")
    port     = os.getenv("SQLSERVER_PORT", "1433")
    database = os.getenv("SQLSERVER_DATABASE") or os.getenv("SQL_DATABASE")
    uid      = os.getenv("SQLSERVER_UID") or os.getenv("SQL_USERNAME")
    pwd      = os.getenv("SQLSERVER_PWD") or os.getenv("SQL_PASSWORD")
    encrypt  = os.getenv("SQLSERVER_ENCRYPT", "yes")
    trust    = os.getenv("SQLSERVER_TRUST_SERVER_CERT", "yes")

    if not server or not database:
        raise RuntimeError("Missing SQLSERVER_SERVER and/or SQLSERVER_DATABASE")

    if uid and pwd:
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER=tcp:{server},{port};"
            f"DATABASE={database};UID={uid};PWD={pwd};"
            f"Encrypt={encrypt};TrustServerCertificate={trust};"
            "Connection Timeout=15;"
        )

    # Trusted connection branch (if using integrated auth)
    return (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=tcp:{server},{port};"
        f"DATABASE={database};Trusted_Connection=yes;"
        f"Encrypt={encrypt};TrustServerCertificate={trust};"
        "Connection Timeout=15;"
    )


def get_sql_connection():
    conn_str = _build_sql_conn_str()
    return pyodbc.connect(conn_str)

# ==========================================
# GOOGLE SHEETS
# ==========================================

def get_sheet_rows():
    """Fetch all rows from the Google Sheet as a list of dicts."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Service account key not found: {SERVICE_ACCOUNT_FILE}")

    scope = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        SERVICE_ACCOUNT_FILE, scope
    )
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)

    records = ws.get_all_records()
    logging.info("Fetched %d rows from Google Sheet", len(records))
    return records

# ==========================================
# SYNC LOGIC – FULL REPLACE + DEDUPE
# ==========================================

def sync_full_replace():
    """
    Full one-way sync: Google Sheet → SQL table.
    - Reads all rows from Sheet
    - Truncates the target table
    - Skips rows with blank id
    - Dedupes by id (last occurrence wins)
    """
    rows = get_sheet_rows()

    if not rows:
        logging.warning("No rows found in Sheet; nothing to sync.")
        return

    # Expect headers:
    # id, stated_hospital_name, appointment_booked, client_type,
    # agent_name, stated_patient_name, reason_not_booked, treatment_type,
    # labled_at, labeled_by, label_id, treatment_type_other
    columns = list(rows[0].keys())
    logging.info("Columns: %s", columns)

    placeholders = ", ".join(["?"] * len(columns))
    col_list = ", ".join(f"[{c}]" for c in columns)

    conn = get_sql_connection()
    cursor = conn.cursor()

    try:
        conn.autocommit = False
        logging.info("Starting SQL transaction...")

        # 1) Wipe table (full replace)
        if USE_TRUNCATE:
            logging.info("Truncating table %s", SQL_TABLE)
            cursor.execute(f"TRUNCATE TABLE {SQL_TABLE};")
        else:
            logging.info("Deleting all rows from %s", SQL_TABLE)
            cursor.execute(f"DELETE FROM {SQL_TABLE};")

        # 2) Prepare rows for insert
        insert_sql = f"INSERT INTO {SQL_TABLE} ({col_list}) VALUES ({placeholders})"

        id_to_values = {}
        skipped_blank_ids = 0
        duplicate_ids = 0

        for row in rows:
            # Normalize "" -> None
            values = [row.get(col) if row.get(col) != "" else None for col in columns]

            # Assuming first column is 'id'
            id_value = values[0]
            if id_value is None or str(id_value).strip() == "":
                skipped_blank_ids += 1
                continue

            id_str = str(id_value).strip()
            if id_str in id_to_values:
                duplicate_ids += 1  # we will overwrite previous with this one

            id_to_values[id_str] = values

        data_to_insert = list(id_to_values.values())

        logging.info(
            "Prepared %d unique-id rows for insert "
            "(skipped %d rows with blank id, found %d duplicate-id rows).",
            len(data_to_insert),
            skipped_blank_ids,
            duplicate_ids,
        )

        if not data_to_insert:
            logging.warning("No rows with valid id to insert; aborting insert.")
            conn.rollback()
            return

        # 3) Bulk insert
        logging.info("Inserting %d rows into %s", len(data_to_insert), SQL_TABLE)
        cursor.executemany(insert_sql, data_to_insert)

        conn.commit()
        logging.info("Sync complete. Inserted %d rows into %s.", len(data_to_insert), SQL_TABLE)

    except Exception:
        logging.exception("Error during sync, rolling back.")
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    logging.info("Using service account key: %s", SERVICE_ACCOUNT_FILE)
    logging.info("Starting full-replace sync: Google Sheet → %s", SQL_TABLE)
    sync_full_replace()
    logging.info("Done.")
