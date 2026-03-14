#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export training data for OpenAI fine-tuning:

- Reads from dbo.vw_CallRail_TrainingDataset
- Writes JSONL in chat-completion fine-tuning format
  one file for training, one for validation (80/20 split)

Output:
  callrail_training.jsonl
  callrail_validation.jsonl
"""

import os
import sys
import json
import random
import logging

import pyodbc
from dotenv import load_dotenv

load_dotenv()

TRAIN_FRACTION = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42


def setup_logging():
    log = logging.getLogger("ExportTraining")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


def build_sql_conn_str(logger):
    server = os.getenv("SQLSERVER_SERVER") or os.getenv("SQL_HOST")
    port = os.getenv("SQLSERVER_PORT", "1433")
    database = os.getenv("SQLSERVER_DATABASE") or os.getenv("SQL_DATABASE")
    uid = os.getenv("SQLSERVER_UID") or os.getenv("SQL_USERNAME")
    pwd = os.getenv("SQLSERVER_PWD") or os.getenv("SQL_PASSWORD")
    encrypt = os.getenv("SQLSERVER_ENCRYPT", "yes")
    trust = os.getenv("SQLSERVER_TRUST_SERVER_CERT", "yes")

    if not server or not database:
        raise RuntimeError("Missing SQLSERVER_SERVER and/or SQLSERVER_DATABASE")

    if uid and pwd:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER=tcp:{server},{port};"
            f"DATABASE={database};UID={uid};PWD={pwd};"
            f"Encrypt={encrypt};TrustServerCertificate={trust};"
        )
    else:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER=tcp:{server},{port};"
            f"DATABASE={database};Trusted_Connection=yes;"
            f"Encrypt={encrypt};TrustServerCertificate={trust};"
        )

    logger.info("Connecting to SQL Server tcp:%s,%s db=%s", server, port, database)
    return conn_str


def fetch_all_rows(conn, logger):
    sql = """
        SELECT
            id,
            transcript_anonymized,
            stated_hospital_name,
            appointment_booked,
            client_type,
            agent_name,
            stated_patient_name,
            reason_not_booked,
            treatment_type
        FROM dbo.vw_CallRail_TrainingDataset;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    logger.info("Fetched %d labeled rows for training", len(rows))
    return rows


def make_example(row):
    (
        cid,
        transcript,
        stated_hospital_name,
        appointment_booked,
        client_type,
        agent_name,
        stated_patient_name,
        reason_not_booked,
        treatment_type,
    ) = row

    # Normalize a bit – keep labels as-is if already good
    def norm(x):
        if x is None:
            return None
        s = str(x).strip()
        return s or None

    data = {
        "stated_hospital_name": norm(stated_hospital_name),
        "appointment_booked": norm(appointment_booked),
        "client_type": norm(client_type),
        "agent_name": norm(agent_name),
        "stated_patient_name": norm(stated_patient_name),
        "reason_not_booked": norm(reason_not_booked),
        "treatment_type": norm(treatment_type),
    }

    # System prompt defines the task; user prompt contains transcript
    system_prompt = (
        "You are an assistant that analyzes anonymized phone call transcripts "
        "for a veterinary hospital group. Given a transcript, you must extract "
        "a JSON object with exactly these keys:\n"
        "  - stated_hospital_name\n"
        "  - appointment_booked (one of 'Yes','No','Inconclusive')\n"
        "  - client_type (one of 'New','Existing','Unknown')\n"
        "  - agent_name\n"
        "  - stated_patient_name\n"
        "  - reason_not_booked\n"
        "  - treatment_type\n\n"
        "If something is not clearly stated, use null or 'Unknown'."
    )

    user_content = f"Transcript:\n{transcript}"

    assistant_content = json.dumps(data, ensure_ascii=False)

    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    return example


def main():
    logger = setup_logging()
    conn_str = build_sql_conn_str(logger)

    with pyodbc.connect(conn_str) as conn:
        rows = fetch_all_rows(conn, logger)

    if not rows:
        logger.error("No labeled rows found; aborting.")
        sys.exit(1)

    examples = [make_example(r) for r in rows]

    random.Random(RANDOM_SEED).shuffle(examples)
    split = int(len(examples) * TRAIN_FRACTION)
    train_examples = examples[:split]
    val_examples = examples[split:]

    logger.info("Train examples: %d, Validation examples: %d", len(train_examples), len(val_examples))

    with open("callrail_training.jsonl", "w", encoding="utf-8") as f_train:
        for ex in train_examples:
            f_train.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open("callrail_validation.jsonl", "w", encoding="utf-8") as f_val:
        for ex in val_examples:
            f_val.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Wrote callrail_training.jsonl and callrail_validation.jsonl")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)
