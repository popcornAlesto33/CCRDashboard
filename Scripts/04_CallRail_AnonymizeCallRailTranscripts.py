import re
import hashlib

PHONE_RE = re.compile(r'\+?\d[\d\-\(\)\s]{7,}\d')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
POSTAL_RE = re.compile(
    r'\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s*\d[ABCEGHJ-NPRSTV-Z]\d\b',
    re.IGNORECASE
)  # Canadian postal
CARD_RE = re.compile(r'\b(?:\d[ -]*?){13,19}\b')

# Very simple hospital name heuristic (improve later)
HOSPITAL_RE = re.compile(
    r'\b(?:animal|vet|veterinary|hospital|clinic|pet)\s+[A-Za-z ]+'
    r'|\b[A-Za-z ]+(?:animal|vet|veterinary|hospital|clinic|pet)\b',
    re.IGNORECASE
)

# Basic "Hi, this is John" / "My name is Sarah" style name capture
INTRO_NAME_RE = re.compile(
    r'\b(?:my name is|this is|speaking,|you\'re speaking with)\s+'
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    re.IGNORECASE
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anonymize_text_basic(text: str) -> str:
    if not text:
        return text

    # 1) Obvious structured stuff first
    text = PHONE_RE.sub("[PHONE]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = POSTAL_RE.sub("[POSTAL_CODE]", text)

    # Credit/debit-style long digit chunks (be conservative)
    def mask_card(m):
        digits = re.sub(r'\D', '', m.group(0))
        if 13 <= len(digits) <= 19:
            return "[CARD]"
        return m.group(0)

    text = CARD_RE.sub(mask_card, text)

    # 2) Hospital / clinic names
    text = HOSPITAL_RE.sub("[HOSPITAL]", text)

    # 3) Caller/staff first names in typical intro phrases
    def mask_intro(m):
        full = m.group(0)
        name = m.group(1)
        return full.replace(name, "[PERSON]")

    text = INTRO_NAME_RE.sub(mask_intro, text)

    # Further improvements later:
    # - Use spaCy / NER for PERSON/ORG
    # - Use custom dictionary of known hospital names

    return text


import os
import sys
import logging
import pyodbc
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    logger = logging.getLogger("AnonymizeTranscripts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("anonymize_callrail_transcripts.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


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


def main():
    logger = setup_logging()
    conn_str = build_sql_conn_str(logger)

    src_table = "dbo.CallRailAPI"
    dst_table = "dbo.CallRailAPI_TranscriptAnonymized"
    version = "v1_regex_only"

    with pyodbc.connect(conn_str) as conn:
        cur = conn.cursor()

        # Pick up only calls with a transcript that we haven't anonymized yet (or version changed)
        cur.execute(f"""
            SELECT c.id, c.transcription
            FROM {src_table} c
            LEFT JOIN {dst_table} a ON a.id = c.id
            WHERE c.transcription IS NOT NULL
              AND (a.id IS NULL OR a.anonymization_version <> ?)
        """, (version,))

        rows = cur.fetchall()
        logger.info("Found %d transcripts needing anonymization", len(rows))

        if not rows:
            return

        # Upsert into anonymized table
        for rid, raw in rows:
            raw_text = raw or ""
            anon = anonymize_text_basic(raw_text)
            raw_hash = sha256_text(raw_text)

            # MERGE-light pattern: try update, if rowcount==0 then insert
            cur.execute(f"""
                UPDATE {dst_table}
                SET transcript_raw_sha256 = ?, transcript_anonymized = ?,
                    anonymization_version = ?, updated_at = SYSUTCDATETIME()
                WHERE id = ?
            """, (raw_hash, anon, version, rid))

            if cur.rowcount == 0:
                cur.execute(f"""
                    INSERT INTO {dst_table}
                        (id, transcript_raw_sha256, transcript_anonymized,
                         anonymization_version, created_at)
                    VALUES (?, ?, ?, ?, SYSUTCDATETIME())
                """, (rid, raw_hash, anon, version))

        conn.commit()
        logger.info("Anonymization finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Anonymization failed: {e}")
        sys.exit(1)
