#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CallRail Historical ETL -> SQL Server (list-endpoint transcripts only)

- Pulls historical CallRail calls
- Gets transcription directly from the list endpoint via fields=transcription
- Normalizes transcription into transcript_text, then into SQL [transcription]
- Paginates with progress
- Cleans/normalizes fields (phones, transcription, milestone flattening)
- Coerces values by SQL metadata
- UPSERT via UPDATE after shell INSERT of missing IDs
- Uses TRY_CONVERT in UPDATE for numeric/temporal/bit columns to avoid TDS type errors

Environment (read-only; keep your .env as-is):
  CALLRAIL_API_KEY
  CALLRAIL_ACCOUNT or CallRail_Account or CALLRAIL_ACCOUNT_ID
  SQLSERVER_SERVER or SQL_HOST
  SQLSERVER_PORT (default 1433)
  SQLSERVER_DATABASE or SQL_DATABASE
  SQLSERVER_UID or SQL_USERNAME
  SQLSERVER_PWD or SQL_PASSWORD
  SQLSERVER_ENCRYPT (yes/no; default yes)
  SQLSERVER_TRUST_SERVER_CERT (yes/no; default yes)
  SQLSERVER_TABLE (default CallRailAPI)
"""
import os, sys, re, json, time, math, logging, argparse
from datetime import datetime, date, timedelta

import requests
import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# ------------- Logging -------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("CallRailETL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("callrail_etl.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ------------- Config -------------
def get_callrail_config():
    api_key = os.getenv("CALLRAIL_API_KEY")
    account = (
        os.getenv("CALLRAIL_ACCOUNT")
        or os.getenv("CallRail_Account")
        or os.getenv("CALLRAIL_ACCOUNT_ID")
    )
    if not api_key:
        raise RuntimeError("CALLRAIL_API_KEY not set")
    if not account:
        raise RuntimeError("CALLRAIL_ACCOUNT / CallRail_Account / CALLRAIL_ACCOUNT_ID not set")
    base_url = f"https://api.callrail.com/v3/a/{account}"
    headers = {"Authorization": f"Token token={api_key}"}
    return base_url, headers


def build_sql_conn_str(logger: logging.Logger) -> str:
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
            f"Connection Timeout=15;"
        )
        user_label = uid
    else:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER=tcp:{server},{port};"
            f"DATABASE={database};Trusted_Connection=yes;"
            f"Encrypt={encrypt};TrustServerCertificate={trust};"
            f"Connection Timeout=15;"
        )
        user_label = "(Trusted_Connection)"

    logger.info(
        "Connecting to SQL Server tcp:%s,%s, db=%s, user=%s, Encrypt=%s, TrustServerCertificate=%s",
        server,
        port,
        database,
        user_label,
        encrypt,
        trust,
    )
    return conn_str

# ------------- Extract (list endpoint) -------------
FIELDS = [
    "id",
    "company_id",
    "company_name",
    "first_call",
    "call_type",
    "speaker_percent",
    "total_calls",
    "prior_calls",
    "milestones",
    "answered",
    "direction",
    "duration",
    "recording",
    "recording_duration",
    "recording_player",
    "start_time",
    "tracking_phone_number",
    "business_phone_number",
    "customer_phone_number",
    "customer_name",
    "customer_city",
    "customer_state",
    "customer_country",
    "voicemail",
    "transcription",
    "sentiment",
    "timeline_url",
]
PER_PAGE = 100


def get_callrail_data(base_url: str, headers: dict, start_date: str, end_date: str) -> pd.DataFrame:
    all_calls = []
    page = 1
    total_pages = None
    using_fields = True

    while True:
        url = f"{base_url}/calls.json"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "page": page,
            "per_page": PER_PAGE,
            "fields": ",".join(FIELDS),
        }

        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code == 400 and using_fields:
            # Fallback for accounts/plans that don't accept fields=
            using_fields = False
            params.pop("fields", None)
            r = requests.get(url, headers=headers, params=params, timeout=60)

        r.raise_for_status()
        data = r.json()
        calls = data.get("calls", [])
        all_calls.extend(calls)

        if total_pages is None:
            total_pages = int(data.get("total_pages", 1))
            print(f"Total pages to fetch: {total_pages}")
        print(f"Fetched page {page}/{total_pages} - {len(calls)} calls")

        if page >= total_pages or not calls:
            break

        page += 1
        time.sleep(0.2)

    print(f"Total calls fetched: {len(all_calls)}")
    return pd.DataFrame(all_calls)

# ------------- Transform helpers -------------
def flatten_speaker_percent(df: pd.DataFrame) -> pd.DataFrame:
    if "speaker_percent" not in df.columns:
        return df
    out = df.copy()

    out["agent_percent"] = out["speaker_percent"].apply(
        lambda x: x.get("agent") if isinstance(x, dict) else None
    )
    out["customer_percent"] = out["speaker_percent"].apply(
        lambda x: x.get("customer") if isinstance(x, dict) else None
    )
    return out


def _g(d, k):
    return d.get(k) if isinstance(d, dict) else None


def flatten_milestones(df: pd.DataFrame) -> pd.DataFrame:
    if "milestones" not in df.columns:
        return df
    out = df.copy()

    out["first_touch_medium"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "medium")
    )
    out["first_touch_source"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "source")
    )
    out["first_touch_landing"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "landing")
    )
    out["first_touch_referrer"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "referrer")
    )
    out["first_touch_campaign"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "campaign")
    )
    out["first_touch_keywords"] = out["milestones"].apply(
        lambda x: _g(_g(x, "first_touch") or {}, "keywords")
    )

    out["last_touch_medium"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "medium")
    )
    out["last_touch_source"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "source")
    )
    out["last_touch_landing"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "landing")
    )
    out["last_touch_referrer"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "referrer")
    )
    out["last_touch_campaign"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "campaign")
    )
    out["last_touch_keywords"] = out["milestones"].apply(
        lambda x: _g(_g(x, "last_touch") or {}, "keywords")
    )
    return out


def extract_date_time(df: pd.DataFrame) -> pd.DataFrame:
    if "start_time" not in df.columns:
        return df
    out = df.copy()

    def parse_ts(s):
        try:
            if pd.isna(s):
                return None, None
            dt = pd.to_datetime(s)
            return dt.date(), int(dt.hour)
        except Exception:
            return None, None

    parsed = out["start_time"].apply(parse_ts)
    out["Date"] = [d for d, _ in parsed]
    out["Time"] = [h for _, h in parsed]
    return out


def clean_phone_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def clean(phone):
        if pd.isna(phone) or phone in ("", None):
            return None
        digits = re.sub(r"[^0-9]", "", str(phone))
        if len(digits) == 10:
            return "+1" + digits
        if len(digits) == 11 and digits.startswith("1"):
            return "+" + digits
        if len(digits) > 11:
            return "+1" + digits[-10:]
        return None if len(digits) == 0 else f"INVALID_{digits}"

    for c in [
        "customer_phone_number",
        "business_phone_number",
        "tracking_phone_number",
    ]:
        if c in out.columns:
            out[f"{c}_clean"] = out[c].apply(clean)
    return out


def normalize_transcription(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize whatever is in 'transcription' into 'transcript_text'.
    - If transcription is dict, prefer .text or .transcript
    - If string, keep string
    """
    out = df.copy()

    def norm(t):
        if isinstance(t, dict):
            return t.get("text") or t.get("transcript") or None
        if isinstance(t, str):
            return t
        return None

    if "transcription" not in out.columns:
        out["transcription"] = None

    out["transcript_text"] = out["transcription"].apply(norm)
    return out

# ------------- SQL type coercion & metadata -------------
def to_sql_friendly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def pyval(x):
        if isinstance(x, (dict, list)):
            return json.dumps(x, separators=(",", ":"))
        if isinstance(x, str) and x.strip() in {"", "NaN", "NaT", "nan"}:
            return None
        return None if pd.isna(x) else x

    for c in out.columns:
        out[c] = out[c].apply(pyval)

    # booleans
    for b in ["answered", "first_call", "voicemail"]:
        if b in out.columns:
            out[b] = out[b].map(
                {
                    "TRUE": True,
                    "FALSE": False,
                    "True": True,
                    "False": False,
                    "true": True,
                    "false": False,
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                }
            )

    # numerics
    for c in [
        "duration",
        "recording_duration",
        "agent_percent",
        "customer_percent",
        "total_calls",
        "prior_calls",
        "Time",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].where(pd.notnull(out[c]), None)

    # rounding
    if "recording_duration" in out.columns:
        out["recording_duration"] = out["recording_duration"].apply(
            lambda v: round(float(v), 2) if v is not None else None
        )
    for c in ("agent_percent", "customer_percent"):
        if c in out.columns:
            out[c] = out[c].apply(
                lambda v: round(float(v), 2) if v is not None else None
            )

    # datetime and date
    if "start_time" in out.columns:
        out["start_time"] = out["start_time"].astype(str).apply(
            lambda s: None if s in ("None", "NaT", "nan") else s
        )
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date

    return out.where(pd.notnull(out), None)


PHONE_COLUMNS = {
    "customer_phone_number",
    "business_phone_number",
    "tracking_phone_number",
    "customer_phone_number_clean",
    "business_phone_number_clean",
    "tracking_phone_number_clean",
}


def fetch_table_metadata(conn, table_name):
    schema = "dbo"
    tbl = table_name.split(".")[-1]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                   NUMERIC_PRECISION, NUMERIC_SCALE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
            (schema, tbl),
        )
        rows = cur.fetchall()

    meta = {}
    for col, dtype, isnull, prec, scale, charlen in rows:
        dtype = (dtype or "").lower()
        if dtype in {"decimal", "numeric"} and prec is not None and scale is not None:
            sql_decl = f"{dtype}({int(prec)},{int(scale)})"
        elif dtype in {"varchar", "nvarchar", "char", "nchar"} and charlen and int(charlen) > 0:
            sql_decl = f"{dtype}({int(charlen)})"
        else:
            sql_decl = dtype
        meta[col] = {
            "data_type": dtype,
            "nullable": (isnull == "YES"),
            "precision": prec,
            "scale": scale,
            "charlen": charlen,
            "sql_decl": sql_decl,
        }
    return meta


def coerce_by_metadata(val, m):
    if val is None:
        return None
    t = m["data_type"]

    # bit
    if t == "bit":
        try:
            return bool(int(val)) if isinstance(val, str) and val.isdigit() else bool(val)
        except Exception:
            return None

    # integers
    if t in {"tinyint", "smallint", "int", "bigint"}:
        try:
            v = int(float(val))  # tolerate "3.0"
            if t == "tinyint" and not (0 <= v <= 255):
                return None
            return v
        except Exception:
            return None

    # decimals / numerics
    if t in {"decimal", "numeric"}:
        try:
            v = float(val)
            if math.isinf(v) or pd.isna(v):
                return None
            scale = int(m.get("scale") or 0)
            return round(v, scale)
        except Exception:
            return None

    # floats/real
    if t in {"float", "real"}:
        try:
            v = float(val)
            if math.isinf(v) or pd.isna(v):
                return None
            return v
        except Exception:
            return None

    # dates
    if t == "date":
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return None

    # datetime types -> ISO string
    if t in {"datetime", "datetime2", "datetimeoffset", "smalldatetime"}:
        s = str(val)
        return None if s in ("None", "NaT", "nan") else s

    # strings
    return str(val)


def existing_columns(conn, table_name):
    schema = "dbo"
    tbl = table_name.split(".")[-1]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=? AND TABLE_NAME=?
        """,
            (schema, tbl),
        )
        return {r[0] for r in cur.fetchall()}


def reconcile_dataframe(conn, table_name, df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    meta = fetch_table_metadata(conn, table_name)
    cols = set(meta.keys())
    out = df.copy()

    # transcript_text -> transcription if present
    if "transcript_text" in out.columns and "transcription" in cols:
        if "transcription" not in out.columns:
            out["transcription"] = None
        out["transcription"] = out["transcription"].where(
            out["transcription"].notna(), out["transcript_text"]
        )
        out.drop(columns=["transcript_text"], inplace=True, errors="ignore")

    # Keep only existing columns
    keep = [c for c in out.columns if c in cols]
    if "id" not in keep:
        keep = ["id"] + keep
    out = out[keep].copy()

    # Coerce per metadata
    for c in keep:
        if c == "id":
            continue
        out[c] = out[c].apply(lambda v: coerce_by_metadata(v, meta[c]))

    # If destination column for any phone field isn't string-typed, hard set to None
    string_types = {"varchar", "nvarchar", "char", "nchar", "text", "ntext"}
    for c in [col for col in keep if col in PHONE_COLUMNS]:
        if meta[c]["data_type"] not in string_types:
            out[c] = None

    logger.info("UPDATE columns order: %s", ", ".join([c for c in keep if c != "id"]))
    return out

# ------------- Load (UPSERT) -------------
def get_existing_ids(conn, table_name, ids):
    if not ids:
        return set()
    res = set()
    chunk = 500
    with conn.cursor() as cur:
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            q = ",".join(["?"] * len(part))
            cur.execute(
                f"SELECT [id] FROM {table_name} WITH (NOLOCK) WHERE [id] IN ({q})",
                part,
            )
            res.update([r[0] for r in cur.fetchall()])
    return res


def upsert_dataframe(conn_str: str, table_name: str, df: pd.DataFrame, logger: logging.Logger, label: str):
    if df.empty:
        return 0, 0

    with pyodbc.connect(conn_str) as conn:
        cur = conn.cursor()
        df2 = reconcile_dataframe(conn, table_name, df, logger)

        ids = df2["id"].dropna().astype(str).tolist()
        have = get_existing_ids(conn, table_name, ids)
        missing = [i for i in ids if i not in have]
        if missing:
            cur.fast_executemany = True
            cur.executemany(
                f"INSERT INTO {table_name} ([id]) VALUES (?)",
                [(m,) for m in missing],
            )
            conn.commit()
            logger.info("%s: Inserted %d new id shells", label, len(missing))

        cols = [c for c in df2.columns if c != "id"]
        if not cols:
            return len(missing), 0

        meta = fetch_table_metadata(conn, table_name)
        string_types = {"varchar", "nvarchar", "char", "nchar", "text", "ntext", "xml"}

        def assign_expr(c):
            dt = meta[c]["data_type"]
            decl = meta[c]["sql_decl"]
            if dt in string_types:
                # Force NVARCHAR binding; TRY_CONVERT to respect length and avoid crashes
                return f"[{c}] = TRY_CONVERT({decl}, CAST(? AS {decl}))"
            if decl == "":
                return f"[{c}] = ?"
            return f"[{c}] = TRY_CONVERT({decl}, ?)"

        set_clause = ", ".join([assign_expr(c) for c in cols])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE [id] = ?"

        def prep_value(x):
            if x is None:
                return None
            s = str(x)
            return None if s.strip() == "" else s

        params = []
        for _, r in df2.iterrows():
            row = [prep_value(r.get(c, None)) for c in cols]
            row.append(r["id"])
            params.append(tuple(row))

        # Disable fast_executemany to avoid weird type inference on mixed columns
        cur.fast_executemany = False
        updated = 0
        for row in params:
            try:
                cur.execute(sql, row)
                updated += 1
            except pyodbc.Error:
                # Diagnose specific column
                for idx, c in enumerate(cols):
                    probe = list(row)
                    probe[idx] = None
                    try:
                        cur.execute(sql, tuple(probe))
                        conn.commit()
                        logger.error(
                            "%s: Column '%s' caused failure. Value=%r (SQL type %s)",
                            label,
                            c,
                            row[idx],
                            meta[c]["sql_decl"],
                        )
                        break
                    except Exception:
                        continue
                raise
        conn.commit()
        return len(missing), updated

# ------------- Orchestrate -------------
def process_range(base_url, headers, conn_str, table_name, start_date, end_date, label, logger):
    logger.info("Processing %s: %s to %s", label, start_date, end_date)
    df = get_callrail_data(base_url, headers, start_date, end_date)
    if df.empty:
        logger.info("%s: No data found", label)
        return 0, 0
    logger.info("%s: Extracted %d records", label, len(df))

    # 1) Normalize any transcription we get from list endpoint
    df = normalize_transcription(df)

    # 2) Rest of your transform pipeline
    df = flatten_speaker_percent(df)
    df = flatten_milestones(df)
    df = extract_date_time(df)
    df["script_run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = clean_phone_numbers(df)
    df = to_sql_friendly(df)

    logger.info("%s: Data processing completed", label)

    ins, upd = upsert_dataframe(conn_str, table_name, df, logger, label)
    logger.info("%s: Completed - Inserted: %d, Updated: %d", label, ins, upd)
    return ins, upd


def main():
    logger = setup_logging()
    ap = argparse.ArgumentParser(
        description="CallRail Historical ETL -> SQL Server (list-endpoint transcripts only)"
    )
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--table", default=os.getenv("SQLSERVER_TABLE", "CallRailAPI"))
    args = ap.parse_args()

    base_url, headers = get_callrail_config()
    conn_str = build_sql_conn_str(logger)
    table = args.table

    # Date handling:
    # - If BOTH --start and --end are provided, use them.
    # - If NEITHER is provided, default to last 90 days ending today.
    # - If only one is provided, fail fast (avoids accidental huge/odd ranges in SQL Agent).
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
        label = f"{args.start}..{args.end}"
    elif (args.start is None) and (args.end is None):
        today = date.today()
        start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        label = f"{start_date}..{end_date}"
        logger.info(f"No --start/--end provided; defaulting to last 90 days: {label}")
    else:
        logger.error("You must provide BOTH --start and --end, or neither (to use the 90-day default).")
        raise SystemExit(2)

    process_range(
        base_url,
        headers,
        conn_str,
        table,
        start_date,
        end_date,
        label,
        logger,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ETL failed: {e}")
        sys.exit(1)
