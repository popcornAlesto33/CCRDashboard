#!/usr/bin/env python3
"""
Run A1b pipeline on full calldata with rate limiting, parallel processing,
checkpointing, and cost tracking.

Usage:
  python Scripts/run_a1b_production.py                    # process all unprocessed calls
  python Scripts/run_a1b_production.py --limit 5000       # process first 5000
  python Scripts/run_a1b_production.py --resume            # resume from checkpoint
  python Scripts/run_a1b_production.py --dry-run           # show counts, don't call API

Rate limits (Gemini 2.5 Flash):
  - 1,000,000 tokens per minute (TPM) — binding constraint
  - 1,000 requests per minute (RPM)
  - 10,000 requests per day (RPD)
"""

import os
import sys
import csv
import json
import argparse
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from openai import OpenAI

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("production")

# ============================================================
# IMPORT A1b PROMPTS AND SCHEMAS
# ============================================================

sys.path.insert(0, script_dir)
from validate_prompt_engineering_vA1b import (
    V9_APPOINTMENT_BOOKED_PROMPT,
    V9_APPOINTMENT_BOOKED_SCHEMA,
    A1B_COMBO_PROMPT,
    A1B_COMBO_SCHEMA,
    TREATMENT_TYPE_ENUMS,
    REASON_NOT_BOOKED_ENUMS,
)

# ============================================================
# PATHS
# ============================================================

CALLDATA_PATH = os.path.join(project_dir, "CallData", "VetCare_CallInsight_CallData - calldata.csv")
LABELS_PATH = os.path.join(project_dir, "CallData", "VetCare_CallInsight_Labels - labels.csv")
OUTPUT_PATH = os.path.join(project_dir, "CallData", "model_predictions.csv")
CHECKPOINT_PATH = os.path.join(project_dir, "CallData", "production_checkpoint.json")

OUTPUT_FIELDS = [
    "id", "appointment_booked", "client_type", "treatment_type",
    "reason_not_booked", "stated_hospital_name", "stated_patient_name",
    "agent_name", "labeled_by", "labeled_at",
]

# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Token bucket rate limiter for TPM, RPM, and RPD."""

    def __init__(self, tpm=1_000_000, rpm=1_000, rpd_remaining=10_000):
        self._lock = threading.Lock()

        # TPM: tokens per minute
        self._tpm_limit = tpm
        self._tpm_tokens = tpm  # start full
        self._tpm_last_refill = time.monotonic()

        # RPM: requests per minute
        self._rpm_limit = rpm
        self._rpm_count = 0
        self._rpm_window_start = time.monotonic()

        # RPD: requests per day
        self._rpd_remaining = rpd_remaining
        self._rpd_used = 0

        # Tracking
        self._total_tokens = 0
        self._total_requests = 0

    def acquire(self, estimated_tokens=30_000):
        """Block until we can make a request within rate limits."""
        while True:
            with self._lock:
                now = time.monotonic()

                # Refill TPM bucket
                elapsed = now - self._tpm_last_refill
                if elapsed >= 1.0:
                    refill = int(self._tpm_limit * elapsed / 60)
                    self._tpm_tokens = min(self._tpm_limit, self._tpm_tokens + refill)
                    self._tpm_last_refill = now

                # Reset RPM window
                if now - self._rpm_window_start >= 60:
                    self._rpm_count = 0
                    self._rpm_window_start = now

                # Check all limits
                can_proceed = (
                    self._tpm_tokens >= estimated_tokens
                    and self._rpm_count < self._rpm_limit
                    and self._rpd_remaining > 0
                )

                if can_proceed:
                    self._tpm_tokens -= estimated_tokens
                    self._rpm_count += 1
                    self._rpd_remaining -= 1
                    self._rpd_used += 1
                    self._total_requests += 1
                    return

            # Wait and retry
            time.sleep(0.5)

    def record_tokens(self, actual_tokens):
        """Record actual token usage after a call completes."""
        with self._lock:
            self._total_tokens += actual_tokens

    @property
    def stats(self):
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "rpd_remaining": self._rpd_remaining,
                "rpd_used": self._rpd_used,
            }

    def check_rpd(self):
        """Return True if we have RPD budget remaining."""
        with self._lock:
            return self._rpd_remaining > 0


# ============================================================
# API CALLS WITH RATE LIMITING
# ============================================================

def call_batch_rated(client, model, calls_batch, system_prompt, schema, rate_limiter, step):
    """Batched API call with rate limiting."""
    # Estimate tokens: ~1800 tokens per call in batch
    estimated_tokens = len(calls_batch) * 1800

    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["transcript"]}
        for c in calls_batch
    ])
    batched_prompt = system_prompt.replace(
        'Return JSON ONLY.',
        'You will receive multiple transcripts as a JSON array. '
        'Process EACH transcript independently — do not let one transcript '
        'influence your classification of another.\n\n'
        'Return JSON: {"results": [{"call_id": "...", ...}, ...]}\n'
        'Return JSON ONLY.'
    )

    # Build batched schema
    single_props = schema["json_schema"]["schema"]["properties"].copy()
    single_req = list(schema["json_schema"]["schema"]["required"])
    item_props = {**single_props, "call_id": {"type": "string"}}
    item_required = single_req + ["call_id"]
    batched_schema = {
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
                            "properties": item_props,
                            "required": item_required,
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            }
        }
    }

    messages = [
        {"role": "system", "content": batched_prompt},
        {"role": "user", "content": payload},
    ]

    for attempt in range(3):
        rate_limiter.acquire(estimated_tokens)
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                response_format=batched_schema,
            )
            actual_tokens = 0
            if resp.usage:
                actual_tokens = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
            rate_limiter.record_tokens(actual_tokens)

            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(it.get("call_id", "")): it for it in items}

        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 2.0
                logger.warning(f"  {step} batch attempt {attempt+1}: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  {step} batch failed after 3 attempts: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}


# ============================================================
# A1b PIPELINE (CHUNK)
# ============================================================

def process_chunk(client, model, calls, batch_size, rate_limiter, chunk_num):
    """Process a chunk of calls through the A1b pipeline."""
    logger.info(f"Chunk {chunk_num}: processing {len(calls)} calls")

    # Call 1: appointment_booked
    batches = [calls[i:i + batch_size] for i in range(0, len(calls), batch_size)]
    appt_results = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                call_batch_rated, client, model, batch,
                V9_APPOINTMENT_BOOKED_PROMPT, V9_APPOINTMENT_BOOKED_SCHEMA,
                rate_limiter, f"chunk{chunk_num}_call1",
            ): i
            for i, batch in enumerate(batches)
        }
        for f in as_completed(futures):
            try:
                appt_results.update(f.result())
            except Exception as e:
                logger.error(f"  Chunk {chunk_num} call1 error: {e}")

    logger.info(f"  Chunk {chunk_num} Call 1 done: {len(appt_results)} results")

    # Group by appointment answer
    calls_by_appt = {"Yes": [], "No": [], "Inconclusive": []}
    for c in calls:
        answer = appt_results.get(c["id"], {}).get("answer", "")
        if answer in calls_by_appt:
            calls_by_appt[answer].append(c)

    # Call 2: combo for each group
    combo_results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for appt_answer, group_calls in calls_by_appt.items():
            if not group_calls:
                continue
            prompt = A1B_COMBO_PROMPT.replace("{appointment_booked}", appt_answer)
            group_batches = [group_calls[i:i + batch_size] for i in range(0, len(group_calls), batch_size)]

            for i, batch in enumerate(group_batches):
                futures[pool.submit(
                    call_batch_rated, client, model, batch,
                    prompt, A1B_COMBO_SCHEMA,
                    rate_limiter, f"chunk{chunk_num}_call2_{appt_answer}",
                )] = (appt_answer, i)

        for f in as_completed(futures):
            try:
                combo_results.update(f.result())
            except Exception as e:
                logger.error(f"  Chunk {chunk_num} call2 error: {e}")

    logger.info(f"  Chunk {chunk_num} Call 2 done: {len(combo_results)} results")

    # Assemble predictions
    predictions = []
    now = datetime.utcnow().isoformat()
    for c in calls:
        cid = c["id"]
        appt = appt_results.get(cid, {}).get("answer")
        combo = combo_results.get(cid, {})
        reason = combo.get("reason_not_booked")

        # Cross-field consistency
        if appt == "Yes" and reason is not None:
            reason = None

        predictions.append({
            "id": cid,
            "appointment_booked": appt or "",
            "client_type": combo.get("client_type") or "",
            "treatment_type": combo.get("treatment_type") or "",
            "reason_not_booked": reason or "",
            "stated_hospital_name": "",
            "stated_patient_name": "",
            "agent_name": "",
            "labeled_by": "a1b_flash",
            "labeled_at": now,
        })

    return predictions


# ============================================================
# DATA LOADING
# ============================================================

def load_calldata():
    """Load call transcripts from calldata CSV."""
    calls = {}
    with open(CALLDATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            transcript = row.get("transcription", "").strip()
            if transcript and transcript != "NULL":
                calls[row["id"]] = {
                    "id": row["id"],
                    "transcript": transcript,
                    "company": row.get("company_name", ""),
                    "call_type": row.get("call_type", ""),
                }
    return calls


def load_completed_ids():
    """Load IDs already in labels + output + checkpoint."""
    completed = set()

    # Gold labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                completed.add(row["id"])

    # Already predicted
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                completed.add(row["id"])

    # Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        completed.update(checkpoint.get("completed_ids", []))

    return completed


def save_checkpoint(completed_ids, stats):
    """Save progress checkpoint."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "completed_ids": list(completed_ids),
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
        }, f)


def append_results(predictions):
    """Append predictions to output CSV."""
    file_exists = os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run A1b on full calldata")
    parser.add_argument("--limit", type=int, default=0, help="Max calls to process (0=all)")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Calls per checkpoint chunk")
    parser.add_argument("--batch-size", type=int, default=15, help="Transcripts per API call")
    parser.add_argument("--rpd-remaining", type=int, default=5380, help="Remaining daily request quota")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only, no API calls")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use")
    args = parser.parse_args()

    # Load data
    logger.info("Loading calldata...")
    all_calls = load_calldata()
    logger.info(f"Loaded {len(all_calls):,} calls with transcripts")

    completed_ids = load_completed_ids()
    logger.info(f"Already completed: {len(completed_ids):,}")

    # Filter to unprocessed
    pending = {cid: c for cid, c in all_calls.items() if cid not in completed_ids}
    logger.info(f"Pending: {len(pending):,}")

    if args.limit > 0:
        pending_ids = sorted(pending.keys())[:args.limit]
        pending = {cid: pending[cid] for cid in pending_ids}
        logger.info(f"Limited to: {len(pending):,}")

    if args.dry_run:
        batches = len(pending) / args.batch_size
        requests = batches * 2
        tokens_est = len(pending) * 1830
        cost_est = tokens_est / 1_000_000 * 0.30  # blended rate
        print(f"\n  Calls to process: {len(pending):,}")
        print(f"  Batches (bs={args.batch_size}): {batches:,.0f}")
        print(f"  API requests: {requests:,.0f}")
        print(f"  Estimated tokens: {tokens_est:,.0f}")
        print(f"  Estimated cost: ${cost_est:.2f} USD")
        print(f"  Estimated time: {len(pending) / 547:.0f} minutes")
        print(f"  RPD remaining: {args.rpd_remaining}")
        max_calls_rpd = (args.rpd_remaining / 2) * args.batch_size
        if requests > args.rpd_remaining:
            print(f"  ⚠  RPD limit: can only process ~{max_calls_rpd:,.0f} calls today")
        return

    if not pending:
        logger.info("Nothing to process!")
        return

    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)

    rate_limiter = RateLimiter(
        tpm=1_000_000,
        rpm=1_000,
        rpd_remaining=args.rpd_remaining,
    )

    # Process in chunks
    pending_list = sorted(pending.values(), key=lambda c: c["id"])
    chunks = [pending_list[i:i + args.chunk_size] for i in range(0, len(pending_list), args.chunk_size)]

    logger.info(f"Processing {len(pending):,} calls in {len(chunks)} chunks of {args.chunk_size}")
    total_processed = 0

    for chunk_num, chunk_calls in enumerate(chunks, 1):
        # Check RPD before starting chunk
        if not rate_limiter.check_rpd():
            logger.warning(f"RPD limit reached after {total_processed:,} calls. Resume tomorrow with --resume")
            break

        predictions = process_chunk(
            client, args.model, chunk_calls, args.batch_size,
            rate_limiter, chunk_num,
        )

        # Save results
        append_results(predictions)
        total_processed += len(predictions)

        # Update checkpoint
        new_completed = completed_ids | {p["id"] for p in predictions}
        stats = rate_limiter.stats
        stats["total_processed"] = total_processed
        stats["cost_estimate_usd"] = stats["total_tokens"] / 1_000_000 * 0.30
        save_checkpoint(new_completed, stats)
        completed_ids = new_completed

        logger.info(f"Chunk {chunk_num}/{len(chunks)} done. "
                     f"Processed: {total_processed:,} | "
                     f"RPD used: {stats['rpd_used']} | "
                     f"Tokens: {stats['total_tokens']:,} | "
                     f"Cost: ${stats['cost_estimate_usd']:.4f}")

    # Final summary
    stats = rate_limiter.stats
    logger.info(f"\n{'='*60}")
    logger.info(f"PRODUCTION RUN COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total processed: {total_processed:,}")
    logger.info(f"Total API requests: {stats['total_requests']:,}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")
    logger.info(f"Estimated cost: ${stats['total_tokens'] / 1_000_000 * 0.30:.2f} USD")
    logger.info(f"RPD remaining: {stats['rpd_remaining']}")
    logger.info(f"Output: {OUTPUT_PATH}")
    if stats["rpd_remaining"] <= 0:
        logger.info(f"⚠  RPD exhausted. Run again tomorrow with --resume to continue.")


if __name__ == "__main__":
    main()
