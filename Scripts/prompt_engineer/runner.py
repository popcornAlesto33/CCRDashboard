#!/usr/bin/env python3
"""Prompt Engineer Runner — generic LLM classification testing & evaluation.

Handles API calls, batching, threading, scoring, cost tracking, and convergence
detection. Driven by prompt-engineer.yaml config.

Usage:
    python runner.py --config prompt-engineer.yaml --phase baseline \
        --field appointment_booked --model gemini-2.5-flash \
        --batch-size 15 --max-calls 50 --output results.json

    python runner.py --config prompt-engineer.yaml --phase evaluate \
        --results results.json --field appointment_booked

    python runner.py --config prompt-engineer.yaml --phase audit \
        --output audit_flagged.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # Fallback: support JSON config if pyyaml not installed

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

from error_analysis import categorize_error, summarize_errors, build_opro_error_context
from gold_label_audit import audit_dataset, analyze_input, generate_report, export_csv

logger = logging.getLogger("prompt_engineer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load and validate config (YAML or JSON)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("pyyaml required for YAML configs. Install with: pip install pyyaml")
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # Defaults
    config.setdefault("output_dir", "prompt_engineer_output")
    config.setdefault("budget", {})
    config["budget"].setdefault("max_total_usd", 10.0)
    config["budget"].setdefault("warn_at_usd", 5.0)
    config["budget"].setdefault("max_per_phase_usd", 3.0)
    config.setdefault("convergence", {})
    config["convergence"].setdefault("min_rounds", 3)
    config["convergence"].setdefault("max_rounds", 15)
    config["convergence"].setdefault("plateau_rounds", 3)
    config["convergence"].setdefault("plateau_threshold", 1.0)
    config["convergence"].setdefault("sample_sizes", {})
    config["convergence"]["sample_sizes"].setdefault("baseline", 50)
    config["convergence"]["sample_sizes"].setdefault("iteration", 100)
    config["convergence"]["sample_sizes"].setdefault("confirmation", 200)

    return config


def get_field_config(config: dict, field_name: str) -> dict:
    """Get field configuration by name."""
    for f in config.get("fields", []):
        if f["name"] == field_name:
            return f
    raise ValueError(f"Field '{field_name}' not found in config. Available: {[f['name'] for f in config.get('fields', [])]}")


def get_model_config(config: dict, model_name: str) -> dict:
    """Get model configuration by name."""
    for m in config.get("models", []):
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in config. Available: {[m['name'] for m in config.get('models', [])]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(config: dict) -> tuple[dict[str, str], dict[str, dict]]:
    """Load inputs and gold labels from config paths.

    Returns:
        (inputs, labels) where:
        - inputs: {id: text}
        - labels: {id: {field_name: value, ...}}
    """
    data_cfg = config["data"]
    inputs = _load_inputs(data_cfg)
    labels = _load_labels(data_cfg)

    # Only keep items that have both input and labels
    common_ids = set(inputs.keys()) & set(labels.keys())
    inputs = {k: v for k, v in inputs.items() if k in common_ids}
    labels = {k: v for k, v in labels.items() if k in common_ids}

    logger.info(f"Loaded {len(inputs)} items with both input and labels")
    return inputs, labels


def _load_inputs(data_cfg: dict) -> dict[str, str]:
    """Load input texts from CSV or JSON."""
    path = data_cfg["inputs_path"]
    id_col = data_cfg.get("id_column", "id")
    text_col = data_cfg.get("text_column", "transcript")
    fmt = data_cfg.get("inputs_format", Path(path).suffix.lstrip("."))

    inputs = {}
    if fmt == "csv":
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                # Try to find columns by name
                try:
                    id_idx = header.index(id_col)
                    text_idx = header.index(text_col)
                except ValueError:
                    # Fall back to positional (id=0, text=1)
                    id_idx, text_idx = 0, 1
                for row in reader:
                    if len(row) > max(id_idx, text_idx):
                        text = row[text_idx]
                        if text and text != "NULL" and text.strip():
                            inputs[row[id_idx]] = text
            else:
                for row in reader:
                    if len(row) >= 2 and row[1] != "NULL" and row[1].strip():
                        inputs[row[0]] = row[1]
    elif fmt == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                inputs[str(item[id_col])] = item[text_col]
        elif isinstance(data, dict):
            inputs = data
    else:
        raise ValueError(f"Unsupported inputs format: {fmt}")

    return inputs


def _load_labels(data_cfg: dict) -> dict[str, dict]:
    """Load gold labels from CSV or JSON."""
    path = data_cfg.get("labels_path", data_cfg["inputs_path"])
    id_col = data_cfg.get("labels_id_column", data_cfg.get("id_column", "id"))
    fmt = data_cfg.get("labels_format", Path(path).suffix.lstrip("."))

    labels = {}
    if fmt == "csv":
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                labels[row[id_col]] = dict(row)
    elif fmt == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                labels[str(item[id_col])] = item
        elif isinstance(data, dict):
            labels = data

    return labels


def sample_data(
    inputs: dict[str, str],
    labels: dict[str, dict],
    n: int,
    seed: int = 42,
    exclude_ids: set | None = None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Sample n items from inputs/labels, optionally excluding IDs (for holdout)."""
    ids = list(inputs.keys())
    if exclude_ids:
        ids = [i for i in ids if i not in exclude_ids]
    rng = random.Random(seed)
    if n < len(ids):
        ids = rng.sample(ids, n)
    return (
        {i: inputs[i] for i in ids},
        {i: labels[i] for i in ids},
    )


def split_holdout(
    inputs: dict[str, str],
    labels: dict[str, dict],
    ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split IDs into train and holdout sets. Returns (train_ids, holdout_ids)."""
    all_ids = list(inputs.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    holdout_n = int(len(all_ids) * ratio)
    holdout_ids = set(all_ids[:holdout_n])
    train_ids = set(all_ids[holdout_n:])
    return train_ids, holdout_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Budget Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class BudgetTracker:
    """Track API costs across phases and enforce budget limits."""

    def __init__(self, config: dict):
        budget_cfg = config.get("budget", {})
        self.max_total = budget_cfg.get("max_total_usd", 10.0)
        self.warn_at = budget_cfg.get("warn_at_usd", 5.0)
        self.max_per_phase = budget_cfg.get("max_per_phase_usd", 3.0)
        self.output_dir = config.get("output_dir", "prompt_engineer_output")

        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._phase_cost = 0.0
        self._model_costs: dict[str, float] = defaultdict(float)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._warned = False

        # Load existing costs if resuming
        self._cost_file = Path(self.output_dir) / "costs.json"
        if self._cost_file.exists():
            with open(self._cost_file) as f:
                saved = json.load(f)
            self._total_cost = saved.get("total_usd", 0.0)
            self._model_costs = defaultdict(float, saved.get("by_model", {}))
            self._total_input_tokens = saved.get("total_input_tokens", 0)
            self._total_output_tokens = saved.get("total_output_tokens", 0)

    def track(self, model_name: str, input_tokens: int, output_tokens: int, model_config: dict) -> float:
        """Record token usage and return cost in USD."""
        input_price = model_config.get("input_price", 0.0) / 1_000_000
        output_price = model_config.get("output_price", 0.0) / 1_000_000
        cost = input_tokens * input_price + output_tokens * output_price

        with self._lock:
            self._total_cost += cost
            self._phase_cost += cost
            self._model_costs[model_name] += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            if not self._warned and self._total_cost >= self.warn_at:
                logger.warning(f"BUDGET WARNING: ${self._total_cost:.4f} spent (warn threshold: ${self.warn_at})")
                self._warned = True

        return cost

    def check_budget(self) -> bool:
        """Return True if within budget, False if exceeded."""
        with self._lock:
            return self._total_cost < self.max_total

    def check_phase_budget(self) -> bool:
        """Return True if within per-phase budget."""
        with self._lock:
            return self._phase_cost < self.max_per_phase

    def reset_phase(self):
        """Reset phase cost counter (call at start of each phase)."""
        with self._lock:
            self._phase_cost = 0.0

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def phase_cost(self) -> float:
        with self._lock:
            return self._phase_cost

    def save(self):
        """Persist cost data to disk."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "total_usd": self._total_cost,
                "phase_usd": self._phase_cost,
                "by_model": dict(self._model_costs),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            }
        with open(self._cost_file, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        with self._lock:
            lines = [f"Total: ${self._total_cost:.4f} / ${self.max_total:.2f}"]
            lines.append(f"Phase: ${self._phase_cost:.4f} / ${self.max_per_phase:.2f}")
            lines.append(f"Tokens: {self._total_input_tokens:,} in + {self._total_output_tokens:,} out")
            for model, cost in sorted(self._model_costs.items(), key=lambda x: -x[1]):
                lines.append(f"  {model}: ${cost:.4f}")
            return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence Detection
# ═══════════════════════════════════════════════════════════════════════════════

class ConvergenceDetector:
    """Detect when OPRO optimization has plateaued."""

    def __init__(self, config: dict):
        conv_cfg = config.get("convergence", {})
        self.min_rounds = conv_cfg.get("min_rounds", 3)
        self.max_rounds = conv_cfg.get("max_rounds", 15)
        self.plateau_rounds = conv_cfg.get("plateau_rounds", 3)
        self.plateau_threshold = conv_cfg.get("plateau_threshold", 1.0)
        self._history: list[dict] = []

    def record(self, round_num: int, accuracy: float, field: str, prompt_version: str):
        """Record a round result."""
        self._history.append({
            "round": round_num,
            "accuracy": accuracy,
            "field": field,
            "prompt_version": prompt_version,
        })

    def should_stop(self, field: str) -> tuple[bool, str]:
        """Check if optimization should stop for a given field.

        Returns:
            (should_stop, reason)
        """
        field_history = [h for h in self._history if h["field"] == field]

        if len(field_history) >= self.max_rounds:
            return True, f"Max rounds ({self.max_rounds}) reached"

        if len(field_history) < self.min_rounds:
            return False, f"Below minimum rounds ({len(field_history)}/{self.min_rounds})"

        # Check for plateau
        recent = field_history[-self.plateau_rounds:]
        if len(recent) >= self.plateau_rounds:
            best_before = max(h["accuracy"] for h in field_history[:-self.plateau_rounds]) if len(field_history) > self.plateau_rounds else 0
            best_recent = max(h["accuracy"] for h in recent)
            gain = best_recent - best_before
            if gain < self.plateau_threshold:
                return True, f"Plateau: {gain:.1f}pp gain in last {self.plateau_rounds} rounds (threshold: {self.plateau_threshold}pp)"

        return False, "Continuing"

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def best_for_field(self, field: str) -> dict | None:
        """Return the best result for a field."""
        field_history = [h for h in self._history if h["field"] == field]
        if not field_history:
            return None
        return max(field_history, key=lambda h: h["accuracy"])


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Client & API Calls
# ═══════════════════════════════════════════════════════════════════════════════

_semaphores: dict[str, threading.Semaphore] = {}


def _get_semaphore(model_name: str, max_concurrent: int) -> threading.Semaphore:
    """Get or create a rate-limiting semaphore for a model."""
    if model_name not in _semaphores:
        _semaphores[model_name] = threading.Semaphore(max_concurrent)
    return _semaphores[model_name]


def make_client(model_config: dict) -> OpenAI:
    """Create an OpenAI-compatible client from model config."""
    provider = model_config.get("provider", "openai")
    api_key = os.getenv(model_config.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise ValueError(f"API key env var '{model_config.get('api_key_env')}' not set")

    # Resolve base URL
    base_url = model_config.get("base_url")
    if not base_url:
        base_url_env = model_config.get("base_url_env")
        if base_url_env:
            base_url = os.getenv(base_url_env)
    if not base_url:
        defaults = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1/",
        }
        base_url = defaults.get(provider, "https://api.openai.com/v1")

    return OpenAI(api_key=api_key, base_url=base_url, max_retries=3)


def call_llm(
    client: OpenAI,
    model_config: dict,
    call_id: str,
    input_text: str,
    system_prompt: str,
    response_schema: dict | None = None,
    budget: BudgetTracker | None = None,
) -> dict:
    """Make a single LLM call with retry and rate limiting.

    Returns:
        {"call_id": ..., "reasoning": ..., "answer": ...} or
        {"call_id": ..., "error": ...} on failure.
    """
    model_name = model_config["name"]
    semaphore = _get_semaphore(model_name, model_config.get("max_concurrent", 15))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if model_config.get("supports_temperature", True):
        kwargs["temperature"] = model_config.get("temperature", 0.0)
    if response_schema:
        kwargs["response_format"] = response_schema
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)

            # Track usage
            if budget and resp.usage:
                in_tok = resp.usage.prompt_tokens or 0
                out_tok = resp.usage.completion_tokens or 0
                budget.track(model_name, in_tok, out_tok, model_config)

            result = json.loads(resp.choices[0].message.content)
            result["call_id"] = call_id
            return result

        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 1.0
                logger.warning(f"  {model_name} {call_id} attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  {model_name} {call_id} failed after 3 attempts: {e}")
                return {"call_id": call_id, "error": str(e)}
        finally:
            semaphore.release()


def call_batch(
    client: OpenAI,
    model_config: dict,
    calls_batch: list[dict],
    system_prompt: str,
    response_schema: dict | None = None,
    budget: BudgetTracker | None = None,
) -> dict[str, dict]:
    """Send multiple inputs in a single API call (batched).

    Returns:
        {call_id: {"reasoning": ..., "answer": ...}}
    """
    model_name = model_config["name"]
    semaphore = _get_semaphore(model_name, model_config.get("max_concurrent", 15))

    payload = json.dumps([
        {"call_id": c["id"], "transcript": c["text"]}
        for c in calls_batch
    ])

    batched_prompt = _make_batched_prompt(system_prompt)
    batched_schema = _make_batched_schema(response_schema) if response_schema else None

    messages = [
        {"role": "system", "content": batched_prompt},
        {"role": "user", "content": payload},
    ]

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if model_config.get("supports_temperature", True):
        kwargs["temperature"] = model_config.get("temperature", 0.0)
    if batched_schema:
        kwargs["response_format"] = batched_schema
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        semaphore.acquire()
        try:
            resp = client.chat.completions.create(**kwargs)

            if budget and resp.usage:
                in_tok = resp.usage.prompt_tokens or 0
                out_tok = resp.usage.completion_tokens or 0
                budget.track(model_name, in_tok, out_tok, model_config)

            result = json.loads(resp.choices[0].message.content)
            items = result.get("results", [])
            return {str(item.get("call_id", "")): item for item in items}

        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 1.0
                logger.warning(f"  batch attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  batch failed after 3 attempts: {e}")
                return {c["id"]: {"call_id": c["id"], "error": str(e)} for c in calls_batch}
        finally:
            semaphore.release()


def _make_batched_prompt(base_prompt: str) -> str:
    """Convert a single-input prompt into a multi-input batched prompt."""
    batch_instruction = (
        "You will receive multiple transcripts as a JSON array. "
        "Process EACH transcript independently — do not let one transcript "
        "influence your classification of another.\n\n"
        'Return JSON: {"results": [{"call_id": "...", "reasoning": "...", "answer": ...}, ...]}\n'
    )
    if "Return JSON ONLY." in base_prompt:
        return base_prompt.replace("Return JSON ONLY.", batch_instruction + "Return JSON ONLY.")
    return base_prompt + "\n\n" + batch_instruction + "Return JSON ONLY."


def _make_batched_schema(single_schema: dict) -> dict:
    """Convert a single-result JSON schema into a batched results schema."""
    if not single_schema or "json_schema" not in single_schema:
        return {"type": "json_object"}

    single_item = single_schema["json_schema"]["schema"].copy()
    item_props = {**single_item["properties"], "call_id": {"type": "string"}}
    item_required = list(single_item["required"]) + ["call_id"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": single_schema["json_schema"]["name"] + "_batch",
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
                        },
                    },
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Field Processing
# ═══════════════════════════════════════════════════════════════════════════════

def run_field(
    client: OpenAI,
    model_config: dict,
    calls: list[dict],
    system_prompt: str,
    batch_size: int,
    field_name: str,
    response_schema: dict | None = None,
    budget: BudgetTracker | None = None,
) -> dict[str, dict]:
    """Run a field classifier for all calls.

    If batch_size <= 1: one input per API call (parallel via thread pool).
    If batch_size > 1: multiple inputs per API call (parallel batches).

    Args:
        calls: List of {"id": ..., "text": ...} dicts.
        system_prompt: The classification prompt.
        batch_size: Number of inputs per API call.
        field_name: For logging.
        response_schema: JSON schema for structured output.
        budget: Optional budget tracker.

    Returns:
        {call_id: {"reasoning": ..., "answer": ...}}
    """
    max_workers = model_config.get("max_concurrent", 15)
    results = {}

    if batch_size <= 1:
        logger.info(f"  {field_name}: processing {len(calls)} calls individually (max_workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    call_llm, client, model_config, c["id"], c["text"],
                    system_prompt, response_schema, budget,
                ): c["id"]
                for c in calls
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    results[cid] = result
                except Exception as e:
                    logger.error(f"  {field_name} {cid} thread error: {e}")
                    results[cid] = {"call_id": cid, "error": str(e)}
    else:
        batches = [calls[i : i + batch_size] for i in range(0, len(calls), batch_size)]
        logger.info(f"  {field_name}: processing {len(calls)} calls in {len(batches)} batches of {batch_size}")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    call_batch, client, model_config, batch,
                    system_prompt, response_schema, budget,
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"  {field_name} batch thread error: {e}")

    return results


def run_field_with_dependency(
    client: OpenAI,
    model_config: dict,
    calls: list[dict],
    field_config: dict,
    upstream_results: dict[str, dict],
    batch_size: int,
    budget: BudgetTracker | None = None,
) -> dict[str, dict]:
    """Run a field that depends on another field's results.

    Handles skip_when logic: short-circuits calls where the dependency
    field has certain values.
    """
    field_name = field_config["name"]
    skip_when = field_config.get("skip_when", {})
    skip_field = skip_when.get("field", "")
    skip_values = skip_when.get("values", [])
    skip_default = skip_when.get("default")
    system_prompt = field_config.get("system_prompt", "")
    response_schema = field_config.get("response_schema")

    results = {}
    calls_needing_llm = []

    for c in calls:
        cid = c["id"]
        upstream = upstream_results.get(cid, {})
        upstream_answer = upstream.get("answer", "")

        if upstream_answer in skip_values:
            results[cid] = {
                "call_id": cid,
                "reasoning": f"{skip_field}={upstream_answer}, skipping",
                "answer": skip_default,
            }
        elif upstream.get("error"):
            results[cid] = {
                "call_id": cid,
                "reasoning": f"{skip_field} unavailable, skipping",
                "answer": skip_default,
            }
        else:
            calls_needing_llm.append(c)

    if calls_needing_llm:
        logger.info(f"  {field_name}: {len(calls_needing_llm)}/{len(calls)} calls need LLM")
        llm_results = run_field(
            client, model_config, calls_needing_llm,
            system_prompt, batch_size, field_name,
            response_schema, budget,
        )
        results.update(llm_results)
    else:
        logger.info(f"  {field_name}: 0/{len(calls)} calls need LLM (all skipped)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Assembly & Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def assemble(
    calls: list[dict],
    field_results: dict[str, dict[str, dict]],
    cross_field_rules: list[dict] | None = None,
    flag_config: dict | None = None,
) -> dict[str, dict]:
    """Assemble final predictions from per-field outputs.

    Applies cross-field consistency rules and input quality flags.

    Args:
        calls: List of {"id": ..., "text": ...} dicts.
        field_results: {field_name: {call_id: {"answer": ...}}}
        cross_field_rules: Optional list of consistency rules.
        flag_config: Optional flagging config with keyword_patterns.

    Returns:
        {call_id: {"call_id": ..., field_name: value, ..., "flags": [...]}}
    """
    predictions = {}
    keyword_patterns = {}
    if flag_config:
        keyword_patterns = flag_config.get("keyword_patterns", {})

    for c in calls:
        cid = c["id"]
        pred = {"call_id": cid}

        for field_name, results in field_results.items():
            pred[field_name] = results.get(cid, {}).get("answer")

        # Apply cross-field rules
        if cross_field_rules:
            for rule in cross_field_rules:
                condition = rule.get("condition", {})
                action = rule.get("action", {})
                cond_field = condition.get("field", "")
                cond_value = condition.get("value")
                if pred.get(cond_field) == cond_value:
                    act_field = action.get("set_field", "")
                    current = pred.get(act_field)
                    new_val = action.get("to_value")
                    if current is not None and current != new_val:
                        sev = rule.get("severity", "warning")
                        logger.log(
                            logging.WARNING if sev == "warning" else logging.ERROR,
                            f"  Assembly {cid}: {cond_field}={cond_value} → setting {act_field} to {new_val} (was {current})",
                        )
                        pred[act_field] = new_val

        # Input quality flags
        flags = flag_input(c.get("text", ""), keyword_patterns)
        pred["flags"] = flags

        predictions[cid] = pred

    return predictions


def flag_input(
    input_text: str,
    keyword_patterns: dict[str, str] | None = None,
) -> list[str]:
    """Flag input quality/ambiguity issues. Returns list of flag strings."""
    flags = []
    turns = len(re.findall(r"(?:Agent|Caller|Speaker\s*\d*):", input_text))
    caller_parts = re.findall(r"Caller:\s*(.*?)(?=(?:Agent|Caller):|$)", input_text, re.DOTALL)
    caller_words = sum(len(p.split()) for p in caller_parts)

    if turns < 3 or len(input_text) < 100:
        flags.append("very_short")
    if caller_words < 5:
        flags.append("voicemail")
    if re.search(r"wrong number|called the wrong", input_text, re.IGNORECASE) and len(input_text) < 500:
        flags.append("wrong_number")

    # Check keyword patterns
    if keyword_patterns:
        has_domain = False
        for category, pattern in keyword_patterns.items():
            if re.search(pattern, input_text, re.IGNORECASE):
                has_domain = True
                break
        if not has_domain and turns >= 4:
            flags.append("no_domain_content")

    return flags


def score_field(
    predictions: dict[str, dict],
    labels: dict[str, dict],
    field_name: str,
    field_config: dict | None = None,
) -> dict:
    """Score predictions against gold labels for a single field.

    Returns:
        {"correct": int, "total": int, "accuracy": float, "errors": [...]}
    """
    correct = 0
    total = 0
    errors = []
    gold_column = field_name
    if field_config:
        gold_column = field_config.get("gold_column", field_name)

    skip_null_gold = field_config.get("skip_null_gold", False) if field_config else False
    skip_null_both = field_config.get("skip_null_both", False) if field_config else False

    for cid, pred_data in predictions.items():
        if cid not in labels:
            continue

        gold_val = labels[cid].get(gold_column, "")
        if isinstance(gold_val, str):
            gold_val = gold_val.strip()
        pred_val = pred_data.get("answer", "")
        if isinstance(pred_val, str):
            pred_val = pred_val.strip()

        # Normalize nulls
        if not gold_val or (isinstance(gold_val, str) and gold_val.lower() in ("null", "none")):
            gold_val = ""
        if not pred_val or (isinstance(pred_val, str) and pred_val.lower() in ("null", "none")):
            pred_val = ""

        if skip_null_gold and not gold_val:
            continue
        if skip_null_both and not gold_val and not pred_val:
            continue

        total += 1
        if gold_val == pred_val:
            correct += 1
        else:
            errors.append({
                "call_id": cid,
                "pred": pred_val,
                "gold": gold_val,
                "reasoning": pred_data.get("reasoning", ""),
            })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "errors": errors,
    }


def score_all_fields(
    field_results: dict[str, dict[str, dict]],
    labels: dict[str, dict],
    field_configs: list[dict],
) -> dict[str, dict]:
    """Score all fields and return summary."""
    scores = {}
    for fc in field_configs:
        fname = fc["name"]
        if fname in field_results:
            scores[fname] = score_field(field_results[fname], labels, fname, fc)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(
    config: dict,
    field_name: str,
    model_name: str,
    batch_size: int,
    max_calls: int,
    output_path: str | None = None,
    system_prompt_override: str | None = None,
    seed: int = 42,
) -> dict:
    """Run a baseline evaluation for one field + model combination.

    Returns dict with accuracy, errors, cost, and full results.
    """
    field_config = get_field_config(config, field_name)
    model_config = get_model_config(config, model_name)

    budget = BudgetTracker(config)
    client = make_client(model_config)

    # Load and sample data
    inputs, labels = load_data(config)
    train_ids, holdout_ids = split_holdout(
        inputs, labels,
        config["data"].get("holdout_ratio", 0.2),
        config["data"].get("holdout_seed", 42),
    )
    sample_inputs, sample_labels = sample_data(inputs, labels, max_calls, seed, exclude_ids=holdout_ids)

    # Prepare calls
    calls = [{"id": cid, "text": text} for cid, text in sample_inputs.items()]

    # Choose prompt
    system_prompt = system_prompt_override or field_config.get("system_prompt", "")
    response_schema = field_config.get("response_schema")

    logger.info(f"Baseline: field={field_name} model={model_name} bs={batch_size} n={len(calls)}")

    # Check for dependencies
    depends_on = field_config.get("depends_on")
    if depends_on:
        dep_config = get_field_config(config, depends_on)
        dep_model_config = model_config  # Use same model for dependency field
        dep_prompt = dep_config.get("system_prompt", "")
        dep_schema = dep_config.get("response_schema")

        logger.info(f"  Running dependency field '{depends_on}' first...")
        dep_results = run_field(
            client, dep_model_config, calls, dep_prompt, batch_size,
            depends_on, dep_schema, budget,
        )

        results = run_field_with_dependency(
            client, model_config, calls, field_config,
            dep_results, batch_size, budget,
        )
    else:
        results = run_field(
            client, model_config, calls, system_prompt, batch_size,
            field_name, response_schema, budget,
        )

    # Score
    score = score_field(results, sample_labels, field_name, field_config)
    budget.save()

    output = {
        "phase": "baseline",
        "field": field_name,
        "model": model_name,
        "batch_size": batch_size,
        "n": len(calls),
        "seed": seed,
        "accuracy": score["accuracy"],
        "correct": score["correct"],
        "total": score["total"],
        "n_errors": len(score["errors"]),
        "cost_usd": budget.total_cost,
        "results": {cid: r for cid, r in results.items()},
        "errors": score["errors"],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    logger.info(f"Baseline result: {field_name} = {score['accuracy']:.1f}% ({score['correct']}/{score['total']})")
    logger.info(f"Cost: {budget.summary()}")

    return output


def run_evaluate(
    config: dict,
    results_path: str,
    field_name: str,
) -> dict:
    """Evaluate saved results: score, categorize errors, produce report.

    Returns dict with accuracy, categorized errors, and report path.
    """
    with open(results_path) as f:
        results_data = json.load(f)

    field_config = get_field_config(config, field_name)
    inputs, labels = load_data(config)

    results = results_data.get("results", {})
    score = score_field(results, labels, field_name, field_config)

    # Categorize errors
    categorized = []
    for error in score["errors"]:
        cid = error["call_id"]
        input_text = inputs.get(cid, "")
        cat = categorize_error(
            field_name, error["pred"], error["gold"],
            error.get("reasoning", ""), input_text, field_config,
        )
        categorized.append({
            **error,
            **cat,
            "input_excerpt": input_text[:200],
        })

    # Generate report
    report = summarize_errors(categorized, field_name, score["total"])
    opro_context = build_opro_error_context(categorized, field_name)

    output_dir = Path(config.get("output_dir", "prompt_engineer_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"error_report_{field_name}.md"
    report_path.write_text(report)

    logger.info(f"Error report written to {report_path}")
    logger.info(f"Accuracy: {score['accuracy']:.1f}% ({score['correct']}/{score['total']})")

    return {
        "accuracy": score["accuracy"],
        "correct": score["correct"],
        "total": score["total"],
        "categorized_errors": categorized,
        "report_path": str(report_path),
        "opro_context": opro_context,
    }


def run_audit(
    config: dict,
    output_path: str | None = None,
) -> dict:
    """Run gold label audit on the full dataset.

    Returns dict with flagged items count and report path.
    """
    inputs, labels = load_data(config)
    field_configs = config.get("fields", [])
    cross_field_rules = config.get("cross_field_rules")
    flags_cfg = config.get("flags", {})
    keyword_patterns = flags_cfg.get("keyword_patterns")

    flagged = audit_dataset(
        inputs, labels, field_configs,
        cross_field_rules, keyword_patterns,
    )

    output_dir = Path(config.get("output_dir", "prompt_engineer_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    report = generate_report(flagged)
    report_path = output_dir / "gold_label_audit.md"
    report_path.write_text(report)

    csv_path = output_path or str(output_dir / "flagged_items.csv")
    export_csv(flagged, csv_path)

    logger.info(f"Audit complete: {len(flagged)} items flagged")
    logger.info(f"Report: {report_path}")
    logger.info(f"CSV: {csv_path}")

    return {
        "flagged_count": len(flagged),
        "report_path": str(report_path),
        "csv_path": csv_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prompt Engineer Runner")
    parser.add_argument("--config", default="prompt-engineer.yaml", help="Config YAML path")
    parser.add_argument("--phase", required=True, choices=["baseline", "evaluate", "audit"],
                        help="Phase to run")
    parser.add_argument("--field", help="Field name to process")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config default)")
    parser.add_argument("--max-calls", type=int, default=50, help="Max items to process")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--results", help="Results JSON path (for evaluate phase)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", help="System prompt override (path to .txt file)")
    args = parser.parse_args()

    # Load .env if available
    env_path = Path("Scripts/.env")
    if env_path.exists():
        _load_dotenv(env_path)

    config = load_config(args.config)

    if args.phase == "baseline":
        if not args.field or not args.model:
            parser.error("--field and --model required for baseline phase")
        model_config = get_model_config(config, args.model)
        bs = args.batch_size or model_config.get("default_batch_size", 1)
        prompt_override = None
        if args.prompt:
            prompt_override = Path(args.prompt).read_text()
        run_baseline(config, args.field, args.model, bs, args.max_calls,
                     args.output, prompt_override, args.seed)

    elif args.phase == "evaluate":
        if not args.field or not args.results:
            parser.error("--field and --results required for evaluate phase")
        run_evaluate(config, args.results, args.field)

    elif args.phase == "audit":
        run_audit(config, args.output)


def _load_dotenv(path: Path):
    """Minimal .env loader (no external dependency needed)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


if __name__ == "__main__":
    main()
