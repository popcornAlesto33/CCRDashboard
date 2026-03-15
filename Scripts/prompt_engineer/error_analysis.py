"""Error analysis module for prompt engineering optimization.

Categorizes prediction errors by type, extracts domain signals,
and produces structured error reports for OPRO optimization.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── Error Categories ──────────────────────────────────────────────────────────

ERROR_CATEGORIES = {
    "cascade": "Error caused by upstream field misclassification",
    "parent_sub": "Correct parent category but wrong sub-category (or vice versa)",
    "format": "Correct intent but wrong format/spelling/casing",
    "gold_suspect": "Gold label itself may be wrong (model reasoning seems correct)",
    "input_quality": "Input too short/garbled/voicemail to classify reliably",
    "genuine": "Genuine misclassification by the model",
}


def categorize_error(
    field_name: str,
    pred: Any,
    gold: Any,
    reasoning: str,
    input_text: str,
    field_config: dict,
    upstream_results: dict | None = None,
) -> dict:
    """Categorize a single prediction error.

    Args:
        field_name: Name of the field being classified.
        pred: Predicted value.
        gold: Gold label value.
        reasoning: Model's reasoning for the prediction.
        input_text: The input text (transcript).
        field_config: Field definition from config YAML.
        upstream_results: Results from dependency fields (if any).

    Returns:
        dict with keys: category, confidence, details.
    """
    pred_str = _normalize(pred)
    gold_str = _normalize(gold)

    # 1. Format mismatch (same intent, different string)
    if _is_format_mismatch(pred_str, gold_str, field_config):
        return {
            "category": "format",
            "confidence": "high",
            "details": f"Predicted '{pred}' matches gold '{gold}' after normalization",
        }

    # 2. Parent/sub confusion (hierarchical fields)
    hierarchy = field_config.get("hierarchy", {}).get("parents", {})
    if hierarchy:
        parent_sub = _check_parent_sub(pred_str, gold_str, hierarchy)
        if parent_sub:
            return parent_sub

    # 3. Cascade error (upstream dependency was wrong)
    depends_on = field_config.get("depends_on")
    if depends_on and upstream_results:
        cascade = _check_cascade(field_name, depends_on, upstream_results)
        if cascade:
            return cascade

    # 4. Input quality issues
    quality = _check_input_quality(input_text)
    if quality:
        return quality

    # 5. Gold label suspect (model reasoning seems sound)
    if _is_gold_suspect(pred_str, gold_str, reasoning, input_text):
        return {
            "category": "gold_suspect",
            "confidence": "medium",
            "details": f"Model reasoning seems plausible for '{pred}' but gold says '{gold}'",
        }

    # 6. Default: genuine error
    return {
        "category": "genuine",
        "confidence": "high",
        "details": f"Predicted '{pred}', gold '{gold}'",
    }


# ── Signal Extraction ─────────────────────────────────────────────────────────

def extract_signals(
    input_text: str,
    keyword_patterns: dict[str, str] | None = None,
) -> dict:
    """Extract domain-agnostic signals from input text.

    Args:
        input_text: The input text to analyze.
        keyword_patterns: Optional dict of {category: regex_pattern} from config.

    Returns:
        dict with signal counts and features.
    """
    signals = {
        "length_chars": len(input_text),
        "length_words": len(input_text.split()),
        "speaker_turns": len(re.findall(r"(?:Agent|Caller|Speaker\s*\d*):", input_text)),
    }

    # Detect caller vs agent word counts
    caller_parts = re.findall(r"Caller:\s*(.*?)(?=(?:Agent|Caller):|$)", input_text, re.DOTALL)
    signals["caller_words"] = sum(len(p.split()) for p in caller_parts)

    agent_parts = re.findall(r"Agent:\s*(.*?)(?=(?:Agent|Caller):|$)", input_text, re.DOTALL)
    signals["agent_words"] = sum(len(p.split()) for p in agent_parts)

    # Redaction density
    redacted = re.findall(r"\[[A-Z_]+\]", input_text)
    signals["redacted_count"] = len(redacted)
    signals["redaction_ratio"] = len(redacted) / max(signals["length_words"], 1)

    # Apply keyword patterns from config
    signals["keyword_matches"] = {}
    if keyword_patterns:
        for category, pattern in keyword_patterns.items():
            matches = re.findall(pattern, input_text, re.IGNORECASE)
            signals["keyword_matches"][category] = list(set(m.lower() if isinstance(m, str) else m[0].lower() for m in matches))

    return signals


# ── Error Summarization ──────────────────────────────────────────────────────

def summarize_errors(
    errors: list[dict],
    field_name: str,
    total_evaluated: int,
) -> str:
    """Generate a markdown error analysis report.

    Args:
        errors: List of error dicts from categorize_error(), each augmented with
                'call_id', 'pred', 'gold', 'reasoning', 'input_excerpt'.
        field_name: Name of the field.
        total_evaluated: Total number of items evaluated.

    Returns:
        Markdown-formatted error report string.
    """
    if not errors:
        return f"## {field_name}\n\nNo errors to analyze.\n"

    accuracy = (total_evaluated - len(errors)) / total_evaluated * 100
    lines = [
        f"## {field_name}",
        f"",
        f"**Accuracy:** {accuracy:.1f}% ({total_evaluated - len(errors)}/{total_evaluated})",
        f"**Total errors:** {len(errors)}",
        f"",
    ]

    # Category breakdown
    cat_counts = Counter(e["category"] for e in errors)
    lines.append("### Error Categories")
    lines.append("")
    lines.append("| Category | Count | % of Errors | Description |")
    lines.append("|----------|-------|-------------|-------------|")
    for cat, count in cat_counts.most_common():
        pct = count / len(errors) * 100
        desc = ERROR_CATEGORIES.get(cat, "Unknown")
        lines.append(f"| {cat} | {count} | {pct:.0f}% | {desc} |")
    lines.append("")

    # Top confusion pairs (pred → gold)
    confusion = Counter((e["pred"], e["gold"]) for e in errors)
    lines.append("### Top Confusion Pairs")
    lines.append("")
    lines.append("| Predicted | Gold | Count |")
    lines.append("|-----------|------|-------|")
    for (pred, gold), count in confusion.most_common(10):
        lines.append(f"| {pred} | {gold} | {count} |")
    lines.append("")

    # Example errors (top 5 genuine, top 3 gold_suspect)
    for category, max_examples in [("genuine", 5), ("gold_suspect", 3), ("cascade", 3)]:
        cat_errors = [e for e in errors if e["category"] == category]
        if not cat_errors:
            continue
        lines.append(f"### {category.title()} Errors (top {min(len(cat_errors), max_examples)})")
        lines.append("")
        for e in cat_errors[:max_examples]:
            lines.append(f"**Call {e.get('call_id', '?')}**: pred=`{e['pred']}` gold=`{e['gold']}`")
            if e.get("reasoning"):
                reasoning_short = e["reasoning"][:200] + "..." if len(e.get("reasoning", "")) > 200 else e["reasoning"]
                lines.append(f"> Reasoning: {reasoning_short}")
            if e.get("input_excerpt"):
                lines.append(f"> Input: {e['input_excerpt'][:150]}...")
            lines.append("")

    return "\n".join(lines)


def build_opro_error_context(
    errors: list[dict],
    field_name: str,
    max_examples: int = 10,
) -> str:
    """Build a concise error context string for the OPRO optimizer.

    Returns a focused summary suitable for injecting into an optimizer prompt.
    """
    if not errors:
        return f"Field '{field_name}': no errors found."

    cat_counts = Counter(e["category"] for e in errors)
    genuine = [e for e in errors if e["category"] == "genuine"]
    confusion = Counter((e["pred"], e["gold"]) for e in genuine)

    parts = [
        f"Field: {field_name}",
        f"Error count: {len(errors)}",
        f"Categories: {dict(cat_counts.most_common())}",
        f"Top confusions: {dict(confusion.most_common(5))}",
        "",
        "Example errors:",
    ]

    for e in genuine[:max_examples]:
        parts.append(f"  - pred={e['pred']!r} gold={e['gold']!r}")
        if e.get("reasoning"):
            parts.append(f"    reasoning: {e['reasoning'][:150]}")
        if e.get("input_excerpt"):
            parts.append(f"    input: {e['input_excerpt'][:100]}")

    return "\n".join(parts)


# ── Private Helpers ───────────────────────────────────────────────────────────

def _normalize(value: Any) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_format_mismatch(pred: str, gold: str, field_config: dict) -> bool:
    """Check if pred and gold differ only in formatting."""
    if pred == gold:
        return False
    # Strip common prefixes/suffixes
    pred_clean = re.sub(r"^\d+[a-z]?\.\s*", "", pred).strip()
    gold_clean = re.sub(r"^\d+[a-z]?\.\s*", "", gold).strip()
    if pred_clean == gold_clean:
        return True
    # Check if one contains the other
    if pred_clean in gold_clean or gold_clean in pred_clean:
        return True
    return False


def _check_parent_sub(pred: str, gold: str, hierarchy: dict) -> dict | None:
    """Check for parent/sub-category confusion."""
    for parent, subs in hierarchy.items():
        parent_lower = parent.lower()
        subs_lower = [s.lower() for s in subs]

        pred_is_parent = pred == parent_lower
        gold_is_parent = gold == parent_lower
        pred_is_sub = pred in subs_lower
        gold_is_sub = gold in subs_lower

        # Pred=parent, gold=sub (or vice versa) under same parent
        if (pred_is_parent and gold_is_sub) or (pred_is_sub and gold_is_parent):
            return {
                "category": "parent_sub",
                "confidence": "high",
                "details": f"Parent/sub confusion under '{parent}': pred='{pred}' vs gold='{gold}'",
            }

        # Both are subs under same parent (wrong sub)
        if pred_is_sub and gold_is_sub:
            return {
                "category": "parent_sub",
                "confidence": "medium",
                "details": f"Wrong sub-category under '{parent}': pred='{pred}' vs gold='{gold}'",
            }

    return None


def _check_cascade(field_name: str, depends_on: str, upstream_results: dict) -> dict | None:
    """Check if error cascaded from upstream field."""
    upstream = upstream_results.get(depends_on, {})
    if upstream.get("error") or upstream.get("is_wrong"):
        return {
            "category": "cascade",
            "confidence": "high",
            "details": f"Upstream field '{depends_on}' was incorrect — cascaded to '{field_name}'",
        }
    return None


def _check_input_quality(input_text: str) -> dict | None:
    """Check if input quality is too poor for reliable classification."""
    turns = len(re.findall(r"(?:Agent|Caller|Speaker\s*\d*):", input_text))
    if turns < 3 or len(input_text) < 100:
        return {
            "category": "input_quality",
            "confidence": "high",
            "details": f"Very short input ({turns} turns, {len(input_text)} chars)",
        }
    caller_parts = re.findall(r"Caller:\s*(.*?)(?=(?:Agent|Caller):|$)", input_text, re.DOTALL)
    caller_words = sum(len(p.split()) for p in caller_parts)
    if caller_words < 5:
        return {
            "category": "input_quality",
            "confidence": "high",
            "details": f"Voicemail/automated — caller spoke only {caller_words} words",
        }
    return None


def _is_gold_suspect(pred: str, gold: str, reasoning: str, input_text: str) -> bool:
    """Heuristic: does the model's reasoning seem more plausible than the gold label?

    This is intentionally conservative — flags for human review, not auto-correction.
    """
    if not reasoning:
        return False
    reasoning_lower = reasoning.lower()
    # If model explicitly mentions the gold label value and explains why it's wrong
    if gold in reasoning_lower and any(
        phrase in reasoning_lower
        for phrase in ["however", "but the transcript", "no evidence of", "doesn't mention"]
    ):
        return True
    return False
