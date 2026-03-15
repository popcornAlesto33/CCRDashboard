"""Gold label audit module for prompt engineering optimization.

Scans a labeled dataset for suspected labeling errors using configurable
rules, cross-field consistency checks, and input quality signals.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── Severity Levels ───────────────────────────────────────────────────────────

SEVERITY_HIGH = "HIGH"      # Impossible/contradictory state
SEVERITY_MEDIUM = "MEDIUM"  # Likely wrong based on heuristics
SEVERITY_LOW = "LOW"        # Suspicious but plausible


# ── Main Audit Function ──────────────────────────────────────────────────────

def audit_dataset(
    inputs: dict[str, str],
    labels: dict[str, dict],
    field_configs: list[dict],
    cross_field_rules: list[dict] | None = None,
    keyword_patterns: dict[str, str] | None = None,
    flag_rules: list[dict] | None = None,
) -> list[dict]:
    """Audit an entire labeled dataset for suspected gold label errors.

    Args:
        inputs: {id: input_text} mapping.
        labels: {id: {field_name: value, ...}} mapping.
        field_configs: List of field definitions from config YAML.
        cross_field_rules: Cross-field consistency rules from config.
        keyword_patterns: {category: regex_pattern} for domain signals.
        flag_rules: Input quality flag rules.

    Returns:
        List of flagged items, sorted by severity (HIGH first).
    """
    flagged = []
    field_map = {f["name"]: f for f in field_configs}

    for item_id, item_labels in labels.items():
        input_text = inputs.get(item_id, "")
        if not input_text:
            continue

        features = analyze_input(input_text, keyword_patterns)

        # 1. Input quality flags
        quality_flags = _check_input_quality(features, item_labels, field_map)
        flagged.extend(_make_flagged(item_id, f, input_text) for f in quality_flags)

        # 2. Cross-field consistency
        if cross_field_rules:
            consistency_flags = _check_cross_field(item_labels, cross_field_rules)
            flagged.extend(_make_flagged(item_id, f, input_text) for f in consistency_flags)

        # 3. Field-specific heuristic checks
        for field_config in field_configs:
            field_flags = _check_field_heuristics(
                item_labels, field_config, features, field_map,
            )
            flagged.extend(_make_flagged(item_id, f, input_text) for f in field_flags)

    # Sort: HIGH → MEDIUM → LOW
    severity_order = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    flagged.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["id"]))

    return flagged


def analyze_input(
    input_text: str,
    keyword_patterns: dict[str, str] | None = None,
) -> dict:
    """Extract features from an input text for audit rule evaluation.

    Returns a feature dict with: turns, caller_words, total_words, text_length,
    is_voicemail, is_very_short, is_wrong_number, is_reschedule, keyword_matches, etc.
    """
    turns = len(re.findall(r"(?:Agent|Caller|Speaker\s*\d*):", input_text))
    caller_parts = re.findall(r"Caller:\s*(.*?)(?=(?:Agent|Caller):|$)", input_text, re.DOTALL)
    caller_words = sum(len(p.split()) for p in caller_parts)
    total_words = len(input_text.split())
    redacted = len(re.findall(r"\[(?:MEDICAL_CONDITION|DRUG|MEDICAL_PROCESS|INJURY)\]", input_text))

    features = {
        "turns": turns,
        "caller_words": caller_words,
        "total_words": total_words,
        "text_length": len(input_text),
        "redacted_count": redacted,
        "is_voicemail": caller_words < 5,
        "is_very_short": turns < 3 or len(input_text) < 100,
        "is_wrong_number": bool(re.search(r"wrong number|called the wrong", input_text, re.IGNORECASE)),
        "is_reschedule": bool(re.search(
            r"reschedule|move.{1,15}appointment|cancel.{1,15}appointment|change.{1,15}appointment",
            input_text, re.IGNORECASE,
        )),
        "is_admin": bool(re.search(
            r"reschedule|cancel|pick.?up|results|records|transfer|file",
            input_text, re.IGNORECASE,
        )),
        "keyword_matches": {},
    }

    # Apply keyword patterns from config
    if keyword_patterns:
        for category, pattern in keyword_patterns.items():
            matches = re.findall(pattern, input_text, re.IGNORECASE)
            features["keyword_matches"][category] = list(
                set(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
            )

    return features


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(flagged: list[dict], output_path: str | None = None) -> str:
    """Generate a markdown audit report.

    Args:
        flagged: List of flagged items from audit_dataset().
        output_path: Optional path to write report.

    Returns:
        Markdown report string.
    """
    if not flagged:
        return "# Gold Label Audit\n\nNo issues found.\n"

    severity_counts = Counter(f["severity"] for f in flagged)
    rule_counts = Counter(f["rule"] for f in flagged)

    lines = [
        "# Gold Label Audit Report",
        "",
        f"**Total flagged:** {len(flagged)}",
        f"- HIGH: {severity_counts.get(SEVERITY_HIGH, 0)}",
        f"- MEDIUM: {severity_counts.get(SEVERITY_MEDIUM, 0)}",
        f"- LOW: {severity_counts.get(SEVERITY_LOW, 0)}",
        "",
        "## Issues by Rule",
        "",
        "| Rule | Count | Severity |",
        "|------|-------|----------|",
    ]
    for rule, count in rule_counts.most_common():
        sev = next(f["severity"] for f in flagged if f["rule"] == rule)
        lines.append(f"| {rule} | {count} | {sev} |")

    lines.extend(["", "## Flagged Items", ""])

    for sev in [SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW]:
        items = [f for f in flagged if f["severity"] == sev]
        if not items:
            continue
        lines.append(f"### {sev} ({len(items)} items)")
        lines.append("")
        for item in items[:20]:  # Cap per severity for readability
            lines.append(f"- **{item['id']}** [{item['rule']}]: {item['details']}")
            if item.get("labels_excerpt"):
                lines.append(f"  Labels: {item['labels_excerpt']}")
        if len(items) > 20:
            lines.append(f"  ... and {len(items) - 20} more")
        lines.append("")

    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report)
    return report


def export_csv(flagged: list[dict], output_path: str) -> None:
    """Export flagged items as CSV for spreadsheet review."""
    if not flagged:
        return

    fieldnames = ["id", "severity", "rule", "details", "input_excerpt", "labels_excerpt"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in flagged:
            writer.writerow({
                "id": item["id"],
                "severity": item["severity"],
                "rule": item["rule"],
                "details": item["details"],
                "input_excerpt": item.get("input_excerpt", "")[:200],
                "labels_excerpt": item.get("labels_excerpt", ""),
            })


# ── Private Helpers ───────────────────────────────────────────────────────────

def _make_flagged(item_id: str, flag: dict, input_text: str) -> dict:
    """Wrap an audit flag into a standard flagged item dict."""
    return {
        "id": item_id,
        "severity": flag["severity"],
        "rule": flag["rule"],
        "details": flag["details"],
        "input_excerpt": input_text[:200] if input_text else "",
        "labels_excerpt": flag.get("labels_excerpt", ""),
    }


def _check_input_quality(
    features: dict,
    item_labels: dict,
    field_map: dict,
) -> list[dict]:
    """Check if input quality contradicts specific labels."""
    flags = []

    # Voicemail/very short with specific classification
    if features["is_voicemail"] or features["is_very_short"]:
        for field_name, field_config in field_map.items():
            label_val = item_labels.get(field_name, "")
            if not label_val or label_val.lower() in ("null", "none", "n/a", ""):
                continue
            # Specific sub-categories on very short inputs are suspect
            hierarchy = field_config.get("hierarchy", {}).get("parents", {})
            if hierarchy:
                for parent, subs in hierarchy.items():
                    if label_val in subs:
                        flags.append({
                            "severity": SEVERITY_MEDIUM,
                            "rule": f"SHORT_INPUT_SPECIFIC_{field_name.upper()}",
                            "details": (
                                f"Very short input ({features['turns']} turns, "
                                f"{features['caller_words']} caller words) labeled with "
                                f"specific sub-category '{label_val}'"
                            ),
                            "labels_excerpt": f"{field_name}={label_val}",
                        })
                        break

    # Wrong number with treatment type
    if features["is_wrong_number"]:
        for field_name, field_config in field_map.items():
            label_val = item_labels.get(field_name, "")
            if label_val and label_val.lower() not in ("other", "n/a", "null", "none", ""):
                hierarchy = field_config.get("hierarchy", {}).get("parents", {})
                if hierarchy:
                    flags.append({
                        "severity": SEVERITY_HIGH,
                        "rule": f"WRONG_NUMBER_{field_name.upper()}",
                        "details": f"Wrong number call labeled with '{label_val}'",
                        "labels_excerpt": f"{field_name}={label_val}",
                    })

    return flags


def _check_cross_field(
    item_labels: dict,
    cross_field_rules: list[dict],
) -> list[dict]:
    """Check cross-field consistency rules."""
    flags = []

    for rule in cross_field_rules:
        condition = rule.get("condition", {})
        action = rule.get("action", {})
        cond_field = condition.get("field", "")
        cond_value = condition.get("value")
        act_field = action.get("set_field", "")
        expected_value = action.get("to_value")

        actual_cond = item_labels.get(cond_field, "")
        actual_act = item_labels.get(act_field, "")

        # Normalize nulls
        if actual_act and actual_act.lower() in ("null", "none"):
            actual_act = ""
        if expected_value is None:
            expected_value_str = ""
        else:
            expected_value_str = str(expected_value)

        if actual_cond == cond_value:
            if actual_act and actual_act != expected_value_str:
                severity = SEVERITY_HIGH if rule.get("severity") == "error" else SEVERITY_MEDIUM
                flags.append({
                    "severity": severity,
                    "rule": rule.get("name", "CROSS_FIELD").upper(),
                    "details": (
                        f"{cond_field}='{actual_cond}' but {act_field}='{actual_act}' "
                        f"(expected '{expected_value}')"
                    ),
                    "labels_excerpt": f"{cond_field}={actual_cond}, {act_field}={actual_act}",
                })

    return flags


def _check_field_heuristics(
    item_labels: dict,
    field_config: dict,
    features: dict,
    field_map: dict,
) -> list[dict]:
    """Check field-specific heuristic rules based on keyword signals."""
    flags = []
    field_name = field_config["name"]
    label_val = item_labels.get(field_name, "")
    if not label_val or label_val.lower() in ("null", "none", ""):
        return flags

    hierarchy = field_config.get("hierarchy", {}).get("parents", {})
    if not hierarchy:
        return flags

    keyword_matches = features.get("keyword_matches", {})

    # For hierarchical fields: sub-category label with no matching keywords
    for parent, subs in hierarchy.items():
        if label_val in subs:
            # Check if there are any domain-relevant keywords
            all_keywords = []
            for cat_keywords in keyword_matches.values():
                all_keywords.extend(cat_keywords)

            if not all_keywords and features["turns"] >= 4:
                flags.append({
                    "severity": SEVERITY_LOW,
                    "rule": f"SUB_NO_KEYWORDS_{field_name.upper()}",
                    "details": (
                        f"Sub-category '{label_val}' assigned but no domain keywords "
                        f"found in {features['turns']}-turn input"
                    ),
                    "labels_excerpt": f"{field_name}={label_val}",
                })

    # Admin/reschedule call with specific non-admin sub-category
    if features["is_reschedule"] and not keyword_matches.get("medical"):
        for parent, subs in hierarchy.items():
            if label_val in subs and parent.lower() not in ("other",):
                flags.append({
                    "severity": SEVERITY_MEDIUM,
                    "rule": f"ADMIN_SPECIFIC_{field_name.upper()}",
                    "details": (
                        f"Reschedule/admin call with no medical keywords "
                        f"labeled as specific sub-category '{label_val}'"
                    ),
                    "labels_excerpt": f"{field_name}={label_val}",
                })
                break

    return flags
