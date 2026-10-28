from __future__ import annotations

import json
import re

from precompute.concentration.helpers import compact_json
from precompute.flow1.claims_check import check_synergy_entry
from precompute.llm_call import format_llm_feedback
from tracing import trace_step

STRENGTH_SCORES = {
    "Strong": 0.166,
    "Moderate": 0.10,
    "Weak": 0.05,
    "None": 0,
}
_SYNERGY_ENTRY_KEYS = ("concern", "pairs", "strength", "reason", "score")
SYNERGY_SCHEMA_NAME = "synergy_reply"


def _cleanup_raw(raw):
    cleaned = raw
    if isinstance(cleaned, str):
        cleaned = cleaned.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            cleaned = cleaned.strip()
    return cleaned


def _reason_empty(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _concentration_for(concentrations, name):
    if not isinstance(concentrations, dict) or not name:
        return None
    if name in concentrations:
        return concentrations[name]
    want = str(name).casefold()
    for key, value in concentrations.items():
        if str(key).casefold() == want:
            return value
    return None


def _row_ingredient_name(row: dict):
    for key in ("ingredient_name", "name", "canonical_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _tier_rows_from_concern_tiers(tiers, concern_keys) -> list[dict]:
    rows = []
    if isinstance(tiers, list):
        for item in tiers:
            if isinstance(item, dict):
                rows.append(item)
        return rows
    if not isinstance(tiers, dict):
        return rows
    keys = list(concern_keys) if isinstance(concern_keys, list) else []
    for key, items in tiers.items():
        if keys and key not in keys:
            continue
        if isinstance(items, dict):
            row = dict(items)
            row.setdefault("concern_key", key)
            rows.append(row)
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and item.strip():
                rows.append({"concern_key": key, "ingredient_name": item})
            elif isinstance(item, dict):
                row = dict(item)
                row.setdefault("concern_key", key)
                rows.append(row)
    return rows


def build_synergy_concern_ingredients(concern_data) -> dict:
    data = concern_data if isinstance(concern_data, dict) else {}
    keys = list(data.get("concern_keys") or [])
    listed = {key: [] for key in keys}
    concentrations = data.get("concentrations")
    for row in _tier_rows_from_concern_tiers(data.get("concern_tiers"), keys):
        key = row.get("concern_key")
        if key is None:
            key = row.get("concern")
        if key not in listed:
            continue
        name = _row_ingredient_name(row)
        if not name:
            continue
        listed[key].append(
            {
                "ingredient_name": name,
                "tier": row.get("tier"),
                "approximate_concentration": _concentration_for(
                    concentrations, name
                ),
            }
        )
    return listed


def _listed_names(rows) -> list[str]:
    names = []
    for row in rows or []:
        if isinstance(row, dict):
            name = _row_ingredient_name(row)
            if name:
                names.append(name)
        elif isinstance(row, str) and row.strip():
            names.append(row)
    return names


def _pair_ingredient_names(pairs) -> list[str]:
    if not isinstance(pairs, list):
        return []
    return [item for item in pairs if isinstance(item, str) and item.strip()]


def concern_list_problems(entry, listed_rows) -> list[str]:
    if not isinstance(entry, dict):
        return []
    key = entry.get("concern")
    names = _listed_names(listed_rows)
    allow = {name.casefold() for name in names}
    problems = []
    pairs = entry.get("pairs")
    pair_names = _pair_ingredient_names(pairs)
    if pair_names:
        shown = compact_json(names)
        for name in pair_names:
            if name.casefold() not in allow:
                problems.append(
                    f"concern {key!r}: {name} is not listed for this concern; "
                    f"use only: {shown}"
                )
    if len(names) < 2:
        strength = entry.get("strength")
        score = entry.get("score")
        empty_pairs = isinstance(pairs, list) and len(pairs) == 0
        if not empty_pairs or strength != "None" or score != 0:
            problems.append(
                f"concern {key!r} has fewer than 2 listed ingredients; "
                "pairs must be [] and strength None"
            )
    return problems


def recompute_total_synergy_score(parsed: dict) -> None:
    total = 0.0
    synergies = parsed.get("synergies") if isinstance(parsed, dict) else None
    if isinstance(synergies, list):
        for entry in synergies:
            if isinstance(entry, dict) and _is_number(entry.get("score")):
                total += float(entry.get("score"))
    if total > 0.5:
        total = 0.5
    parsed["total_synergy_score"] = total


def reset_synergy_concern_list_failures(parsed, listed_by_concern) -> list[tuple]:
    if not isinstance(parsed, dict) or not isinstance(listed_by_concern, dict):
        return []
    resets = []
    for entry in parsed.get("synergies") or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("concern")
        problems = concern_list_problems(entry, listed_by_concern.get(key) or [])
        if not problems:
            continue
        entry["pairs"] = []
        entry["strength"] = "None"
        entry["score"] = 0
        entry["reason"] = "No meaningful synergy found."
        note = f"synergy '{key}' reset to None: {problems[0]}"
        with trace_step("synergy_reset") as fields:
            fields["debug_output"] = note
        resets.append((key, problems[0]))
    if resets:
        recompute_total_synergy_score(parsed)
    return resets


def collect_synergy_problems(
    text, concern_keys, product_names=None, data=None, listed_by_concern=None
) -> tuple:
    """Return (parsed_or_None, problem strings)."""
    keys = list(concern_keys) if isinstance(concern_keys, list) else []
    problems = []
    cleaned = _cleanup_raw(text)
    try:
        parsed = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
    except Exception:
        return None, ["synergy reply is not valid JSON"]
    if not isinstance(parsed, dict):
        return None, ["synergy reply is not valid JSON"]
    synergies = parsed.get("synergies")
    if "total_synergy_score" not in parsed or not isinstance(synergies, list):
        problems.append(
            "synergy reply is not valid JSON "
            "{synergies:[...], total_synergy_score}"
        )
        return parsed, problems
    total = parsed.get("total_synergy_score")
    if not _is_number(total):
        problems.append(
            "synergy reply is not valid JSON "
            "{synergies:[...], total_synergy_score}"
        )

    usable = []
    names = []
    for entry in synergies:
        if not isinstance(entry, dict):
            problems.append(
                "synergy reply is not valid JSON "
                "{synergies:[{concern, pairs, strength, reason, score}]}"
            )
            continue
        missing_keys = [key for key in _SYNERGY_ENTRY_KEYS if key not in entry]
        if missing_keys:
            problems.append(
                "synergy reply is not valid JSON "
                "{synergies:[{concern, pairs, strength, reason, score}]}"
            )
        if "concern" in entry:
            names.append(entry.get("concern"))
        if not missing_keys:
            usable.append(entry)

    expected = set(keys)
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    missing = [key for key in keys if counts.get(key, 0) == 0]
    extra = []
    seen_extra = set()
    for name in names:
        if name not in expected and name not in seen_extra:
            seen_extra.add(name)
            extra.append(name)
    duplicates = []
    seen_dup = set()
    for name in names:
        if counts.get(name, 0) > 1 and name not in seen_dup:
            seen_dup.add(name)
            duplicates.append(name)
    if missing:
        problems.append("missing concerns: " + ", ".join(str(key) for key in missing))
    if extra:
        problems.append("extra concerns: " + ", ".join(str(key) for key in extra))
    if len(duplicates) == 1:
        problems.append(f"duplicate concern: {duplicates[0]}")
    elif duplicates:
        problems.append(
            "duplicate concerns: " + ", ".join(str(key) for key in duplicates)
        )

    for entry in usable:
        strength = entry.get("strength")
        score = entry.get("score")
        concern = entry.get("concern")
        if strength not in STRENGTH_SCORES:
            problems.append(
                f"concern {concern!r} score {score}, "
                f"must match strength {strength!r}"
            )
        elif score != STRENGTH_SCORES[strength]:
            problems.append(
                f"concern {concern!r} score {score}, "
                f"must be {STRENGTH_SCORES[strength]} for strength {strength!r}"
            )

    if _is_number(total) and total > 0.5:
        problems.append(f"total_synergy_score {total} > 0.5")

    for entry in usable:
        if _reason_empty(entry.get("reason")):
            problems.append(f"concern {entry.get('concern')!r} reason empty")

    given = {"given": data, "reply": parsed}
    for entry in usable:
        problems.extend(check_synergy_entry(entry, product_names, given))

    listed = listed_by_concern
    if listed is None and isinstance(data, dict) and "concern_tiers" in data:
        listed = build_synergy_concern_ingredients(
            {
                "concern_keys": keys,
                "concern_tiers": data.get("concern_tiers"),
                "concentrations": data.get("concentrations"),
            }
        )
    if listed is not None:
        for entry in usable:
            key = entry.get("concern")
            problems.extend(
                concern_list_problems(entry, listed.get(key) or [])
            )

    return parsed, problems


def synergy_feedback_extra(concern_keys) -> list:
    keys = list(concern_keys) if isinstance(concern_keys, list) else []
    listed = ", ".join(str(key) for key in keys)
    return [
        " Return the complete JSON again with exactly one entry for each of "
        f"these {len(keys)} concerns: {listed}."
    ]


def format_synergy_feedback(problems, concern_keys) -> str:
    return format_llm_feedback(problems, synergy_feedback_extra(concern_keys))


def build_synergy_schema(concern_keys) -> dict:
    keys = []
    seen = set()
    for key in concern_keys if isinstance(concern_keys, list) else []:
        name = str(key)
        if name in seen:
            continue
        seen.add(name)
        keys.append(name)
    concern_prop = {"type": "string"}
    if keys:
        concern_prop["enum"] = keys
    return {
        "type": "object",
        "properties": {
            "synergies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concern": concern_prop,
                        "pairs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "strength": {
                            "type": "string",
                            "enum": list(STRENGTH_SCORES),
                        },
                        "reason": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": list(_SYNERGY_ENTRY_KEYS),
                    "additionalProperties": False,
                },
            },
            "total_synergy_score": {"type": "number"},
        },
        "required": ["synergies", "total_synergy_score"],
        "additionalProperties": False,
    }


def validate_synergy_reply(text, concern_keys) -> dict:
    """D.7b. Raises ValueError with the failing check(s)."""
    parsed, problems = collect_synergy_problems(text, concern_keys)
    if problems:
        raise ValueError("\n".join(problems))
    return parsed
