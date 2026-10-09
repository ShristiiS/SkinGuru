from __future__ import annotations

import json
import re

from precompute.llm_call import format_llm_feedback

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


def collect_synergy_problems(text, concern_keys) -> tuple:
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
