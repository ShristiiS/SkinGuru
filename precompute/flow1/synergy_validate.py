from __future__ import annotations

import json
import re

STRENGTH_SCORES = {
    "Strong": 0.166,
    "Moderate": 0.10,
    "Weak": 0.05,
    "None": 0,
}
_SYNERGY_ENTRY_KEYS = ("concern", "pairs", "strength", "reason", "score")


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


def validate_synergy_reply(text, concern_keys) -> dict:
    """D.7b. Raises ValueError with the failing check."""
    keys = list(concern_keys) if isinstance(concern_keys, list) else []
    cleaned = _cleanup_raw(text)
    try:
        parsed = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
    except Exception:
        raise ValueError("synergy reply is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("synergy reply is not valid JSON")
    synergies = parsed.get("synergies")
    if "total_synergy_score" not in parsed or not isinstance(synergies, list):
        raise ValueError(
            "synergy reply is not valid JSON "
            "{synergies:[...], total_synergy_score}"
        )
    total = parsed.get("total_synergy_score")
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise ValueError(
            "synergy reply is not valid JSON "
            "{synergies:[...], total_synergy_score}"
        )
    for entry in synergies:
        if not isinstance(entry, dict):
            raise ValueError(
                "synergy reply is not valid JSON "
                "{synergies:[{concern, pairs, strength, reason, score}]}"
            )
        missing = [key for key in _SYNERGY_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(
                "synergy reply is not valid JSON "
                "{synergies:[{concern, pairs, strength, reason, score}]}"
            )

    names = [entry.get("concern") for entry in synergies]
    if len(names) != len(keys) or len(set(names)) != len(names) or set(names) != set(
        keys
    ):
        raise ValueError("synergy reply is not one entry per concern")

    for entry in synergies:
        strength = entry.get("strength")
        if strength not in STRENGTH_SCORES:
            raise ValueError(
                f'concern {entry.get("concern")!r} score does not match strength'
            )
        if entry.get("score") != STRENGTH_SCORES[strength]:
            raise ValueError(
                f'concern {entry.get("concern")!r} score does not match strength'
            )

    if total > 0.5:
        raise ValueError("total_synergy_score > 0.5")

    for entry in synergies:
        if _reason_empty(entry.get("reason")):
            raise ValueError(f'concern {entry.get("concern")!r} reason empty')

    return parsed
