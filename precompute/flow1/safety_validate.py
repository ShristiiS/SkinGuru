from __future__ import annotations

IRRITATION_SYNERGY_EMPTY = "No ingredient-level irritation synergies found"
INTERACTION_SYNERGY_EMPTY = (
    "No harmful interaction pairs causing irritation were found"
)
_REASON_FIELDS = (
    "safety_reasoning",
    "full_explanation",
    "irritation_synergy_reasoning",
    "interaction_synergy_reasoning",
)


def _entries(tool_log, name: str) -> list:
    return [
        entry
        for entry in tool_log or []
        if isinstance(entry, dict) and entry.get("name") == name
    ]


def _last_ok(entries, name: str) -> None:
    if entries[-1].get("ok") is not True:
        raise ValueError(f"{name} returned an error")


def _empty_text(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _is_empty_list(value) -> bool:
    return not isinstance(value, list) or len(value) == 0


def validate_safety_agent(result, interaction_result) -> None:
    """D.9.1–5."""
    tool_log = getattr(result, "tool_log", None) or []
    max_calls_hit = bool(getattr(result, "max_calls_hit", False))
    pipeline = interaction_result if isinstance(interaction_result, dict) else {}

    calc_entries = _entries(tool_log, "calculate_and_store_safety_flags")
    if not calc_entries:
        raise ValueError("calculate_and_store_safety_flags not called")
    _last_ok(calc_entries, "calculate_and_store_safety_flags")

    store_entries = _entries(tool_log, "store_safety_flags")
    if not store_entries:
        raise ValueError("store_safety_flags not called")
    _last_ok(store_entries, "store_safety_flags")

    if max_calls_hit:
        raise ValueError("hit 10 turns")

    store_args = store_entries[-1].get("arguments") or {}
    if not isinstance(store_args, dict):
        store_args = {}
    for field in _REASON_FIELDS:
        if _empty_text(store_args.get(field)):
            raise ValueError(f"{field} empty")

    if _is_empty_list(pipeline.get("irritation_synergies")):
        if store_args.get("irritation_synergy_reasoning") != IRRITATION_SYNERGY_EMPTY:
            raise ValueError(
                "irritation_synergy_reasoning must be "
                f"{IRRITATION_SYNERGY_EMPTY!r}"
            )
    if _is_empty_list(pipeline.get("irritation_interactions")):
        if (
            store_args.get("interaction_synergy_reasoning")
            != INTERACTION_SYNERGY_EMPTY
        ):
            raise ValueError(
                "interaction_synergy_reasoning must be "
                f"{INTERACTION_SYNERGY_EMPTY!r}"
            )
