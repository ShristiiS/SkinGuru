from __future__ import annotations

import json

from precompute.flow1.claims_check import (
    INTERACTION_SYNERGY_EMPTY,
    IRRITATION_SYNERGY_EMPTY,
    check_safety_texts,
    product_names_from_context,
)

_REASON_FIELDS = (
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


def last_successful(tool_log, name: str):
    for entry in reversed(_entries(tool_log, name)):
        if entry.get("ok") is True:
            return entry
    return None


def _empty_text(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _is_empty_list(value) -> bool:
    return not isinstance(value, list) or len(value) == 0


def _parse_calc(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def collect_safety_store_problems(
    store_args, calc_result, interaction_result, product_id=None
) -> list:
    problems = []
    args = store_args if isinstance(store_args, dict) else {}
    for field in _REASON_FIELDS:
        if _empty_text(args.get(field)):
            problems.append(f"{field} empty")
    pipeline = interaction_result if isinstance(interaction_result, dict) else {}
    if _is_empty_list(pipeline.get("irritation_synergies")):
        if args.get("irritation_synergy_reasoning") != IRRITATION_SYNERGY_EMPTY:
            problems.append(
                "irritation_synergy_reasoning must be "
                f"{IRRITATION_SYNERGY_EMPTY!r}"
            )
    if _is_empty_list(pipeline.get("irritation_interactions")):
        if args.get("interaction_synergy_reasoning") != INTERACTION_SYNERGY_EMPTY:
            problems.append(
                "interaction_synergy_reasoning must be "
                f"{INTERACTION_SYNERGY_EMPTY!r}"
            )
    names = product_names_from_context(product_id)
    problems.extend(
        check_safety_texts(args, calc_result, pipeline, names)
    )
    return problems


def safety_store_precheck(
    store_args, calc_result, interaction_result, product_id=None
):
    problems = collect_safety_store_problems(
        store_args, calc_result, interaction_result, product_id
    )
    if not problems:
        return None
    return (
        "ERROR: not saved. Fix these and call store_safety_flags again:\n"
        + "\n".join(problems)
    )


def validate_safety_agent(result, interaction_result) -> None:
    """D.9.1–5 plus claims-vs-data."""
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

    calc_entry = last_successful(tool_log, "calculate_and_store_safety_flags")
    calc_result = _parse_calc(
        calc_entry.get("result") if calc_entry else None
    )
    problems = check_safety_texts(
        store_args,
        calc_result,
        pipeline,
        product_names_from_context(),
    )
    if problems:
        raise ValueError(problems[0])
