from __future__ import annotations

import json
import re

from precompute.flow1.value_equal import values_equal

_REQUIRED_GETS = (
    "get_product_ingredients_with_functions",
    "count_botanicals",
    "calculate_formulation_score",
)
_COMPONENTS = (
    "structural_completeness",
    "preservation_design",
    "stability_engineering",
    "filler_ratio",
)
_REASON_FIELDS = (
    "formulation_reasoning",
    "structural_completeness_reasoning",
    "preservation_design_reasoning",
    "stability_engineering_reasoning",
    "filler_ratio_reasoning",
    "full_explanation",
)


def _cleanup_raw(raw):
    cleaned = raw
    if isinstance(cleaned, str):
        cleaned = cleaned.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            cleaned = cleaned.strip()
    return cleaned


def _entries(tool_log, name: str) -> list:
    return [
        entry
        for entry in tool_log or []
        if isinstance(entry, dict) and entry.get("name") == name
    ]


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_text(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _parse_json(raw, label: str):
    cleaned = _cleanup_raw(raw)
    try:
        return json.loads(cleaned) if isinstance(cleaned, str) else cleaned
    except Exception:
        raise ValueError(f"{label} is not valid JSON") from None


def _botanical_count(raw):
    parsed = _parse_json(raw, "count_botanicals result")
    if isinstance(parsed, dict) and "botanical_count" in parsed:
        return parsed["botanical_count"]
    return parsed


def validate_formulation_agent(result, node22: dict) -> None:
    """D.10.1–7."""
    tool_log = getattr(result, "tool_log", None) or []
    max_calls_hit = bool(getattr(result, "max_calls_hit", False))
    pipeline = node22 if isinstance(node22, dict) else {}

    for name in _REQUIRED_GETS:
        entries = _entries(tool_log, name)
        if not entries:
            raise ValueError(f"{name} not called")
        if any(entry.get("ok") is not True for entry in entries):
            raise ValueError(f"{name} returned an error")

    store_entries = _entries(tool_log, "store_formulation_score")
    if not store_entries:
        raise ValueError("store_formulation_score not called")
    if any(entry.get("ok") is not True for entry in store_entries):
        raise ValueError("store_formulation_score returned an error")

    if max_calls_hit:
        raise ValueError("hit 10 turns")

    count_raw = _entries(tool_log, "count_botanicals")[-1].get("result")
    count_value = _botanical_count(count_raw)
    calc_args = _entries(tool_log, "calculate_formulation_score")[-1].get(
        "arguments"
    )
    if not isinstance(calc_args, dict):
        calc_args = {}
    if calc_args.get("botanical_count") != count_value:
        raise ValueError(
            f"botanical_count {calc_args.get('botanical_count')!r} "
            f"≠ count_botanicals {count_value!r}"
        )

    store_args = store_entries[-1].get("arguments") or {}
    store_data = store_args.get("data") if isinstance(store_args, dict) else None
    parsed_store = _parse_json(store_data, "store data")
    if not isinstance(parsed_store, dict):
        raise ValueError("store_formulation_score data is not valid JSON")

    pipeline_id = _as_int(pipeline.get("product_id"))
    if _as_int(parsed_store.get("product_id")) != pipeline_id:
        raise ValueError(
            f"product_id {parsed_store.get('product_id')!r} != pipeline "
            f"product_id {pipeline.get('product_id')!r}"
        )
    if parsed_store.get("formulation_product_type") != pipeline.get(
        "formulation_product_type"
    ):
        raise ValueError(
            f"formulation_product_type "
            f"{parsed_store.get('formulation_product_type')!r} "
            f"≠ pass-through {pipeline.get('formulation_product_type')!r}"
        )

    calc_raw = _entries(tool_log, "calculate_formulation_score")[-1].get(
        "result"
    )
    parsed_calc = _parse_json(calc_raw, "calculate_formulation_score result")
    if not isinstance(parsed_calc, dict):
        raise ValueError("calculate_formulation_score result is not an object")
    breakdown = parsed_calc.get("score_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = {}
    if parsed_store.get("formulation_score") != parsed_calc.get("final_score"):
        raise ValueError(
            f"formulation_score {parsed_store.get('formulation_score')!r} "
            f"≠ final_score {parsed_calc.get('final_score')!r}"
        )
    for name in _COMPONENTS:
        component = breakdown.get(name)
        if not isinstance(component, dict):
            raise ValueError(f"{name} missing from calculate score_breakdown")
        score_key = f"{name}_score"
        ingredients_key = f"{name}_ingredients"
        if parsed_store.get(score_key) != component.get("score"):
            raise ValueError(
                f"{score_key} {parsed_store.get(score_key)!r} "
                f"≠ {name}.score {component.get('score')!r}"
            )
        if not values_equal(parsed_store.get(ingredients_key), component):
            raise ValueError(
                f"{ingredients_key} ≠ calculate {name} component"
            )

    for field in _REASON_FIELDS:
        if _empty_text(parsed_store.get(field)):
            raise ValueError(f"{field} empty")
