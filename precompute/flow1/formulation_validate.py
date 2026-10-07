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


def _last_ok(entries, name: str) -> None:
    if entries[-1].get("ok") is not True:
        raise ValueError(f"{name} returned an error")


def last_successful(tool_log, name: str):
    for entry in reversed(_entries(tool_log, name)):
        if entry.get("ok") is True:
            return entry
    return None


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


def _try_parse_json(raw, label: str):
    try:
        return _parse_json(raw, label), None
    except ValueError as exc:
        return None, str(exc)


def _botanical_count(raw):
    parsed = _parse_json(raw, "count_botanicals result")
    if isinstance(parsed, dict) and "botanical_count" in parsed:
        return parsed["botanical_count"]
    return parsed


def collect_formulation_store_problems(store_data, tool_log, node22) -> list:
    problems = []
    pipeline = node22 if isinstance(node22, dict) else {}
    count_entry = last_successful(tool_log, "count_botanicals")
    calc_entry = last_successful(tool_log, "calculate_formulation_score")

    count_value = None
    if count_entry is not None:
        try:
            count_value = _botanical_count(count_entry.get("result"))
        except ValueError as exc:
            problems.append(str(exc))

    calc_args = {}
    if calc_entry is not None and isinstance(calc_entry.get("arguments"), dict):
        calc_args = calc_entry.get("arguments")
    if count_value is not None and calc_args.get("botanical_count") != count_value:
        problems.append(
            f"botanical_count {calc_args.get('botanical_count')}, "
            f"must be {count_value}"
        )

    parsed_store, store_err = _try_parse_json(
        store_data, "store_formulation_score data"
    )
    if store_err:
        problems.append("store_formulation_score data is not valid JSON")
        return problems
    if not isinstance(parsed_store, dict):
        problems.append("store_formulation_score data is not valid JSON")
        return problems

    pipeline_id = _as_int(pipeline.get("product_id"))
    if _as_int(parsed_store.get("product_id")) != pipeline_id:
        problems.append(
            f"product_id {parsed_store.get('product_id')}, must be "
            f"{pipeline.get('product_id')}"
        )
    if parsed_store.get("formulation_product_type") != pipeline.get(
        "formulation_product_type"
    ):
        problems.append(
            f"formulation_product_type "
            f"{parsed_store.get('formulation_product_type')}, must be "
            f"{pipeline.get('formulation_product_type')}"
        )

    parsed_calc = None
    if calc_entry is not None:
        parsed_calc, calc_err = _try_parse_json(
            calc_entry.get("result"), "calculate_formulation_score result"
        )
        if calc_err:
            problems.append(calc_err)
        elif not isinstance(parsed_calc, dict):
            problems.append(
                "calculate_formulation_score result is not an object"
            )
            parsed_calc = None

    if isinstance(parsed_calc, dict):
        if parsed_store.get("formulation_score") != parsed_calc.get("final_score"):
            problems.append(
                f"formulation_score {parsed_store.get('formulation_score')}, "
                f"must be {parsed_calc.get('final_score')}"
            )
        for name in _COMPONENTS:
            component = parsed_calc.get(name)
            if not isinstance(component, dict):
                problems.append(f"{name} missing from calculate result")
                continue
            score_key = f"{name}_score"
            ingredients_key = f"{name}_ingredients"
            if parsed_store.get(score_key) != component.get("score"):
                problems.append(
                    f"{score_key} {parsed_store.get(score_key)}, "
                    f"must be {component.get('score')}"
                )
            if not values_equal(parsed_store.get(ingredients_key), component):
                problems.append(
                    f"{ingredients_key}, must match calculate {name} component"
                )

    for field in _REASON_FIELDS:
        if _empty_text(parsed_store.get(field)):
            problems.append(f"{field} empty")
    return problems


def formulation_store_precheck(store_data, tool_log, node22):
    if last_successful(tool_log, "calculate_formulation_score") is None:
        return (
            "ERROR: call calculate_formulation_score successfully before storing."
        )
    if last_successful(tool_log, "count_botanicals") is None:
        return "ERROR: call count_botanicals successfully before storing."
    problems = collect_formulation_store_problems(store_data, tool_log, node22)
    if not problems:
        return None
    return (
        "ERROR: not saved. Fix these and call store_formulation_score again:\n"
        + "\n".join(problems)
    )


def validate_formulation_agent(result, node22: dict) -> None:
    """D.10.1–7."""
    tool_log = getattr(result, "tool_log", None) or []
    max_calls_hit = bool(getattr(result, "max_calls_hit", False))

    for name in _REQUIRED_GETS:
        entries = _entries(tool_log, name)
        if not entries:
            raise ValueError(f"{name} not called")
        _last_ok(entries, name)

    store_entries = _entries(tool_log, "store_formulation_score")
    if not store_entries:
        raise ValueError("store_formulation_score not called")
    _last_ok(store_entries, "store_formulation_score")

    if max_calls_hit:
        raise ValueError("hit 10 turns")

    store_args = store_entries[-1].get("arguments") or {}
    store_data = store_args.get("data") if isinstance(store_args, dict) else None
    problems = collect_formulation_store_problems(store_data, tool_log, node22)
    if problems:
        raise ValueError(problems[0])
