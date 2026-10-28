from __future__ import annotations

import copy
import json
import re

from precompute.concentration.helpers import compact_json
from precompute.flow1.claims_check import (
    check_concern_texts,
    product_names_from_context,
)
from precompute.flow1.value_equal import ingredients_equal

_TEXT_FIELDS = (
    "concern_reasoning",
    "bonus_reasoning",
    "synergy_reasoning",
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


def _first_ok_index(tool_log, name: str):
    for index, entry in enumerate(tool_log or []):
        if (
            isinstance(entry, dict)
            and entry.get("name") == name
            and entry.get("ok") is True
        ):
            return index
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


def _synergy_entry(synergy: dict, concern_key):
    synergies = synergy.get("synergies") if isinstance(synergy, dict) else None
    if not isinstance(synergies, list):
        return None
    for entry in synergies:
        if isinstance(entry, dict) and entry.get("concern") == concern_key:
            return entry
    return None


def _synergy_pair_names(value):
    """[] or exactly two non-empty name strings. None if the shape is wrong."""
    if not isinstance(value, list):
        return None
    if len(value) == 0:
        return []
    if (
        len(value) == 2
        and all(isinstance(name, str) and name.strip() for name in value)
    ):
        return [name.strip() for name in value]
    return None


def _pair_key(names):
    return frozenset(name.casefold() for name in names)


def _strip_surrounding_reason(value: str) -> str:
    cleaned = value.strip().strip(".").strip()
    return cleaned.casefold()


def _reason_equal(left, right) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return _strip_surrounding_reason(left) == _strip_surrounding_reason(right)


def _name_or_first_word_in(name: str, text_cf: str) -> bool:
    full = name.strip().casefold()
    if full and full in text_cf:
        return True
    parts = name.strip().split()
    if not parts:
        return False
    first = parts[0].casefold()
    return bool(first) and first in text_cf


def _synergy_reasoning_ok(text, entry) -> bool:
    if _empty_text(text):
        return False
    if not isinstance(entry, dict):
        return False
    if entry.get("strength") == "None":
        return True
    if _reason_equal(text, entry.get("reason")):
        return True
    names = _synergy_pair_names(entry.get("pairs"))
    if names is not None and len(names) == 2:
        text_cf = text.casefold()
        if all(_name_or_first_word_in(name, text_cf) for name in names):
            return True
    return False


def parse_calculate_concern_rows(raw):
    parsed, err = _try_parse_json(raw, "calculate_concern_scores result")
    if err or not isinstance(parsed, list):
        return None
    rows = [
        row
        for row in parsed
        if isinstance(row, dict) and row.get("concern_key") is not None
    ]
    if not rows:
        return None
    return rows


def _trace_value(value) -> str:
    try:
        return compact_json(value)
    except Exception:
        return str(value)


def _sql_values(calc_row: dict) -> dict:
    contributing = copy.deepcopy(calc_row.get("concern_contributing_ingredients"))
    bonus = calc_row.get("bonus_contributing_ingredients")
    if bonus is None:
        bonus = copy.deepcopy(contributing)
    else:
        bonus = copy.deepcopy(bonus)
    return {
        "concern_score": calc_row.get("concern_score"),
        "bonus_score": calc_row.get("bonus_score"),
        "concern_contributing_ingredients": contributing,
        "bonus_contributing_ingredients": bonus,
    }


def _synergy_values(synergy, key) -> dict:
    entry = _synergy_entry(synergy if isinstance(synergy, dict) else {}, key)
    if not isinstance(entry, dict):
        return {"synergy_score": 0, "synergy_pairs": []}
    pairs = entry.get("pairs")
    if pairs is None:
        pairs = []
    else:
        pairs = copy.deepcopy(pairs)
    return {"synergy_score": entry.get("score"), "synergy_pairs": pairs}


def _synergy_pairs_equal(left, right) -> bool:
    left_names = _synergy_pair_names(left)
    right_names = _synergy_pair_names(right)
    if left_names is None or right_names is None:
        return False
    return _pair_key(left_names) == _pair_key(right_names)


def _values_mismatch(field, agent, source) -> bool:
    if field in (
        "concern_contributing_ingredients",
        "bonus_contributing_ingredients",
    ):
        return not ingredients_equal(agent, source)
    if field == "synergy_pairs":
        return not _synergy_pairs_equal(agent, source)
    return agent != source


def _empty_texts() -> dict:
    return {field: "" for field in _TEXT_FIELDS}


def correct_concern_store_values(store_data, calc_rows, product_id, synergy):
    """Overwrite value columns from SQL / Synergy. Return (json, traces, added_keys)."""
    traces = []
    added_keys = set()
    parsed_store, store_err = _try_parse_json(store_data, "store data")
    if store_err or not isinstance(parsed_store, list):
        return store_data, traces, added_keys
    if not isinstance(calc_rows, list) or not calc_rows:
        return store_data, traces, added_keys

    agent_by_key = {}
    for row in parsed_store:
        if not isinstance(row, dict):
            continue
        key = row.get("concern_key")
        if key in agent_by_key:
            continue
        agent_by_key[key] = row

    calc_by_key = {}
    sql_keys = []
    for calc_row in calc_rows:
        if not isinstance(calc_row, dict):
            continue
        key = calc_row.get("concern_key")
        if key is None or key in calc_by_key:
            continue
        calc_by_key[key] = calc_row
        sql_keys.append(key)

    corrected = []
    for key in sql_keys:
        sql_vals = _sql_values(calc_by_key[key])
        syn_vals = _synergy_values(synergy, key)
        agent_row = agent_by_key.get(key)
        if agent_row is None:
            row = {
                "product_id": product_id,
                "concern_key": key,
            }
            row.update(sql_vals)
            row.update(syn_vals)
            row.update(_empty_texts())
            traces.append(
                f"concern_correction: {key} missing from agent output "
                "-> added from SQL, texts empty"
            )
            added_keys.add(key)
            corrected.append(row)
            continue
        row = dict(agent_row)
        row["product_id"] = product_id
        row["concern_key"] = key
        replacements = (
            ("concern_score", sql_vals["concern_score"], "sql"),
            ("bonus_score", sql_vals["bonus_score"], "sql"),
            (
                "concern_contributing_ingredients",
                sql_vals["concern_contributing_ingredients"],
                "sql",
            ),
            (
                "bonus_contributing_ingredients",
                sql_vals["bonus_contributing_ingredients"],
                "sql",
            ),
            ("synergy_score", syn_vals["synergy_score"], "synergy"),
            ("synergy_pairs", syn_vals["synergy_pairs"], "synergy"),
        )
        for field, source, origin in replacements:
            agent_val = row.get(field)
            if _values_mismatch(field, agent_val, source):
                traces.append(
                    f"concern_correction: {key}.{field} "
                    f"agent={_trace_value(agent_val)} -> "
                    f"{origin}={_trace_value(source)}"
                )
                row[field] = source
        corrected.append(row)

    for key in agent_by_key:
        if key not in calc_by_key:
            traces.append(
                f"concern_correction: {key} extra in agent output -> removed"
            )

    if not traces:
        return store_data, traces, added_keys
    return compact_json(corrected), traces, added_keys


def _rows_for_text_checks(rows, calc_by_key, skip_keys) -> list:
    skip = set(skip_keys or ())
    checked = []
    seen = set()
    for row in rows:
        key = row.get("concern_key")
        if key in skip or key not in calc_by_key or key in seen:
            continue
        seen.add(key)
        checked.append(row)
    return checked


def collect_concern_store_problems(
    store_data,
    calc_entry,
    product_id,
    synergy,
    concern_data=None,
    skip_keys=None,
) -> list:
    problems = []
    parsed_store, store_err = _try_parse_json(store_data, "store data")
    if store_err:
        return [store_err]
    if not isinstance(parsed_store, list):
        return ["store data is not valid JSON"]
    rows = [row for row in parsed_store if isinstance(row, dict)]

    parsed_calc = None
    if calc_entry is not None:
        parsed_calc, calc_err = _try_parse_json(
            calc_entry.get("result"), "calculate_concern_scores result"
        )
        if calc_err:
            problems.append(calc_err)
        elif not isinstance(parsed_calc, list):
            problems.append("calculate_concern_scores result is not a list")
            parsed_calc = None

    calc_by_key = {}
    if isinstance(parsed_calc, list):
        for row in parsed_calc:
            if isinstance(row, dict) and row.get("concern_key") not in calc_by_key:
                calc_by_key[row.get("concern_key")] = row

    checked_rows = _rows_for_text_checks(rows, calc_by_key, skip_keys)
    synergy_obj = synergy if isinstance(synergy, dict) else {}
    for row in checked_rows:
        key = row.get("concern_key")
        entry = _synergy_entry(synergy_obj, key)
        if _empty_text(row.get("synergy_reasoning")):
            problems.append(f"concern {key!r} synergy_reasoning empty")
        elif not _synergy_reasoning_ok(row.get("synergy_reasoning"), entry):
            problems.append(f"concern {key!r} synergy_reasoning does not match")
        if _empty_text(row.get("concern_reasoning")):
            problems.append(f"concern {key!r} concern_reasoning empty")
        if _empty_text(row.get("bonus_reasoning")):
            problems.append(f"concern {key!r} bonus_reasoning empty")
        if _empty_text(row.get("full_explanation")):
            problems.append(f"concern {key!r} full_explanation empty")
    problems.extend(
        check_concern_texts(
            checked_rows,
            calc_by_key,
            synergy,
            product_names_from_context(product_id),
            concern_data,
        )
    )
    return problems


def concern_store_precheck(
    store_data,
    tool_log,
    product_id,
    synergy,
    concern_data=None,
    calc_entry=None,
    skip_keys=None,
):
    if calc_entry is None:
        calc_entry = last_successful(tool_log, "calculate_concern_scores")
    if calc_entry is None:
        return (
            "ERROR: call calculate_concern_scores successfully before storing."
        )
    problems = collect_concern_store_problems(
        store_data,
        calc_entry,
        product_id,
        synergy,
        concern_data,
        skip_keys=skip_keys,
    )
    if not problems:
        return None
    return (
        "ERROR: not saved. Fix these and call store_concern_results again:\n"
        + "\n".join(problems)
    )


def validate_concern_agent(result, product_id, synergy) -> None:
    """D.7c.1–11. Raises ValueError with the failing check."""
    tool_log = getattr(result, "tool_log", None) or []
    max_calls_hit = bool(getattr(result, "max_calls_hit", False))

    calc_entries = _entries(tool_log, "calculate_concern_scores")
    if not calc_entries:
        raise ValueError("calculate_concern_scores not called")
    _last_ok(calc_entries, "calculate_concern_scores")

    store_entries = _entries(tool_log, "store_concern_results")
    if not store_entries:
        raise ValueError("store_concern_results not called")
    _last_ok(store_entries, "store_concern_results")

    first_calc = _first_ok_index(tool_log, "calculate_concern_scores")
    first_store = _first_ok_index(tool_log, "store_concern_results")
    if (
        first_calc is not None
        and first_store is not None
        and first_store < first_calc
    ):
        raise ValueError(
            "store_concern_results called before calculate_concern_scores"
        )

    if max_calls_hit:
        raise ValueError("hit 10 turns")

    store_args = store_entries[-1].get("arguments") or {}
    store_data = store_args.get("data") if isinstance(store_args, dict) else None
    calc_entry = last_successful(tool_log, "calculate_concern_scores")
    problems = collect_concern_store_problems(
        store_data, calc_entry, product_id, synergy
    )
    if problems:
        raise ValueError(problems[0])
