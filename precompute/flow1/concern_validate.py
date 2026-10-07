from __future__ import annotations

import json
import re

from precompute.flow1.value_equal import ingredients_equal

_CONCERN_COUNT = 15
_CONTRIBUTING_KEYS = (
    "ingredient_name",
    "tier",
    "tier_value",
    "effective_value",
    "concentration_status",
    "required_concentration",
    "approximate_concentration",
    "mechanism",
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


def _contributing_format_problems(rows, label: str, concern_key) -> list:
    if not isinstance(rows, list):
        return [f"concern {concern_key!r} {label} is not a list"]
    problems = []
    for item in rows:
        if not isinstance(item, dict):
            problems.append(
                f"concern {concern_key!r} {label} item missing fields"
            )
            continue
        missing = [key for key in _CONTRIBUTING_KEYS if key not in item]
        if missing:
            problems.append(
                f"concern {concern_key!r} {label} item missing fields"
            )
    return problems


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


def collect_concern_store_problems(
    store_data, calc_entry, product_id, synergy
) -> list:
    problems = []
    parsed_store, store_err = _try_parse_json(store_data, "store data")
    if store_err:
        return [store_err]
    if not isinstance(parsed_store, list):
        return ["store data is not valid JSON"]
    if len(parsed_store) != _CONCERN_COUNT:
        problems.append("store data is not exactly 15 concerns")
    store_keys = []
    rows = []
    for row in parsed_store:
        if not isinstance(row, dict):
            problems.append("store data is not exactly 15 concerns")
            continue
        rows.append(row)
        store_keys.append(row.get("concern_key"))
    if len(store_keys) != len(set(store_keys)):
        problems.append("duplicate concern_key")

    pipeline_id = _as_int(product_id)
    for row in rows:
        row_id = _as_int(row.get("product_id"))
        if row_id != pipeline_id:
            problems.append(
                f"product_id {row.get('product_id')}, must be {product_id}"
            )

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

    for row in rows:
        key = row.get("concern_key")
        calc_row = calc_by_key.get(key)
        if calc_row is None:
            problems.append(f"concern {key!r} missing from calculate result")
            continue
        if row.get("concern_score") != calc_row.get("concern_score"):
            problems.append(
                f"concern {key!r} concern_score {row.get('concern_score')}, "
                f"must be {calc_row.get('concern_score')}"
            )
        if row.get("bonus_score") != calc_row.get("bonus_score"):
            problems.append(
                f"concern {key!r} bonus_score {row.get('bonus_score')}, "
                f"must be {calc_row.get('bonus_score')}"
            )
        if not ingredients_equal(
            row.get("concern_contributing_ingredients"),
            calc_row.get("concern_contributing_ingredients"),
        ):
            problems.append(
                f"concern {key!r} concern_contributing_ingredients, "
                f"must match calculate"
            )
        if "bonus_contributing_ingredients" not in row:
            problems.append(
                f"concern {key!r} bonus_contributing_ingredients missing"
            )
        elif not ingredients_equal(
            row.get("bonus_contributing_ingredients"),
            calc_row.get("concern_contributing_ingredients"),
        ):
            problems.append(
                f"concern {key!r} bonus_contributing_ingredients, "
                f"must match calculate concern_contributing_ingredients"
            )

    synergy_obj = synergy if isinstance(synergy, dict) else {}
    for row in rows:
        key = row.get("concern_key")
        entry = _synergy_entry(synergy_obj, key)
        if entry is None:
            problems.append(f"concern {key!r} missing from synergy reply")
            continue
        if row.get("synergy_score") != entry.get("score"):
            problems.append(
                f"concern {key!r} synergy_score {row.get('synergy_score')}, "
                f"must be {entry.get('score')}"
            )
        store_pair = _synergy_pair_names(row.get("synergy_pairs"))
        if store_pair is None:
            problems.append(
                f"concern {key!r} synergy_pairs is not [] or two names"
            )
        else:
            synergy_pair = _synergy_pair_names(entry.get("pairs"))
            if synergy_pair is None or _pair_key(store_pair) != _pair_key(
                synergy_pair
            ):
                problems.append(
                    f"concern {key!r} synergy_pairs, must match synergy pairs"
                )
        if _empty_text(row.get("synergy_reasoning")):
            problems.append(f"concern {key!r} synergy_reasoning empty")
        elif not _synergy_reasoning_ok(row.get("synergy_reasoning"), entry):
            problems.append(f"concern {key!r} synergy_reasoning does not match")

    for row in rows:
        key = row.get("concern_key")
        if _empty_text(row.get("concern_reasoning")):
            problems.append(f"concern {key!r} concern_reasoning empty")
        if _empty_text(row.get("bonus_reasoning")):
            problems.append(f"concern {key!r} bonus_reasoning empty")
        if _empty_text(row.get("full_explanation")):
            problems.append(f"concern {key!r} full_explanation empty")
        problems.extend(
            _contributing_format_problems(
                row.get("concern_contributing_ingredients"),
                "concern_contributing_ingredients",
                key,
            )
        )
        problems.extend(
            _contributing_format_problems(
                row.get("bonus_contributing_ingredients"),
                "bonus_contributing_ingredients",
                key,
            )
        )
    return problems


def concern_store_precheck(store_data, tool_log, product_id, synergy):
    calc_entry = last_successful(tool_log, "calculate_concern_scores")
    if calc_entry is None:
        return (
            "ERROR: call calculate_concern_scores successfully before storing."
        )
    problems = collect_concern_store_problems(
        store_data, calc_entry, product_id, synergy
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
