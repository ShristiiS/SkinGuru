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


def _synergy_entry(synergy: dict, concern_key):
    synergies = synergy.get("synergies") if isinstance(synergy, dict) else None
    if not isinstance(synergies, list):
        return None
    for entry in synergies:
        if isinstance(entry, dict) and entry.get("concern") == concern_key:
            return entry
    return None


def _contributing_format_ok(rows, label: str, concern_key) -> None:
    if not isinstance(rows, list):
        raise ValueError(
            f'concern {concern_key!r} {label} is not a list'
        )
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(
                f'concern {concern_key!r} {label} item missing fields'
            )
        missing = [key for key in _CONTRIBUTING_KEYS if key not in item]
        if missing:
            raise ValueError(
                f'concern {concern_key!r} {label} item missing fields'
            )


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


def _synergy_pairs_format_ok(rows, concern_key) -> list:
    names = _synergy_pair_names(rows)
    if names is None:
        raise ValueError(
            f'concern {concern_key!r} synergy_pairs is not [] or two names'
        )
    return names


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

    first_calc = next(
        i
        for i, entry in enumerate(tool_log)
        if isinstance(entry, dict)
        and entry.get("name") == "calculate_concern_scores"
    )
    first_store = next(
        i
        for i, entry in enumerate(tool_log)
        if isinstance(entry, dict)
        and entry.get("name") == "store_concern_results"
    )
    if first_store < first_calc:
        raise ValueError(
            "store_concern_results called before calculate_concern_scores"
        )

    if max_calls_hit:
        raise ValueError("hit 10 turns")

    store_args = store_entries[-1].get("arguments") or {}
    store_data = store_args.get("data") if isinstance(store_args, dict) else None
    parsed_store = _parse_json(store_data, "store data")
    if not isinstance(parsed_store, list):
        raise ValueError("store data is not valid JSON")
    if len(parsed_store) != _CONCERN_COUNT:
        raise ValueError("store data is not exactly 15 concerns")
    store_keys = []
    for row in parsed_store:
        if not isinstance(row, dict):
            raise ValueError("store data is not exactly 15 concerns")
        store_keys.append(row.get("concern_key"))
    if len(set(store_keys)) != len(store_keys):
        raise ValueError("duplicate concern_key")

    pipeline_id = _as_int(product_id)
    for row in parsed_store:
        row_id = _as_int(row.get("product_id"))
        if row_id != pipeline_id:
            raise ValueError(
                f"product_id {row.get('product_id')!r} != pipeline product_id "
                f"{product_id!r}"
            )

    calc_raw = calc_entries[-1].get("result")
    parsed_calc = _parse_json(calc_raw, "calculate_concern_scores result")
    if not isinstance(parsed_calc, list):
        raise ValueError("calculate_concern_scores result is not a list")
    calc_by_key = {}
    for row in parsed_calc:
        if isinstance(row, dict) and row.get("concern_key") not in calc_by_key:
            calc_by_key[row.get("concern_key")] = row

    for row in parsed_store:
        key = row.get("concern_key")
        calc_row = calc_by_key.get(key)
        if calc_row is None:
            raise ValueError(f'concern {key!r} missing from calculate result')
        if row.get("concern_key") != calc_row.get("concern_key"):
            raise ValueError(
                f'concern {key!r} concern_key {row.get("concern_key")!r} '
                f'≠ calculate {calc_row.get("concern_key")!r}'
            )
        if row.get("concern_score") != calc_row.get("concern_score"):
            raise ValueError(
                f'concern {key!r} concern_score {row.get("concern_score")!r} '
                f'≠ calculate {calc_row.get("concern_score")!r}'
            )
        if row.get("bonus_score") != calc_row.get("bonus_score"):
            raise ValueError(
                f'concern {key!r} bonus_score {row.get("bonus_score")!r} '
                f'≠ calculate {calc_row.get("bonus_score")!r}'
            )
        if not ingredients_equal(
            row.get("concern_contributing_ingredients"),
            calc_row.get("concern_contributing_ingredients"),
        ):
            raise ValueError(
                f'concern {key!r} concern_contributing_ingredients '
                f'≠ calculate'
            )

    for row in parsed_store:
        key = row.get("concern_key")
        calc_row = calc_by_key[key]
        if "bonus_contributing_ingredients" not in row:
            raise ValueError(
                f'concern {key!r} bonus_contributing_ingredients missing'
            )
        if not ingredients_equal(
            row.get("bonus_contributing_ingredients"),
            calc_row.get("concern_contributing_ingredients"),
        ):
            raise ValueError(
                f'concern {key!r} bonus_contributing_ingredients '
                f'≠ calculate concern_contributing_ingredients'
            )

    synergy_obj = synergy if isinstance(synergy, dict) else {}
    for row in parsed_store:
        key = row.get("concern_key")
        entry = _synergy_entry(synergy_obj, key)
        if entry is None:
            raise ValueError(
                f'concern {key!r} missing from synergy reply'
            )
        if row.get("synergy_score") != entry.get("score"):
            raise ValueError(
                f'concern {key!r} synergy_score {row.get("synergy_score")!r} '
                f'≠ synergy {entry.get("score")!r}'
            )
        store_pair = _synergy_pairs_format_ok(row.get("synergy_pairs"), key)
        synergy_pair = _synergy_pair_names(entry.get("pairs"))
        if synergy_pair is None or _pair_key(store_pair) != _pair_key(
            synergy_pair
        ):
            raise ValueError(
                f'concern {key!r} synergy_pairs ≠ synergy pairs'
            )
        if _empty_text(row.get("synergy_reasoning")):
            raise ValueError(f'concern {key!r} synergy_reasoning empty')
        if synergy_pair:
            text = row.get("synergy_reasoning")
            text_cf = text.casefold()
            for name in synergy_pair:
                if name.casefold() not in text_cf:
                    raise ValueError(
                        f'concern {key!r} synergy_reasoning missing {name!r}'
                    )

    for row in parsed_store:
        key = row.get("concern_key")
        if _empty_text(row.get("concern_reasoning")):
            raise ValueError(f'concern {key!r} concern_reasoning empty')
        if _empty_text(row.get("bonus_reasoning")):
            raise ValueError(f'concern {key!r} bonus_reasoning empty')
        if _empty_text(row.get("full_explanation")):
            raise ValueError(f'concern {key!r} full_explanation empty')

    for row in parsed_store:
        key = row.get("concern_key")
        _contributing_format_ok(
            row.get("concern_contributing_ingredients"),
            "concern_contributing_ingredients",
            key,
        )
        _contributing_format_ok(
            row.get("bonus_contributing_ingredients"),
            "bonus_contributing_ingredients",
            key,
        )
