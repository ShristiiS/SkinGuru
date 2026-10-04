from __future__ import annotations

import json
import math


def round2(value) -> float:
    """JS Math.round(v * 100) / 100 — halves toward +∞, not banker's round."""
    return math.floor(value * 100 + 0.5) / 100


def js_number_str(value) -> str:
    """JS String(number): whole numbers have no .0."""
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if number == int(number):
        return str(int(number))
    return repr(number)


def js_percent(value) -> str:
    """JS `conc + '%'`."""
    return js_number_str(value) + "%"


def compact_json(value) -> str:
    """JS JSON.stringify: compact, insertion order, no ASCII escaping.

    Whole-number floats are written without .0 (10.0 → 10); other floats as
    repr, matching JS ToString/JSON.stringify for these values.
    """
    return _compact_json(value)


def _compact_json(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return js_number_str(value)
    if isinstance(value, dict):
        parts = [
            json.dumps(str(key), ensure_ascii=False) + ":" + _compact_json(item)
            for key, item in value.items()
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_compact_json(item) for item in value) + "]"
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
