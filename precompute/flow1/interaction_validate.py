from __future__ import annotations

import json
import re

from tracing import trace_step
from tracing.debug import to_jsonable

_ALLOWED_WORST_CASE = (
    "avoid",
    "caution-high",
    "caution-medium",
    "caution-low",
    "none",
)
_REQUIRED_ARRAY_FIELDS = (
    "interactions",
    "irritation_interactions",
    "interaction_synergies",
    "irritation_synergies",
)
_REQUIRED_PAIR_KEYS = (
    "ingredient_a",
    "ingredient_b",
    "interaction_type",
    "severity",
    "source",
    "reason",
)
_NOT_CALLED_IRRITATION_FLAGS = (
    "NOT_CALLED: get_ingredient_irritation_flags — required every run "
    "to build interaction_synergies correctly"
)
_GET_TOOLS = (
    "get_active_ingredients_for_interactions",
    "get_predefined_interactions",
    "get_ingredient_irritation_flags",
)
_OUTPUT_ARRAYS = (
    "interactions",
    "irritation_interactions",
    "interaction_synergies",
    "irritation_synergies",
)


def _parse_obs(step):
    obs = None
    if step is not None:
        obs = step.get("result")
        if obs is None:
            obs = step.get("observation")
    try:
        return json.loads(obs) if isinstance(obs, str) else obs
    except Exception:
        return None


def _flat_one(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _ingredient_name(item):
    if isinstance(item, dict):
        return item.get("ingredient_name")
    return None


def _pair_name(pair, key):
    if not isinstance(pair, dict):
        return None
    return pair.get(key)


def _cleanup_raw(raw):
    cleaned = raw
    if isinstance(cleaned, str):
        cleaned = cleaned.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            cleaned = cleaned.strip()
    return cleaned


def _irritant_set_from_tool_log(tool_log):
    steps = tool_log or []
    irritation_flags_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("name") == "get_ingredient_irritation_flags"
        ),
        None,
    )
    irritation_flags_obs = (
        _parse_obs(irritation_flags_step) if irritation_flags_step else None
    )
    if isinstance(irritation_flags_obs, list):
        irritant_list = [
            _ingredient_name(item) for item in _flat_one(irritation_flags_obs)
        ]
    else:
        irritant_list = []
    return irritation_flags_step, set(irritant_list)


def validate_interaction_result(raw, tool_log=None) -> dict:
    with trace_step("validate_interaction_result") as fields:
        problems = []
        irritation_flags_step, irritant_set = _irritant_set_from_tool_log(
            tool_log
        )
        fields["debug_input"] = {
            "raw": to_jsonable(raw),
            "irritant_set": to_jsonable(
                sorted(name for name in irritant_set if name is not None)
            ),
        }

        try:
            cleaned = _cleanup_raw(raw)
            parsed = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
        except Exception:
            message = (
                "INVALID_JSON: Interaction Builder Agent did not return "
                "valid JSON: " + str(raw)[:300]
            )
            fields["debug_error"] = [message]
            raise ValueError(message)

        data = parsed if isinstance(parsed, dict) else {}

        for field in _REQUIRED_ARRAY_FIELDS:
            if not isinstance(data.get(field), list):
                problems.append(
                    f'MISSING_OR_INVALID_FIELD: "{field}" is missing or not an array'
                )

        if (
            not isinstance(data.get("worst_case"), str)
            or data.get("worst_case") not in _ALLOWED_WORST_CASE
        ):
            problems.append(f'INVALID_WORST_CASE: got "{data.get("worst_case")}"')

        if not isinstance(data.get("reason"), str):
            problems.append('MISSING_FIELD: "reason" is missing or not a string')

        if isinstance(data.get("interactions"), list):
            for i, pair in enumerate(data["interactions"]):
                for key in _REQUIRED_PAIR_KEYS:
                    value = pair.get(key) if isinstance(pair, dict) else None
                    if value is None or value == "":
                        problems.append(f'interactions[{i}] missing "{key}"')

        worst_case = data.get("worst_case")
        if worst_case and worst_case != "none":
            interactions = data.get("interactions") or []
            if not isinstance(interactions, list):
                interactions = []
            has_matching_pair = any(
                isinstance(pair, dict) and pair.get("interaction_type") == worst_case
                for pair in interactions
            )
            if not has_matching_pair:
                problems.append(
                    f'WORST_CASE_MISMATCH: worst_case="{worst_case}" but no '
                    "interaction pair has that exact interaction_type"
                )

        if irritation_flags_step is None:
            problems.append(_NOT_CALLED_IRRITATION_FLAGS)

        agent_interaction_synergies = (
            list(data["interaction_synergies"])
            if isinstance(data.get("interaction_synergies"), list)
            else []
        )
        agent_irritation_synergies = (
            list(data["irritation_synergies"])
            if isinstance(data.get("irritation_synergies"), list)
            else []
        )

        interactions = (
            data["interactions"]
            if isinstance(data.get("interactions"), list)
            else []
        )
        data["interaction_synergies"] = [
            pair
            for pair in interactions
            if isinstance(pair, dict)
            and pair.get("interaction_type") == "synergistic"
            and pair.get("is_irritation_related") is True
            and _pair_name(pair, "ingredient_a") not in irritant_set
            and _pair_name(pair, "ingredient_b") not in irritant_set
        ]
        data["irritation_synergies"] = [
            pair
            for pair in interactions
            if isinstance(pair, dict)
            and pair.get("interaction_type") == "synergistic"
            and pair.get("is_irritation_related") is True
            and (
                (_pair_name(pair, "ingredient_a") in irritant_set)
                != (_pair_name(pair, "ingredient_b") in irritant_set)
            )
        ]

        if problems:
            fields["debug_error"] = problems
            raise ValueError(
                "Interaction Builder Agent validation failed:\n"
                + "\n".join(problems)
            )

        fields["debug_output"] = {
            "interaction_result": to_jsonable(parsed),
            "agent_interaction_synergies": to_jsonable(
                agent_interaction_synergies
            ),
            "agent_irritation_synergies": to_jsonable(
                agent_irritation_synergies
            ),
            "rebuilt_interaction_synergies": to_jsonable(
                data.get("interaction_synergies")
            ),
            "rebuilt_irritation_synergies": to_jsonable(
                data.get("irritation_synergies")
            ),
        }
        return parsed


def _entries(tool_log, name: str) -> list:
    return [
        entry
        for entry in tool_log or []
        if isinstance(entry, dict) and entry.get("name") == name
    ]


def _last_ok(entries, name: str) -> None:
    if entries[-1].get("ok") is not True:
        raise ValueError(f"{name} returned an error")


def _pair_key(left, right):
    if not left or not right:
        return None
    return frozenset((left, right))


def _output_pair_key(pair):
    if not isinstance(pair, dict):
        return None
    return _pair_key(pair.get("ingredient_a"), pair.get("ingredient_b"))


def _name_set(raw) -> set:
    parsed = _parse_obs({"result": raw})
    if parsed is None:
        return set()
    if isinstance(parsed, list):
        items = _flat_one(parsed)
    else:
        items = [parsed]
    names = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            names.add(item)
        elif isinstance(item, dict):
            name = item.get("ingredient_name")
            if not name:
                name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name)
    return names


def _predefined_pairs(raw) -> list:
    parsed = _parse_obs({"result": raw})
    if not isinstance(parsed, list):
        return []
    pairs = []
    for item in _flat_one(parsed):
        if not isinstance(item, dict):
            continue
        key = _pair_key(item.get("ingredient_a"), item.get("ingredient_b"))
        if key is None:
            key = _pair_key(
                item.get("ingredient_name_a"), item.get("ingredient_name_b")
            )
        if key is not None:
            pairs.append((key, item))
    return pairs


def validate_interaction_builder_checks(parsed: dict, result) -> None:
    """D.8.1–7. Existing JSON/array checks stay in validate_interaction_result."""
    tool_log = getattr(result, "tool_log", None) or []
    max_calls_hit = bool(getattr(result, "max_calls_hit", False))

    active_entries = _entries(
        tool_log, "get_active_ingredients_for_interactions"
    )
    predefined_entries = _entries(tool_log, "get_predefined_interactions")
    if not active_entries:
        raise ValueError("get_active_ingredients_for_interactions not called")
    if not predefined_entries:
        raise ValueError("get_predefined_interactions not called")

    for name in _GET_TOOLS:
        entries = _entries(tool_log, name)
        if entries:
            _last_ok(entries, name)

    save_entries = _entries(tool_log, "save_llm_interaction")
    if save_entries:
        _last_ok(save_entries, "save_llm_interaction")

    if max_calls_hit:
        raise ValueError("hit 50 turns")

    output_pairs = []
    for pair in parsed.get("interactions") or []:
        key = _output_pair_key(pair)
        if key is not None:
            output_pairs.append(key)
    predefined = _predefined_pairs(predefined_entries[-1].get("result"))
    output_set = set(output_pairs)
    for key, item in predefined:
        if key not in output_set:
            names = sorted(key)
            raise ValueError(
                f"predefined pair {names[0]} + {names[-1]} missing"
            )

    saved = []
    for entry in save_entries:
        args = entry.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        key = _pair_key(args.get("ingredient_a"), args.get("ingredient_b"))
        if key is not None:
            saved.append(key)
    saved_set = set(saved)
    for pair in parsed.get("interactions") or []:
        if not isinstance(pair, dict) or pair.get("source") != "llm":
            continue
        key = _output_pair_key(pair)
        if key is None or key not in saved_set:
            a = pair.get("ingredient_a")
            b = pair.get("ingredient_b")
            raise ValueError(f'source "llm" pair {a} + {b} was not saved')

    active_names = _name_set(active_entries[-1].get("result"))
    for field in _OUTPUT_ARRAYS:
        rows = parsed.get(field) or []
        if not isinstance(rows, list):
            continue
        for pair in rows:
            if not isinstance(pair, dict):
                continue
            for key in ("ingredient_a", "ingredient_b"):
                name = pair.get(key)
                if name not in active_names:
                    raise ValueError(
                        f"{name!r} in {field} is not an active ingredient"
                    )
