from __future__ import annotations

import re

from precompute.concentration.llm_checks import ALLOWED_PRODUCT_TYPES
from precompute.flow1.canonical_catalog import get_canonical_names
from precompute.flow1.product_record import current_product_record

IRRITATION_SYNERGY_EMPTY = "No ingredient-level irritation synergies found"
INTERACTION_SYNERGY_EMPTY = (
    "No harmful interaction pairs causing irritation were found"
)

_MIN_NAME_LEN = 5
_STATUSES = (
    "Adequate",
    "Borderline",
    "Insufficient",
    "No data",
    "Very Insufficient",
    "Trace",
)
_COUNT_FIELDS = (
    "preservative_count",
    "botanical_count",
    "filler_count",
    "filler_percent",
    "filler_pct",
    "filler_percentage",
)
_PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
_DECIMAL_RE = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.%])")
_TIER_RE = re.compile(r"Tier\s+(\d+)", re.I)
_FORM_SCORE_RE = re.compile(r"FORMULATION_SCORE\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_FORM_BAND_RE = re.compile(r"FORMULATION_BAND\s*:\s*(\S+)", re.I)
_WORD_SPLIT = re.compile(r"\s*(?:,|;| and | & )\s*", re.I)


def _empty_text(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _is_empty_list(value) -> bool:
    return not isinstance(value, list) or len(value) == 0


def _name_or_first_word_in(name: str, text_cf: str) -> bool:
    full = name.strip().casefold()
    if full and full in text_cf:
        return True
    parts = name.strip().split()
    if not parts:
        return False
    first = parts[0].casefold()
    return bool(first) and first in text_cf


def _whole_word(name: str) -> re.Pattern:
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
        re.I,
    )


def product_names_from_context(product_id=None) -> list[str]:
    record = current_product_record()
    if record is not None and record.ingredient_names:
        return list(record.ingredient_names)
    return []


def collect_numbers(obj, out=None) -> list[float]:
    if out is None:
        out = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return out
    if isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, out)
        return out
    if isinstance(obj, (list, tuple)):
        for value in obj:
            collect_numbers(value, out)
    return out


def _last_digit_place(shown: str) -> float:
    if "." in shown:
        decimals = len(shown.split(".", 1)[1])
        return 10 ** (-decimals)
    return 1.0


def _number_in_data(claimed: float, shown: str, data_numbers: list[float]) -> bool:
    tolerance = max(0.01, _last_digit_place(shown))
    return any(abs(claimed - value) <= tolerance for value in data_numbers)


def _text_numbers(text: str) -> list[tuple[str, float]]:
    found = []
    seen_spans = []
    for match in _PERCENT_RE.finditer(text):
        found.append((match.group(0).strip(), float(match.group(1))))
        seen_spans.append(match.span())
    for match in _DECIMAL_RE.finditer(text):
        start, end = match.span()
        overlap = any(
            not (end <= left or start >= right) for left, right in seen_spans
        )
        if overlap:
            continue
        found.append((match.group(1), float(match.group(1))))
    return found


def check_numbers(field: str, text: str, data) -> list[str]:
    data_numbers = collect_numbers(data)
    problems = []
    for shown, claimed in _text_numbers(text):
        if not _number_in_data(claimed, shown.replace("%", "").strip(), data_numbers):
            problems.append(f"{field} says {shown}, which is not in the data.")
    return problems


def check_common(field: str, text: str, product_names, data) -> list[str]:
    return check_numbers(field, text, data)


def _flag_names(calc_result) -> list[str]:
    if not isinstance(calc_result, dict):
        return []
    names = []
    for key in calc_result:
        if isinstance(key, str) and (
            key.startswith("has_") or key.startswith("is_")
        ):
            names.append(key)
    return names


def _flag_list_field(flag: str, calc_result: dict) -> str | None:
    if flag.startswith("has_"):
        candidate = flag[4:] + "_ingredients"
    elif flag.startswith("is_"):
        candidate = flag[3:] + "_ingredients"
    else:
        return None
    if candidate in calc_result:
        return candidate
    return None


def _flag_actual(calc_result: dict, flag: str):
    return calc_result.get(flag)


def _claimed_bool(token: str) -> bool:
    return token.casefold() in ("true", "triggered")


def _actual_bool_label(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def check_flags_mentioned(text: str, calc_result) -> list[str]:
    problems = []
    for flag in _flag_names(calc_result):
        if not _whole_word(flag).search(text or ""):
            problems.append(f"{flag} is not mentioned in safety_reasoning.")
    return problems


def _flag_alt(flags: list[str]) -> str:
    return "|".join(re.escape(flag) for flag in sorted(flags, key=len, reverse=True))


def check_flag_values(field: str, text: str, calc_result) -> list[str]:
    flags = _flag_names(calc_result)
    if not flags or not isinstance(text, str) or not text.strip():
        return []
    alt = _flag_alt(flags)
    problems = []
    seen = set()
    value_re = re.compile(
        rf"(?i)\b({alt})\s+is\s+(true|false|triggered)\b"
    )
    for match in value_re.finditer(text):
        flag = next(
            name for name in flags if name.casefold() == match.group(1).casefold()
        )
        claimed = _claimed_bool(match.group(2))
        actual = _flag_actual(calc_result, flag)
        key = (flag, claimed)
        if key in seen:
            continue
        seen.add(key)
        if actual is not claimed:
            problems.append(
                f"{field} says {flag} is {_actual_bool_label(claimed)}, "
                f"but the calculation says {_actual_bool_label(actual)}."
            )
    list_re = re.compile(
        rf"(?i)((?:{alt})(?:\s*,\s*(?:{alt}))+(?:\s+(?:and|&)\s+(?:{alt}))?"
        rf"|(?:{alt})(?:\s*,\s*(?:{alt}))*\s+(?:and|&)\s+(?:{alt}))"
        rf"\s+are\s+all\s+(true|false)\b"
    )
    for match in list_re.finditer(text):
        claimed = _claimed_bool(match.group(2))
        chunk = match.group(1)
        for flag in flags:
            if not _whole_word(flag).search(chunk):
                continue
            actual = _flag_actual(calc_result, flag)
            key = (flag, claimed)
            if key in seen:
                continue
            seen.add(key)
            if actual is not claimed:
                problems.append(
                    f"{field} says {flag} is {_actual_bool_label(claimed)}, "
                    f"but the calculation says {_actual_bool_label(actual)}."
                )
    return problems


def _names_in_clause(clause: str, known_names) -> list[str]:
    found = []
    seen = set()
    for name in known_names:
        if not name or name.casefold() in seen:
            continue
        if _whole_word(name).search(clause):
            seen.add(name.casefold())
            found.append(name)
    for part in _WORD_SPLIT.split(clause):
        cleaned = part.strip().strip(".")
        if not cleaned or cleaned.casefold() in seen:
            continue
        found.append(cleaned)
        seen.add(cleaned.casefold())
    return found


def _list_has_name(items, name: str) -> bool:
    if not isinstance(items, list):
        return False
    want = name.casefold()
    for item in items:
        if isinstance(item, str) and item.casefold() == want:
            return True
        if isinstance(item, dict):
            for key in ("ingredient_name", "name", "canonical_name"):
                value = item.get(key)
                if isinstance(value, str) and value.casefold() == want:
                    return True
    return False


def check_flag_ingredients(field: str, text: str, calc_result) -> list[str]:
    flags = _flag_names(calc_result)
    if not flags or not isinstance(text, str):
        return []
    alt = _flag_alt(flags)
    known = []
    catalog = get_canonical_names() or frozenset()
    known.extend(name for name in catalog if len(name) >= _MIN_NAME_LEN)
    for flag in flags:
        list_field = _flag_list_field(flag, calc_result)
        rows = calc_result.get(list_field) if list_field else None
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, str):
                    known.append(item)
                elif isinstance(item, dict):
                    for key in ("ingredient_name", "name", "canonical_name"):
                        if item.get(key):
                            known.append(item[key])
    known.sort(key=len, reverse=True)
    problems = []
    clause_re = re.compile(
        rf"(?i)\b({alt})\s+is\s+(?:triggered\s+by|true\s+because)\s+([^.]+)"
    )
    for match in clause_re.finditer(text):
        flag = next(
            name for name in flags if name.casefold() == match.group(1).casefold()
        )
        list_field = _flag_list_field(flag, calc_result)
        if not list_field:
            continue
        rows = calc_result.get(list_field)
        for name in _names_in_clause(match.group(2), known):
            if not _list_has_name(rows, name):
                problems.append(
                    f"{field} names {name} for {flag}, but it is not in {list_field}."
                )
    return problems


def check_safety_texts(
    store_args: dict,
    calc_result,
    interaction_result,
    product_names,
) -> list[str]:
    pipeline = interaction_result if isinstance(interaction_result, dict) else {}
    problems = []
    if _is_empty_list(pipeline.get("irritation_synergies")):
        if (
            store_args.get("irritation_synergy_reasoning")
            != IRRITATION_SYNERGY_EMPTY
        ):
            problems.append(
                "irritation_synergy_reasoning must be "
                f"{IRRITATION_SYNERGY_EMPTY!r}"
            )
    if _is_empty_list(pipeline.get("irritation_interactions")):
        if (
            store_args.get("interaction_synergy_reasoning")
            != INTERACTION_SYNERGY_EMPTY
        ):
            problems.append(
                "interaction_synergy_reasoning must be "
                f"{INTERACTION_SYNERGY_EMPTY!r}"
            )
    return problems


def _normalize_tier(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    match = _TIER_RE.search(text)
    if match:
        return int(match.group(1))
    try:
        return int(text)
    except (TypeError, ValueError):
        return text.casefold()


def _status_label(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def check_tier_status(field: str, text: str, contributing) -> list[str]:
    if not isinstance(text, str) or not isinstance(contributing, list):
        return []
    problems = []
    rows = []
    for item in contributing:
        if isinstance(item, dict) and item.get("ingredient_name"):
            rows.append(item)
    rows.sort(
        key=lambda item: len(str(item.get("ingredient_name") or "")),
        reverse=True,
    )
    for item in rows:
        name = item["ingredient_name"]
        match = _whole_word(name).search(text)
        if not match:
            continue
        window = text[match.end() : match.end() + 120]
        tier_match = _TIER_RE.search(window)
        if tier_match:
            claimed = int(tier_match.group(1))
            actual = _normalize_tier(item.get("tier"))
            if actual is not None and claimed != actual:
                problems.append(
                    f"{field} says {name} is Tier {claimed}, data says "
                    f"Tier {actual}."
                )
        for status in _STATUSES:
            if _whole_word(status).search(window):
                actual = _status_label(item.get("concentration_status"))
                if actual and status.casefold() != actual.casefold():
                    problems.append(
                        f"{field} says {name} is {status}, data says {actual}."
                    )
                break
    return problems


def check_concern_texts(rows, calc_by_key, synergy, product_names, concern_data) -> list[str]:
    problems = []
    synergy_obj = synergy if isinstance(synergy, dict) else {}
    synergies = synergy_obj.get("synergies")
    synergy_by_key = {}
    if isinstance(synergies, list):
        for entry in synergies:
            if isinstance(entry, dict) and entry.get("concern") not in synergy_by_key:
                synergy_by_key[entry.get("concern")] = entry
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = row.get("concern_key")
        calc_row = calc_by_key.get(key) if isinstance(calc_by_key, dict) else None
        entry = synergy_by_key.get(key)
        contributing = []
        if isinstance(calc_row, dict):
            contributing = calc_row.get("concern_contributing_ingredients") or []
        elif isinstance(row.get("concern_contributing_ingredients"), list):
            contributing = row.get("concern_contributing_ingredients")
        data = {
            "calc": calc_row,
            "synergy": entry,
            "concern_data": concern_data,
            "row": {
                "concern_score": row.get("concern_score"),
                "bonus_score": row.get("bonus_score"),
                "synergy_score": row.get("synergy_score"),
            },
        }
        for field in (
            "concern_reasoning",
            "bonus_reasoning",
            "synergy_reasoning",
            "full_explanation",
        ):
            text = row.get(field)
            if _empty_text(text):
                continue
            labeled = f"concern {key!r} {field}"
            problems.extend(check_common(labeled, text, product_names, data))
            problems.extend(check_tier_status(labeled, text, contributing))
    return problems


def _find_fields(obj, names, found=None):
    if found is None:
        found = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and key not in found:
                found[key] = value
            _find_fields(value, names, found)
    elif isinstance(obj, list):
        for item in obj:
            _find_fields(item, names, found)
    return found


def check_formulation_texts(
    store: dict,
    calc_result,
    node22,
    product_names,
    extra_data=None,
) -> list[str]:
    calc = calc_result if isinstance(calc_result, dict) else {}
    pipeline = node22 if isinstance(node22, dict) else {}
    data = {"calc": calc, "node22": pipeline}
    if extra_data is not None:
        data["extra"] = extra_data
    problems = []
    fields = (
        "formulation_reasoning",
        "structural_completeness_reasoning",
        "preservation_design_reasoning",
        "stability_engineering_reasoning",
        "filler_ratio_reasoning",
        "full_explanation",
    )
    combined = []
    for field in fields:
        text = store.get(field) if isinstance(store, dict) else None
        if _empty_text(text):
            continue
        problems.extend(check_common(field, text, product_names, data))
        combined.append(text)
    blob = "\n".join(combined)
    expected_type = pipeline.get("formulation_product_type")
    for product_type in ALLOWED_PRODUCT_TYPES:
        if _whole_word(product_type).search(blob) and product_type != expected_type:
            problems.append(
                f"formulation_reasoning says {product_type}, data says "
                f"{expected_type}."
            )
            break
    score_match = _FORM_SCORE_RE.search(blob)
    if score_match and "final_score" in calc:
        claimed = float(score_match.group(1))
        actual = calc.get("final_score")
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            actual_f = None
        if actual_f is not None and abs(claimed - actual_f) > 0.01:
            problems.append(
                f"formulation_reasoning says FORMULATION_SCORE {score_match.group(1)}, "
                f"data says {actual}."
            )
    band_match = _FORM_BAND_RE.search(blob)
    if band_match and "score_band" in calc:
        claimed = band_match.group(1).strip()
        actual = calc.get("score_band")
        if actual is not None and str(claimed).casefold() != str(actual).casefold():
            problems.append(
                f"formulation_reasoning says FORMULATION_BAND {claimed}, "
                f"data says {actual}."
            )
    present = _find_fields(calc, _COUNT_FIELDS)
    for key, actual in present.items():
        if actual is None or isinstance(actual, bool):
            continue
        if isinstance(actual, (int, float)):
            pattern = re.compile(
                rf"(?i)\b{re.escape(key)}\b[^0-9]{{0,20}}(\d+(?:\.\d+)?)"
            )
            match = pattern.search(blob)
            if match:
                claimed = float(match.group(1))
                if abs(claimed - float(actual)) > 0.01:
                    problems.append(
                        f"formulation_reasoning says {key} {match.group(1)}, "
                        f"data says {actual}."
                    )
        if key in ("filler_percent", "filler_pct", "filler_percentage"):
            match = re.search(
                r"(?i)\bfiller\b[^%]{0,20}(\d+(?:\.\d+)?)\s*%", blob
            )
            if match and isinstance(actual, (int, float)):
                claimed = float(match.group(1))
                if abs(claimed - float(actual)) > 0.01:
                    problems.append(
                        f"formulation_reasoning says {match.group(0)}, "
                        f"data says {actual}."
                    )
    return problems


def check_pair_reason(
    field: str,
    text: str,
    pair_names,
    product_names,
    *,
    require_in_product=True,
    membership_label="this product",
) -> list[str]:
    names = [name for name in pair_names or [] if isinstance(name, str) and name.strip()]
    if len(names) != 2 or _empty_text(text):
        return []
    problems = []
    text_cf = text.casefold()
    if not all(_name_or_first_word_in(name, text_cf) for name in names):
        problems.append(f"{field} does not name both pair ingredients.")
    if require_in_product and product_names is not None:
        allow = {str(name).casefold() for name in product_names if name}
        if allow:
            for name in names:
                if name.casefold() not in allow:
                    problems.append(
                        f"{field} names {name}, which is not in {membership_label}."
                    )
    return problems


def check_synergy_entry(entry, product_names, data) -> list[str]:
    if not isinstance(entry, dict):
        return []
    concern = entry.get("concern")
    field = f"concern {concern!r} reason"
    text = entry.get("reason")
    if _empty_text(text):
        return []
    problems = []
    problems.extend(check_common(field, text, product_names, data))
    pairs = entry.get("pairs")
    pair_names = []
    if isinstance(pairs, list) and len(pairs) == 2:
        pair_names = [name for name in pairs if isinstance(name, str)]
    if len(pair_names) == 2:
        problems.extend(
            check_pair_reason(field, text, pair_names, product_names)
        )
    return problems


def check_llm_interaction_reason(
    reason,
    ingredient_a,
    ingredient_b,
    active_names,
    product_names=None,
    data=None,
) -> list[str]:
    field = "reason"
    problems = []
    text = reason if isinstance(reason, str) else ""
    names = [ingredient_a, ingredient_b]
    allow_product = product_names if product_names is not None else active_names
    problems.extend(check_common(field, text, allow_product, data or names))
    problems.extend(
        check_pair_reason(
            field,
            text,
            names,
            active_names,
            membership_label="the active ingredients",
        )
    )
    return problems
