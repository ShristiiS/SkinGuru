from __future__ import annotations

import json

from config import OPENAI_TIMEOUT_SECONDS
from precompute.flow1.claims_check import product_names_from_context
from precompute.llm_call import run_llm_call
from tracing import trace_step
from tracing.step import record_warnings

NAME_REVIEWER_MODEL = "gpt-4o-mini"
NAME_REVIEWER_SCHEMA_NAME = "name_reviewer"
NAME_REVIEWER_MAX_TOKENS = 16384
NAME_REVIEWER_FAILED_PREFIX = "name reviewer failed: "

NAME_REVIEWER_SYSTEM = """You are an ingredient-name reviewer for skincare analysis texts.

You receive:
1. PRODUCT INGREDIENT LIST — the exact, verified ingredient names of this product. This list is the only truth about what the product contains.
2. TEXTS — reasoning texts written by another AI about this product, each with a field name.

Your job is ONLY to fix ingredient names in the TEXTS. Do the following for every ingredient mentioned:

A. If it refers to an ingredient on the PRODUCT INGREDIENT LIST but is written differently (shortened, misspelled, common name, different word order, missing words), replace it with the exact name from the list.
   Example: list has "GLYCERYL STEARATE CITRATE", text says "glyceryl stearate" → write "Glyceryl Stearate Citrate".

B. If it is NOT on the PRODUCT INGREDIENT LIST and does not refer to any ingredient on it, remove it. Then rephrase only that sentence so the text still reads naturally and keeps its meaning.
   Example: list has no retinol, text says "Niacinamide and retinol improve tone" → "Niacinamide improves tone".

C. If it is already written exactly as on the list, leave it unchanged.

Do NOT change anything else:
- Do not change any number, percentage, score, tier, concentration status (Adequate, Borderline, Insufficient, etc.), true/false flag, product type, or concern name.
- Do not add new information, claims, or ingredients.
- Do not remove or reword anything that is not an ingredient name, except the minimum rephrasing needed in rule B.
- Words for ingredient groups or properties are NOT ingredient names — keep them as they are (for example: "fragrance-free", "alcohol-free", "preservatives", "humectants", "emollients", "actives").
- Keep the same paragraphs, order, and length as far as possible.

If a text needs no change, return it exactly as given.
Return every field, in the same order, with its "changes" list ([] if nothing changed)."""

CONCERN_TEXT_FIELDS = (
    "concern_reasoning",
    "bonus_reasoning",
    "synergy_reasoning",
    "full_explanation",
)
FORMULATION_TEXT_FIELDS = (
    "formulation_reasoning",
    "structural_completeness_reasoning",
    "preservation_design_reasoning",
    "stability_engineering_reasoning",
    "filler_ratio_reasoning",
    "full_explanation",
)

NAME_REVIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "text": {"type": "string"},
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["from", "to", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["field", "text", "changes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fields"],
    "additionalProperties": False,
}


def _render_user(product_names, fields: list[tuple[str, str]]) -> str:
    lines = ["PRODUCT INGREDIENT LIST"]
    if product_names:
        for name in product_names:
            if name:
                lines.append(f"- {name}")
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append("TEXTS")
    for field, text in fields:
        lines.append(f"field: {field}")
        lines.append(text if text is not None else "")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _check_review(expected_fields: list[str]):
    def check(text: str):
        try:
            parsed = json.loads(text)
        except Exception:
            return ["name reviewer reply is not valid JSON"]
        items = parsed.get("fields") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            return ["name reviewer reply missing fields"]
        have = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("field"), str):
                have.append(item["field"])
        missing = [name for name in expected_fields if name not in have]
        if missing:
            return [f"name reviewer missing fields: {missing}"]
        return []

    return check


def _log_failure(exc: BaseException) -> None:
    message = f"{NAME_REVIEWER_FAILED_PREFIX}{exc}"
    with trace_step("name_reviewer") as fields:
        fields["debug_output"] = message
        record_warnings([message])


def review_texts(fields: list[tuple[str, str]], product_names=None) -> dict[str, str]:
    originals = {name: text for name, text in fields}
    if not fields:
        return originals
    names = _product_names(product_names)
    expected = [name for name, _ in fields]
    try:
        result = run_llm_call(
            api="chat",
            model=NAME_REVIEWER_MODEL,
            timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": NAME_REVIEWER_SYSTEM},
                {"role": "user", "content": _render_user(names, fields)},
            ],
            max_tokens=NAME_REVIEWER_MAX_TOKENS,
            schema_name=NAME_REVIEWER_SCHEMA_NAME,
            schema=NAME_REVIEWER_SCHEMA,
            check=_check_review(expected),
            step="name_reviewer",
        )
        parsed = json.loads(result.text)
        applied = dict(originals)
        changes_by_field = {name: [] for name in expected}
        for item in parsed.get("fields") or []:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            if field not in applied or not isinstance(item.get("text"), str):
                continue
            applied[field] = item["text"]
            changes = item.get("changes")
            changes_by_field[field] = changes if isinstance(changes, list) else []
        with trace_step("name_reviewer") as span:
            span["debug_output"] = changes_by_field
        return applied
    except Exception as exc:
        _log_failure(exc)
        return originals


def review_concern_store_data(store_data, product_names=None):
    if not isinstance(store_data, str):
        return store_data
    try:
        rows = json.loads(store_data)
    except Exception:
        return store_data
    if not isinstance(rows, list):
        return store_data
    fields = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("concern_key")
        for field in CONCERN_TEXT_FIELDS:
            text = row.get(field)
            if isinstance(text, str):
                fields.append((f"{key}.{field}", text))
    if not fields:
        return store_data
    reviewed = review_texts(fields, product_names)
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("concern_key")
        for field in CONCERN_TEXT_FIELDS:
            mapped = f"{key}.{field}"
            if mapped in reviewed and row.get(field) != reviewed[mapped]:
                row[field] = reviewed[mapped]
                changed = True
    if not changed:
        return store_data
    return json.dumps(rows, ensure_ascii=False)


def review_formulation_store_data(store_data, product_names=None):
    if not isinstance(store_data, str):
        return store_data
    try:
        parsed = json.loads(store_data)
    except Exception:
        return store_data
    if not isinstance(parsed, dict):
        return store_data
    fields = [
        (field, parsed[field] if isinstance(parsed.get(field), str) else "")
        for field in FORMULATION_TEXT_FIELDS
        if field in parsed
    ]
    if not fields:
        return store_data
    reviewed = review_texts(fields, product_names)
    changed = False
    for field in FORMULATION_TEXT_FIELDS:
        if field in parsed and reviewed.get(field) != parsed.get(field):
            parsed[field] = reviewed[field]
            changed = True
    if not changed:
        return store_data
    return json.dumps(parsed, ensure_ascii=False)


def _product_names(product_names=None):
    if product_names is not None:
        return list(product_names)
    return product_names_from_context()
