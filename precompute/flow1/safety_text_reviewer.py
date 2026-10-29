from __future__ import annotations

import json

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.helpers import compact_json
from precompute.llm_call import run_llm_call
from tracing import trace_step

SAFETY_TEXT_REVIEWER_MODEL = "gpt-4.1-mini"
SAFETY_TEXT_REVIEWER_SCHEMA_NAME = "safety_text_reviewer"
SAFETY_TEXT_REVIEWER_MAX_TOKENS = 8192

TEXT_FIELDS = (
    "irritation_synergy_reasoning",
    "interaction_synergy_reasoning",
)
_PAIR_LIST_KEYS = (
    "irritation_synergies",
    "interaction_synergies",
    "irritation_interactions",
)
_PAIR_KEYS = (
    "ingredient_a",
    "ingredient_b",
    "severity",
    "reason",
)

SAFETY_TEXT_REVIEWER_SYSTEM = """You are the Safety Synergy Text Reviewer for SkinGuru, a skincare platform. Shoppers read these texts.

You receive:
- the PAIR DATA from the Interaction Builder (final and correct):
  - irritation_synergies: pairs that reduce ingredient-level irritation
  - interaction_synergies: pairs that counter irritation from harmful interaction pairs
  - irritation_interactions: the harmful interaction pairs
  Each pair has: ingredient_a, ingredient_b, severity, reason.
- two TEXTS written by another agent:
  - irritation_synergy_reasoning (explains irritation_synergies)
  - interaction_synergy_reasoning (explains interaction_synergies)

Your job: check EVERY SENTENCE of BOTH TEXTS against every rule below. Unless a rule says otherwise, every rule applies to every sentence of both texts.
- If a sentence breaks no rule: keep it exactly as written.
- If a sentence breaks a rule: rewrite only that sentence (or remove it, or add a missing sentence, as the rule says).
Judge the MEANING of each sentence, not only the words. A sentence is wrong if what it means conflicts with the pair data, even if it uses no forbidden word.

=====================================
EXACT SENTENCES
=====================================
- irritation_synergy_reasoning:
  - if irritation_synergies is empty, the whole text must be exactly: "No ingredient-level irritation synergies found"
- interaction_synergy_reasoning:
  - if irritation_interactions is empty, the whole text must be exactly: "No harmful interaction pairs causing irritation were found"
  - if irritation_interactions is not empty but interaction_synergies is empty, the whole text must be exactly: "No synergies were found to counter the harmful interaction pairs"
If an exact sentence applies, replace the whole text with it and skip rules R1–R6 for that text.
If an exact sentence does NOT apply, the text must not contain any of the three exact sentences or say that nothing was found.

=====================================
STRENGTH WORDS
=====================================
From each pair's severity: high = "strong", medium = "moderate", low = "mild".

=====================================
CHECK STEP BY STEP
=====================================
Step 1 — Read the PAIR DATA: every pair in irritation_synergies, interaction_synergies and irritation_interactions.
Step 2 — Apply the EXACT SENTENCES section to each text.
Step 3 — For each text where no exact sentence applies, go sentence by sentence and apply every rule R1–R6.
Step 4 — Record every change in "changes".

=====================================
RULES
=====================================
R1. EVERY PAIR COVERED
Applies to: irritation_synergy_reasoning (pairs in irritation_synergies) and interaction_synergy_reasoning (pairs in interaction_synergies).
Check: is every pair of the matching list covered, with both of its ingredients named?
Keep: sentences that already cover a pair correctly.
Change: add a sentence for every missing pair, written from that pair's reason and severity.

R2. SAME MEANING AS THE PAIR'S REASON
Applies to: every sentence of both texts that describes a pair.
Check: does it have the same meaning as that pair's reason — same mechanism, same effect? Words may differ. Is any detail of the reason dropped? Is anything added that conflicts with the reason?
Keep: sentences that match the reason in meaning. Extra detail is allowed only if it supports the reason.
Change: add dropped details, remove or rewrite anything that conflicts.

R3. STRENGTH MATCHES SEVERITY
Applies to: every sentence of both texts that states how strong a pair's effect is.
Check: does the strength word match the pair's severity (high = strong, medium = moderate, low = mild)?
Keep: correct strength words.
Change: replace a wrong strength word with the correct one.

R4. ONLY WHAT IS IN THE PAIR DATA
Applies to: every sentence of both texts.
Check: does the sentence mention a pair, ingredient, or mechanism that is not in the pair data? Does it put a pair in the wrong text (an irritation_synergies pair in interaction_synergy_reasoning, or the other way round)?
Keep: sentences about pairs in the correct list.
Change: remove anything not in the pair data. Move a pair's sentence to the correct text.

R5. NO NUMBERS AND NO SYSTEM WORDS
Applies to: every sentence of both texts.
Check: does the sentence contain any number (offset, deduction, score, percentage, severity value) or mention the system — "interaction_result", "calculated", "calculation", "flag", "score", "deduction", "offset applied", "status", "pair list", "data says" or similar?
Keep: the meaning that is about the product.
Change: remove the number or system word and rewrite as a plain statement for a shopper.

R6. INGREDIENT NAMES
Applies to: every sentence of both texts.
Check: is every ingredient named as it appears in the pair data?
Keep: correct names.
Change: correct misspelled or altered names to the name in the pair data.

=====================================
WHAT YOU MUST NOT DO
=====================================
- Do not change a sentence that breaks no rule.
- Do not add any number.
- Do not invent pairs, ingredients, mechanisms or effects that are not in the pair data.
- Do not change the meaning of a correct sentence.

=====================================
OUTPUT
=====================================
Return JSON only, matching the schema. Return both fields.
- "text": the final text (corrected, or unchanged if nothing was wrong).
- "changes": one entry per changed, removed, or added sentence: "from" (original sentence, or "" if added), "to" (new sentence, or "" if removed), "rule" ("EXACT" or R1–R6), "reason" (short). If nothing changed, "changes" is []."""

SAFETY_TEXT_REVIEWER_USER_TEMPLATE = """PAIR DATA:
{pair_data_json}

TEXTS:
{texts_json}

Check every sentence of both texts against the rules and return the final texts."""

_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "from": {"type": "string"},
        "to": {"type": "string"},
        "rule": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["from", "to", "rule", "reason"],
    "additionalProperties": False,
}
_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "string",
            "enum": list(TEXT_FIELDS),
        },
        "text": {"type": "string"},
        "changes": {
            "type": "array",
            "items": _CHANGE_SCHEMA,
        },
    },
    "required": ["field", "text", "changes"],
    "additionalProperties": False,
}
SAFETY_TEXT_REVIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": _FIELD_SCHEMA,
        }
    },
    "required": ["fields"],
    "additionalProperties": False,
}


def build_pair_data(interaction_result) -> dict:
    source = interaction_result if isinstance(interaction_result, dict) else {}
    data = {}
    for key in _PAIR_LIST_KEYS:
        value = source.get(key)
        if not isinstance(value, list):
            data[key] = []
            continue
        pairs = []
        for item in value:
            if not isinstance(item, dict):
                continue
            pairs.append({field: item.get(field) for field in _PAIR_KEYS})
        data[key] = pairs
    return data


def build_texts(store_args) -> dict:
    source = store_args if isinstance(store_args, dict) else {}
    texts = {}
    for field in TEXT_FIELDS:
        value = source.get(field)
        texts[field] = value if isinstance(value, str) else ""
    return texts


def render_reviewer_user(pair_data, texts) -> str:
    return SAFETY_TEXT_REVIEWER_USER_TEMPLATE.replace(
        "{pair_data_json}", compact_json(pair_data)
    ).replace("{texts_json}", compact_json(texts))


def _emit(line: str, warning: bool = False) -> None:
    with trace_step("safety_text_review") as fields:
        fields["debug_output"] = line
        if warning:
            fields["warnings"] = [line]


def _trace_change(field, change: dict) -> str:
    rule = change.get("rule") if isinstance(change.get("rule"), str) else ""
    frm = change.get("from") if isinstance(change.get("from"), str) else ""
    to = change.get("to") if isinstance(change.get("to"), str) else ""
    reason = change.get("reason") if isinstance(change.get("reason"), str) else ""
    return (
        f"safety_text_review: {field} rule={rule} "
        f'from="{frm}" to="{to}" reason="{reason}"'
    )


def _fields_by_name(parsed) -> dict:
    by_name = {}
    items = parsed.get("fields") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return by_name
    for item in items:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        if field in TEXT_FIELDS and field not in by_name:
            by_name[field] = item
    return by_name


def _empty_text(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _apply_review(originals, parsed):
    fields = _fields_by_name(parsed)
    applied = dict(originals)
    change_lines = []
    warnings = []
    changed_fields = set()
    for field in TEXT_FIELDS:
        item = fields.get(field)
        if item is None:
            warnings.append(
                "safety_text_review: "
                + field
                + " missing from reviewer output \u2014 original text kept"
            )
            continue
        text = item.get("text")
        if _empty_text(text):
            warnings.append(
                "safety_text_review: "
                + field
                + " empty reviewer text \u2014 original text kept"
            )
            continue
        applied[field] = text
        changes = item.get("changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            change_lines.append(_trace_change(field, change))
            changed_fields.add(field)
    return applied, change_lines, warnings, len(changed_fields)


def review_safety_synergy_texts(store_args, interaction_result) -> dict:
    originals = build_texts(store_args)
    user = render_reviewer_user(build_pair_data(interaction_result), originals)
    try:
        result = run_llm_call(
            api="chat",
            model=SAFETY_TEXT_REVIEWER_MODEL,
            timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": SAFETY_TEXT_REVIEWER_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=SAFETY_TEXT_REVIEWER_MAX_TOKENS,
            schema_name=SAFETY_TEXT_REVIEWER_SCHEMA_NAME,
            schema=SAFETY_TEXT_REVIEWER_SCHEMA,
            step="safety_text_reviewer",
        )
        parsed = json.loads(result.text)
        if not isinstance(parsed, dict):
            raise ValueError("reviewer reply is not an object")
        applied, change_lines, warnings, field_count = _apply_review(
            originals, parsed
        )
    except Exception as exc:
        _emit(
            f"safety_text_review: FAILED ({exc}) \u2014 original texts kept",
            warning=True,
        )
        return originals
    for line in change_lines:
        _emit(line)
    for line in warnings:
        _emit(line, warning=True)
    _emit(
        "safety_text_review: "
        + str(len(change_lines))
        + " changes across "
        + str(field_count)
        + " fields"
    )
    return applied
