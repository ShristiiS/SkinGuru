from __future__ import annotations

import json

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.helpers import compact_json
from precompute.flow1.claims_check import product_names_from_context
from precompute.llm_call import run_llm_call
from tracing import trace_step

FORMULATION_TEXT_REVIEWER_MODEL = "gpt-4.1-mini"
FORMULATION_TEXT_REVIEWER_SCHEMA_NAME = "formulation_text_reviewer"
FORMULATION_TEXT_REVIEWER_MAX_TOKENS = 16384

TEXT_FIELDS = (
    "formulation_reasoning",
    "structural_completeness_reasoning",
    "preservation_design_reasoning",
    "stability_engineering_reasoning",
    "filler_ratio_reasoning",
    "full_explanation",
)
_COMPONENT_NAMES = (
    "structural_completeness",
    "preservation_design",
    "stability_engineering",
    "filler_ratio",
)
_EXACT_FLAG_KEYS = ("water_in_top3", "is_water_based")
_COUNT_KEYS_ENDING_WITH_INGREDIENTS = ("total_ingredients",)

FORMULATION_TEXT_REVIEWER_SYSTEM = """You are the Formulation Text Reviewer for SkinGuru, a skincare platform. Shoppers read these texts.

You receive:
- PRODUCT TYPE: the product's formulation type (final and correct)
- COMPONENT DATA (final and correct): for each of the 4 components \u2014 structural_completeness, preservation_design, stability_engineering, filler_ratio \u2014 the ingredient lists per role (for example humectant_ingredients, preservative_ingredients, chelating_ingredients, buffering_ingredients, antioxidant_ingredients, viscosity_ingredients, filler_ingredients) and yes/no flags (for example has_chelating, has_buffering, water_in_top3). An empty list means no ingredient in the product has that role.
- BOTANICAL NAMES: the botanical extracts in the product
- PRODUCT INGREDIENT LIST: all ingredients in the product
- six TEXTS written by another agent:
  - formulation_reasoning (product type, short summary of all four components, overall verdict)
  - structural_completeness_reasoning
  - preservation_design_reasoning
  - stability_engineering_reasoning
  - filler_ratio_reasoning
  - full_explanation (complete paragraph summarising everything)

Your job: check EVERY SENTENCE of ALL SIX TEXTS against every rule below. Unless a rule says otherwise, every rule applies to every sentence of all six texts.
- If a sentence breaks no rule: keep it exactly as written.
- If a sentence breaks a rule: rewrite only that sentence (or remove it, or add a missing sentence, as the rule says).
Judge the MEANING of each sentence, not only the words. A sentence is wrong if what it means conflicts with the data, even if it uses no forbidden word.

=====================================
CHECK STEP BY STEP
=====================================
Step 1 \u2014 Read the PRODUCT TYPE, every list and flag in the COMPONENT DATA, and the BOTANICAL NAMES.
Step 2 \u2014 For each of the six texts, go sentence by sentence and apply every rule R1\u2013R8.
Step 3 \u2014 Record every change in "changes".

=====================================
RULES
=====================================
R1. ROLE NAMES MATCH THE LISTS
Applies to: every sentence of all six texts that names an ingredient with a role.
Check: is that ingredient in that role's list in the COMPONENT DATA? (Botanical extracts: in BOTANICAL NAMES.) Does the sentence give an ingredient a role it does not have in the lists (for example calling an ingredient a preservative or preservative booster when it is not in preservative_ingredients)?
Keep: correct ingredient\u2013role pairs.
Change: remove the wrong role from the ingredient, or move the ingredient to its correct role from the lists. Never invent a role.

R2. EVERY LISTED INGREDIENT NAMED
Applies to: structural_completeness_reasoning, preservation_design_reasoning, stability_engineering_reasoning, filler_ratio_reasoning.
Check: is every ingredient in every non-empty list of that component named in that component's text, with its role?
Keep: sentences that already name them correctly.
Change: add the missing ingredients to the sentence about that role, or add a sentence for the role.

R3. MISSING ITEMS MATCH THE DATA
Applies to: every sentence of all six texts that says the formula has no / lacks / is missing something.
Check: is that role's list really empty (or its flag false) in the COMPONENT DATA?
Keep: correct "missing" statements (for example "no chelating agent" when chelating_ingredients is empty).
Change: if the list is not empty, rewrite the sentence to say the formula has it, naming the ingredients from the list.

R4. PRODUCT TYPE
Applies to: every sentence of all six texts that states the product type or why the product is that type.
Check: does it match the PRODUCT TYPE? Does it claim which ingredients caused the classification?
Keep: the product type stated as given.
Change: correct a wrong product type; remove any invented reason for the classification.

R5. NO SCORE AND NO NUMBERS
Applies to: every sentence of all six texts.
Check: does the sentence contain any number \u2014 final score, score band written as a data label, component score, count, percentage or ratio?
Keep: the rest of the sentence. Plain words like "low", "high", "few", "several" are allowed.
Change: remove the number and rewrite in plain words.

R6. NO SYSTEM WORDS
Applies to: every sentence of all six texts.
Check: does the sentence mention the system or the data structure \u2014 "the model", "flag", "flagged", "calculation", "calculated", "score", "score breakdown", "function category", "data says", "in this analysis", or any key name such as has_viscosity, filler_count, humectant_ingredients, water_in_top3?
Keep: the meaning that is about the product.
Change: rewrite as a plain statement for a shopper.

R7. NO CONFLICT WITH THE DATA
Applies to: every sentence of all six texts.
Check: does the meaning of the sentence conflict with anything in the COMPONENT DATA, the PRODUCT TYPE or the BOTANICAL NAMES? Does it name an ingredient that is not in the PRODUCT INGREDIENT LIST?
Keep: sentences that agree with the data. Extra detail is allowed only if it supports the data (for example explaining what a humectant does).
Change: rewrite any sentence whose meaning conflicts with the data; remove ingredients not in the product.

R8. RIGHT TEXT
Applies to: structural_completeness_reasoning, preservation_design_reasoning, stability_engineering_reasoning, filler_ratio_reasoning.
Check: does each text talk about its own component? (structural content in structural_completeness_reasoning, preservation in preservation_design_reasoning, stability in stability_engineering_reasoning, filler in filler_ratio_reasoning.)
Keep: content in its own text.
Change: remove content that belongs to another component's text, and make sure that content exists in its own text.

=====================================
WHAT YOU MUST NOT DO
=====================================
- Do not change a sentence that breaks no rule.
- Do not add any number.
- Do not invent ingredients, roles or effects that are not in the data.
- Do not change the meaning of a correct sentence.
- Do not leave any text empty.

=====================================
OUTPUT
=====================================
Return JSON only, matching the schema. Return all six fields.
- "text": the final text (corrected, or unchanged if nothing was wrong).
- "changes": one entry per changed, removed, or added sentence: "from" (original sentence, or "" if added), "to" (new sentence, or "" if removed), "rule" (R1\u2013R8), "reason" (short). If nothing changed, "changes" is [].
"""

FORMULATION_TEXT_REVIEWER_USER_TEMPLATE = """PRODUCT TYPE:
{product_type}

COMPONENT DATA:
{components_json}

BOTANICAL NAMES:
{botanical_names_json}

PRODUCT INGREDIENT LIST:
{product_ingredients_json}

TEXTS:
{texts_json}

Check every sentence of all six texts against the rules and return the final texts."""

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
FORMULATION_TEXT_REVIEWER_SCHEMA = {
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


def _keep_component_key(key) -> bool:
    return isinstance(key, str) and (
        (
            key.endswith("_ingredients")
            and key not in _COUNT_KEYS_ENDING_WITH_INGREDIENTS
        )
        or key.startswith("has_")
        or key in _EXACT_FLAG_KEYS
    )


def slim_components(calc_result) -> dict:
    source = calc_result if isinstance(calc_result, dict) else {}
    slimmed = {}
    for name in _COMPONENT_NAMES:
        component = source.get(name)
        if not isinstance(component, dict):
            slimmed[name] = {}
            continue
        slimmed[name] = {
            key: value
            for key, value in component.items()
            if _keep_component_key(key)
        }
    return slimmed


def botanical_names(count_result):
    if isinstance(count_result, list):
        return count_result
    if isinstance(count_result, dict):
        named = count_result.get("botanicals")
        if isinstance(named, list):
            return named
        for value in count_result.values():
            if isinstance(value, list):
                return value
    return []


def build_texts(store_args) -> dict:
    source = store_args if isinstance(store_args, dict) else {}
    texts = {}
    for field in TEXT_FIELDS:
        value = source.get(field)
        texts[field] = value if isinstance(value, str) else ""
    return texts


def _product_type(node22) -> str:
    if not isinstance(node22, dict):
        return ""
    value = node22.get("formulation_product_type")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def render_reviewer_user(product_type, components, names, ingredients, texts) -> str:
    return (
        FORMULATION_TEXT_REVIEWER_USER_TEMPLATE.replace(
            "{product_type}", product_type
        )
        .replace("{components_json}", compact_json(components))
        .replace("{botanical_names_json}", compact_json(names))
        .replace("{product_ingredients_json}", compact_json(ingredients))
        .replace("{texts_json}", compact_json(texts))
    )


def _emit(line: str, warning: bool = False) -> None:
    with trace_step("formulation_text_review") as fields:
        fields["debug_output"] = line
        if warning:
            fields["warnings"] = [line]


def _trace_change(field, change: dict) -> str:
    rule = change.get("rule") if isinstance(change.get("rule"), str) else ""
    frm = change.get("from") if isinstance(change.get("from"), str) else ""
    to = change.get("to") if isinstance(change.get("to"), str) else ""
    reason = change.get("reason") if isinstance(change.get("reason"), str) else ""
    return (
        f"formulation_text_review: {field} rule={rule} "
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
                "formulation_text_review: "
                + field
                + " missing from reviewer output \u2014 original text kept"
            )
            continue
        text = item.get("text")
        if _empty_text(text):
            warnings.append(
                "formulation_text_review: "
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


def review_formulation_texts(store_args, node22, calc_result, count_result) -> dict:
    originals = build_texts(store_args)
    names = product_names_from_context()
    if not isinstance(names, list):
        names = []
    user = render_reviewer_user(
        _product_type(node22),
        slim_components(calc_result),
        botanical_names(count_result),
        names,
        originals,
    )
    try:
        result = run_llm_call(
            api="chat",
            model=FORMULATION_TEXT_REVIEWER_MODEL,
            timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": FORMULATION_TEXT_REVIEWER_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=FORMULATION_TEXT_REVIEWER_MAX_TOKENS,
            schema_name=FORMULATION_TEXT_REVIEWER_SCHEMA_NAME,
            schema=FORMULATION_TEXT_REVIEWER_SCHEMA,
            step="formulation_text_reviewer",
        )
        parsed = json.loads(result.text)
        if not isinstance(parsed, dict):
            raise ValueError("reviewer reply is not an object")
        applied, change_lines, warnings, field_count = _apply_review(
            originals, parsed
        )
    except Exception as exc:
        _emit(
            "formulation_text_review: FAILED ("
            + str(exc)
            + ") \u2014 original texts kept",
            warning=True,
        )
        return originals
    for line in change_lines:
        _emit(line)
    for line in warnings:
        _emit(line, warning=True)
    _emit(
        "formulation_text_review: "
        + str(len(change_lines))
        + " changes across "
        + str(field_count)
        + " fields"
    )
    return applied
