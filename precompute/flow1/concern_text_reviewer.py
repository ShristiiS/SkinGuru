from __future__ import annotations

import json

from precompute.concentration.helpers import compact_json
from precompute.flow1.claims_check import product_names_from_context
from precompute.llm_call import run_llm_call
from tracing import trace_step

CONCERN_TEXT_REVIEWER_MODEL = "gpt-4.1-mini"
CONCERN_TEXT_REVIEWER_SCHEMA_NAME = "concern_text_reviewer"
CONCERN_TEXT_REVIEWER_MAX_TOKENS = 32768
CONCERN_TEXT_REVIEWER_TIMEOUT_SECONDS = 300
CONCERN_TEXT_REVIEWER_FAILED_PREFIX = "concern_text_review: FAILED ("

TEXT_FIELDS = (
    "concern_reasoning",
    "bonus_reasoning",
    "synergy_reasoning",
    "full_explanation",
)
_INGREDIENT_KEYS = (
    "ingredient_name",
    "tier",
    "concentration_status",
    "effective_value",
    "mechanism",
)
FORBIDDEN_INGREDIENT_KEYS = (
    "approximate_concentration",
    "required_concentration",
    "tier_value",
)

CONCERN_TEXT_REVIEWER_SYSTEM = """You are the Concern Text Reviewer for SkinGuru, a skincare platform. Shoppers read these texts.

You receive, for each of 15 concerns:
- the DATA for that concern (correct values, already checked by code)
- the SYNERGY ORIGINAL for that concern (what the Synergy agent said)
- the four TEXTS written by another agent: concern_reasoning, bonus_reasoning, synergy_reasoning, full_explanation
You also receive the PRODUCT INGREDIENT LIST.

Your job: check EVERY SENTENCE of ALL FOUR TEXTS (concern_reasoning, bonus_reasoning, synergy_reasoning, full_explanation) against every rule below. Unless a rule says otherwise, every rule applies to every sentence of all four texts.
- If a sentence breaks no rule: keep it exactly as written.
- If a sentence breaks a rule: rewrite only that sentence (or remove it, or add a missing sentence, as the rule says).
Judge the MEANING of each sentence, not only the words. A sentence is wrong if what it means conflicts with the data, even if it uses no forbidden word.

=====================================
HOW AN INGREDIENT MUST BE DESCRIBED (in all four texts)
=====================================
- Ingredient name + evidence in plain words + concentration status.
- Evidence in plain words (from tier):
  - tier 1 = "strong clinical evidence"
  - tier 2 = "good evidence"
  - tier 3 = "early/limited evidence"
- Concentration status: the status exactly as in the data: Adequate, Borderline, Insufficient, Very Insufficient, Trace.
- If the status is "No data", use one of:
  - "makes only a minimal contribution"
  - "plays only a minor supporting role"
  - "adds a very small benefit"
- The mechanism may be mentioned if it is in the data.

=====================================
CHECK EACH CONCERN STEP BY STEP
=====================================
Step 1 — Read the DATA: concern_score, bonus_score, every ingredient in concern_contributing_ingredients and bonus_contributing_ingredients with its tier, concentration_status, effective_value and mechanism.
Step 2 — Read the SYNERGY ORIGINAL: the pair, the strength, and the reason.
Step 3 — For each of the four texts, go sentence by sentence and apply every rule R1–R11.
Step 4 — Then apply R12 (empty texts).
Step 5 — Record every change in "changes".

=====================================
RULES
=====================================
R1. SCORE ZERO
Applies to: every sentence of concern_reasoning, bonus_reasoning, synergy_reasoning and full_explanation.
Check: if concern_score is 0, does any sentence claim or suggest that any ingredient in this product helps this concern in any way — fully, partly, slightly, cosmetically, indirectly, temporarily, or "may help"? Examples that are WRONG: "may help skin comfort", "may offer some smoothing", "contains brightening components", "some ingredients may support", "could give a mild benefit", "helpful in practice".
Keep: sentences that explain in general what this concern needs, and sentences that say this product does not address this concern.
Change: rewrite or remove every sentence that suggests any help. The only message about this product must be that it does not address this concern. No exceptions.

R2. NO PERCENTAGES
Applies to: every sentence of all four texts.
Check: does the sentence contain a product concentration or a required concentration — any number with %, or phrases like "versus X", "around X", "at about X", "X effective level", "range of X"?
Keep: the rest of the sentence.
Change: remove the number and the comparison. If the sentence was about concentration, replace it with the concentration status (or the "No data" wording).

R3. NO SCORE VALUES AND NO TIER LABELS
Applies to: every sentence of all four texts.
Check: does the sentence contain any score value (concern score, bonus score, synergy score, effective value, tier value, e.g. "a score of 0.38") or the words "Tier 1", "Tier 2", "Tier 3"?
Keep: the rest of the sentence.
Change: remove score values. Replace tier labels with the evidence words.

R4. INGREDIENT FACTS MATCH THE DATA
Applies to: every sentence of all four texts that mentions an ingredient.
Check: what the sentence says about that ingredient's evidence, concentration status, mechanism or contribution — does it match the data for that ingredient in this concern?
Keep: sentences that match, and extra detail that supports the data.
Change: rewrite any part that does not match (wrong evidence level, wrong status, wrong mechanism, wrong size of contribution).

R5. "NO DATA" WORDING
Applies to: every sentence of all four texts that mentions an ingredient whose status is "No data".
Check: is it described with one of the three agreed sentences? Does it use wording like "role unclear", "not clearly defined", "low percentage", "poorly quantified", "likely limited", "not well specified", "not quantified", or anything similar?
Keep: the ingredient name and evidence words.
Change: replace that wording with one of the three agreed sentences.

R6. EVERY INGREDIENT NAMED
Applies to: concern_reasoning and bonus_reasoning.
Check:
- concern_reasoning: is every ingredient in concern_contributing_ingredients named and described as above?
- bonus_reasoning: is every ingredient in bonus_contributing_ingredients named and described as above? Is any ingredient NOT in that list named as contributing to the bonus?
Keep: sentences already correct.
Change: add a sentence for each missing ingredient. Remove the claim for any ingredient named as contributing to the bonus that is not in bonus_contributing_ingredients.

R7. NO SYSTEM WORDS
Applies to: every sentence of all four texts.
Check: does the sentence mention the system or how scores are made — "calculation", "calculated", "pre-calculated", "scoring", "framework", "assigned", "credited", "concern key", "reference", "listed concentration", "data says", "in this analysis", "the system" or similar?
Keep: the meaning that is about the product.
Change: rewrite it as a plain statement for a shopper, with no mention of the system.

R8. RANKING
Applies to: every sentence of all four texts.
Check: is an ingredient called the main, strongest, most important, key, standout, or biggest contributor (or "drives" the result)? Does it have the highest effective_value for this concern?
Keep: correct rankings.
Change: if it is not the highest effective_value, rewrite it. If an ingredient has strong evidence but a low concentration status, the sentence must say both: strong evidence, but only a minimal contribution.

R9. SYNERGY
Applies to: synergy_reasoning, and every sentence about synergy in concern_reasoning, bonus_reasoning and full_explanation.
Check: does it contain every detail of the SYNERGY ORIGINAL with the same meaning — both ingredients, the mechanism of each, the combined effect, and the strength? Words may differ. Is anything dropped? Is anything added that conflicts? Strength: synergy score 0.166 = strong, 0.10 = moderate, 0.05 = weak, 0 = no synergy (never write the number).
Keep: sentences that match the SYNERGY ORIGINAL in meaning.
Change: add missing details, remove conflicting ones. If the SYNERGY ORIGINAL has no pair, every synergy sentence must say there is no synergy for this concern and must not name any pair.

R10. NO CONFLICT WITH THE DATA
Applies to: every sentence of all four texts.
Check: does the meaning of the sentence conflict with anything in the data — how strong the product is for this concern (score level), the ingredient lists, evidence, concentration status, mechanism, or synergy?
Keep: sentences that agree with the data, and extra detail that supports the data.
Change: rewrite any sentence whose meaning conflicts with the data.

R11. INGREDIENT NAMES
Applies to: every sentence of all four texts.
Check: is every ingredient that is described as being in this product present in the PRODUCT INGREDIENT LIST? Is the name spelled as in the list?
Keep: correct names. Sentences that say the product lacks a type of ingredient (for example "it has no retinoids") are allowed.
Change: correct misspelled names to the name in the list. Remove any claim that the product contains an ingredient not in the list.

R12. EMPTY TEXT
Applies to: any of the four texts that is empty ("") or only whitespace.
Write it from the data only, following all rules R1–R11:
- concern_reasoning:
  - if concern_contributing_ingredients has ingredients: name each one, described as above, and say whether support is strong or weak.
  - if concern_contributing_ingredients is empty: say this product does not address this concern. Do not name any ingredient.
- bonus_reasoning:
  - if bonus_contributing_ingredients has ingredients: name each one, described as above, and say whether the bonus benefit is strong or weak.
  - if bonus_contributing_ingredients is empty: say there is no bonus benefit for this concern. Do not name any ingredient.
- synergy_reasoning:
  - if the SYNERGY ORIGINAL has a pair: write every detail of the SYNERGY ORIGINAL.
  - if it has no pair: say there is no synergy for this concern.
- full_explanation: a complete paragraph of at least 100 words: what this concern is about in general, how well this product addresses it (if both ingredient lists are empty: it does not address it), the contributing ingredients described as above (only if there are any), synergy (only if there is a pair), and a final verdict.
Never invent an ingredient, mechanism or benefit that is not in the data.

=====================================
WHAT YOU MUST NOT DO
=====================================
- Do not change a sentence that breaks no rule.
- Do not add any number, percentage, score, or tier label.
- Do not invent facts, mechanisms, or ingredients that are not in the data or the SYNERGY ORIGINAL.
- Do not change the meaning of a correct sentence.
- full_explanation must stay at least 100 words.

=====================================
OUTPUT
=====================================
Return JSON only, matching the schema. For every one of the 15 concerns return all four fields.
- "text": the final text (corrected, or unchanged if nothing was wrong).
- "changes": one entry per changed, removed, or added sentence: "from" (original sentence, or "" if added), "to" (new sentence, or "" if removed), "rule" (R1–R12), "reason" (short). If nothing changed, "changes" is []."""

CONCERN_TEXT_REVIEWER_USER_TEMPLATE = """PRODUCT INGREDIENT LIST:
{product_ingredients_json}

CONCERNS (15), each with DATA, SYNERGY ORIGINAL and TEXTS:
{concerns_json}

Check every sentence of every text against the rules and return the final texts."""

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
CONCERN_TEXT_REVIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concern_key": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": _FIELD_SCHEMA,
                    },
                },
                "required": ["concern_key", "fields"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["concerns"],
    "additionalProperties": False,
}


def _slim_ingredient(item):
    if not isinstance(item, dict):
        return item
    return {key: item.get(key) for key in _INGREDIENT_KEYS}


def _slim_ingredients(value):
    if not isinstance(value, list):
        return []
    return [
        _slim_ingredient(item) if isinstance(item, dict) else item for item in value
    ]


def _synergy_original(synergy, key) -> dict:
    synergies = synergy.get("synergies") if isinstance(synergy, dict) else None
    entry = None
    if isinstance(synergies, list):
        for item in synergies:
            if isinstance(item, dict) and item.get("concern") == key:
                entry = item
                break
    if not isinstance(entry, dict):
        return {"pairs": [], "synergy_score": 0, "reason": ""}
    pairs = entry.get("pairs")
    if pairs is None:
        pairs = []
    reason = entry.get("reason")
    if not isinstance(reason, str):
        reason = "" if reason is None else str(reason)
    return {
        "pairs": pairs,
        "synergy_score": entry.get("score"),
        "reason": reason,
    }


def _row_texts(row: dict) -> dict:
    texts = {}
    for field in TEXT_FIELDS:
        value = row.get(field)
        texts[field] = value if isinstance(value, str) else ""
    return texts


def build_concerns_payload(rows, synergy) -> list:
    concerns = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = row.get("concern_key")
        concerns.append(
            {
                "concern_key": key,
                "data": {
                    "concern_score": row.get("concern_score"),
                    "bonus_score": row.get("bonus_score"),
                    "concern_contributing_ingredients": _slim_ingredients(
                        row.get("concern_contributing_ingredients")
                    ),
                    "bonus_contributing_ingredients": _slim_ingredients(
                        row.get("bonus_contributing_ingredients")
                    ),
                },
                "synergy_original": _synergy_original(synergy, key),
                "texts": _row_texts(row),
            }
        )
    return concerns


def render_reviewer_user(product_names, concerns) -> str:
    names = list(product_names or [])
    return CONCERN_TEXT_REVIEWER_USER_TEMPLATE.replace(
        "{product_ingredients_json}", compact_json(names)
    ).replace("{concerns_json}", compact_json(concerns))


def _emit(line: str, warning: bool = False) -> None:
    with trace_step("concern_text_review") as fields:
        fields["debug_output"] = line
        if warning:
            fields["warnings"] = [line]


def _trace_change(key, field, change: dict) -> str:
    rule = change.get("rule") if isinstance(change.get("rule"), str) else ""
    frm = change.get("from") if isinstance(change.get("from"), str) else ""
    to = change.get("to") if isinstance(change.get("to"), str) else ""
    reason = change.get("reason") if isinstance(change.get("reason"), str) else ""
    return (
        f"concern_text_review: {key}.{field} rule={rule} "
        f'from="{frm}" to="{to}" reason="{reason}"'
    )


def _reviewed_by_key(parsed) -> dict:
    by_key = {}
    items = parsed.get("concerns") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return by_key
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("concern_key")
        if key not in by_key:
            by_key[key] = item
    return by_key


def _fields_by_name(reviewed) -> dict:
    by_name = {}
    items = reviewed.get("fields") if isinstance(reviewed, dict) else None
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


def _apply_review(rows, parsed) -> tuple[bool, list[str], list[str], int]:
    by_key = _reviewed_by_key(parsed)
    changed = False
    change_lines = []
    warnings = []
    changed_fields = set()
    for row in rows:
        key = row.get("concern_key")
        reviewed = by_key.get(key)
        if reviewed is None:
            warnings.append(
                f"concern_text_review: {key} missing from reviewer output "
                "— original texts kept"
            )
            continue
        fields = _fields_by_name(reviewed)
        for field in TEXT_FIELDS:
            item = fields.get(field)
            if item is None:
                warnings.append(
                    f"concern_text_review: {key}.{field} missing from "
                    "reviewer output — original texts kept"
                )
                continue
            text = item.get("text")
            if _empty_text(text):
                warnings.append(
                    f"concern_text_review: {key}.{field} empty reviewer text "
                    "— original texts kept"
                )
                continue
            if text != row.get(field):
                row[field] = text
                changed = True
            changes = item.get("changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, dict):
                    continue
                change_lines.append(_trace_change(key, field, change))
                changed_fields.add((key, field))
    return changed, change_lines, warnings, len(changed_fields)


def review_concern_store_texts(store_data, synergy, product_names=None) -> str:
    if not isinstance(store_data, str):
        return store_data
    try:
        rows = json.loads(store_data)
    except Exception:
        return store_data
    if not isinstance(rows, list):
        return store_data
    work = [dict(row) for row in rows if isinstance(row, dict)]
    if not work:
        return store_data
    concerns = build_concerns_payload(work, synergy)
    names = (
        list(product_names)
        if product_names is not None
        else product_names_from_context()
    )
    user = render_reviewer_user(names, concerns)
    try:
        result = run_llm_call(
            api="chat",
            model=CONCERN_TEXT_REVIEWER_MODEL,
            timeout_seconds=CONCERN_TEXT_REVIEWER_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": CONCERN_TEXT_REVIEWER_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=CONCERN_TEXT_REVIEWER_MAX_TOKENS,
            schema_name=CONCERN_TEXT_REVIEWER_SCHEMA_NAME,
            schema=CONCERN_TEXT_REVIEWER_SCHEMA,
            step="concern_text_reviewer",
        )
        parsed = json.loads(result.text)
        if not isinstance(parsed, dict):
            raise ValueError("reviewer reply is not an object")
        changed, change_lines, warnings, field_count = _apply_review(work, parsed)
    except Exception as exc:
        _emit(
            f"concern_text_review: FAILED ({exc}) — original texts kept",
            warning=True,
        )
        return store_data
    for line in change_lines:
        _emit(line)
    for line in warnings:
        _emit(line, warning=True)
    _emit(
        f"concern_text_review: {len(change_lines)} changes across {field_count} fields"
    )
    if not changed:
        return store_data
    return compact_json(work)
