from __future__ import annotations

import json
import re

from precompute.concentration.helpers import compact_json
from tracing import traced

_LLM4_EMPTY_PREFIX = "LLM4 output empty — got: "


def strip_code_fences(text: str) -> str:
    """n8n `.replace(/```json|```/g, '').trim()`."""
    return re.sub(r"```json|```", "", text).strip()


def unwrap_items_reply(content: str) -> str:
    """Turn {\"items\": [...]} into the JSON array merge_after_llm3/4 expect."""
    try:
        parsed = json.loads(strip_code_fences(content))
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return json.dumps(parsed["items"])
    return content


@traced("merge_after_llm1")
def merge_after_llm1(llm1_text: str, ingredients: list) -> dict:
    """Node 5 — Merge After LLM1."""
    cleaned = strip_code_fences(llm1_text)
    llm1_result = json.loads(cleaned)
    return {
        "ingredients": ingredients,
        "product_type": llm1_result["product_type"],
    }


_MARKER_JSON = re.compile(r'\{[^{}]*"marker_position"[^{}]*\}')
_MARKER_MISSING = "Could not find marker JSON in LLM2 output"


@traced("merge_after_llm2")
def merge_after_llm2(llm1_text: str, llm2_text: str, ingredients: list) -> dict:
    """Node 8 — Merge After LLM2."""
    llm1_cleaned = strip_code_fences(llm1_text)
    llm1_result = json.loads(llm1_cleaned)

    json_match = _MARKER_JSON.search(llm2_text)
    if not json_match:
        raise ValueError(_MARKER_MISSING)
    llm2_result = json.loads(json_match.group(0))

    return {
        "ingredients": ingredients,
        "product_type": llm1_result["product_type"],
        "marker_position": llm2_result["marker_position"],
        "marker_ingredient": llm2_result["marker_ingredient"],
    }


@traced("merge_after_llm3")
def merge_after_llm3(
    llm3_text: str,
    product_type,
    marker_position,
    marker_ingredient,
) -> dict:
    """Node 11 — Merge After LLM3."""
    cleaned = strip_code_fences(llm3_text)
    classified_ingredients = json.loads(cleaned)
    return {
        "product_type": product_type,
        "marker_position": marker_position,
        "marker_ingredient": marker_ingredient,
        "classified_ingredients": classified_ingredients,
    }


@traced("merge_after_llm4")
def merge_after_llm4(
    llm4_text: str,
    product_type,
    marker_position,
    marker_ingredient,
    classified_ingredients: list,
) -> dict:
    """Node 14 — Merge After LLM4."""
    if not llm4_text:
        raise ValueError(_LLM4_EMPTY_PREFIX + compact_json(llm4_text))

    cleaned = strip_code_fences(llm4_text)
    bands = json.loads(cleaned)

    merged = []
    for band in bands:
        classified = next(
            (
                item
                for item in classified_ingredients
                if item["ingredient_name"] == band["ingredient_name"]
            ),
            None,
        )
        merged.append(
            {
                "ingredient_name": band["ingredient_name"],
                "display_order": band["display_order"],
                "classification": (
                    classified["classification"] if classified else "unknown"
                ),
                "zone": band["zone"],
                "band_L": band["band_L"],
                "band_U": band["band_U"],
            }
        )

    return {
        "product_type": product_type,
        "marker_position": marker_position,
        "marker_ingredient": marker_ingredient,
        "ingredients_with_bands": merged,
    }
