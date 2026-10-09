from __future__ import annotations

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.llm_checks import (
    ALLOWED_CLASSIFICATION_LIST,
    ALLOWED_ZONE_LIST,
    check_llm3,
)
from precompute.concentration.merge import merge_after_llm3, unwrap_items_reply
from precompute.concentration.prompts import LLM3_SYSTEM, render_llm3_user
from precompute.llm_call import run_llm_call
from tracing import traced

LLM3_MODEL = "gpt-4.1-mini"
LLM3_MAX_TOKENS = 3000
LLM3_SCHEMA_NAME = "llm3_classified"
LLM3_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ingredient_name": {"type": "string"},
                    "display_order": {"type": "number"},
                    "classification": {
                        "type": "string",
                        "enum": list(ALLOWED_CLASSIFICATION_LIST),
                    },
                    "zone": {
                        "type": "string",
                        "enum": list(ALLOWED_ZONE_LIST),
                    },
                },
                "required": [
                    "ingredient_name",
                    "display_order",
                    "classification",
                    "zone",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def _record_quality(run, problems) -> None:
    from precompute.flow1.product_record import record_rerun

    record_rerun("llm3", run, "\n".join(problems))


def _llm3_check(product_type, marker_position, marker_ingredient, ingredients):
    def check(text: str):
        node = merge_after_llm3(
            unwrap_items_reply(text),
            product_type,
            marker_position,
            marker_ingredient,
        )
        check_llm3(node, ingredients)
        return []

    return check


@traced("llm3_classification_zones")
def run_llm3(
    product_type, marker_position, marker_ingredient, ingredients: list
) -> str:
    """Node 9 — LLM3 Classification + Zones."""
    result = run_llm_call(
        api="chat",
        model=LLM3_MODEL,
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": LLM3_SYSTEM},
            {
                "role": "user",
                "content": render_llm3_user(
                    product_type,
                    marker_position,
                    marker_ingredient,
                    ingredients,
                ),
            },
        ],
        max_tokens=LLM3_MAX_TOKENS,
        schema_name=LLM3_SCHEMA_NAME,
        schema=LLM3_SCHEMA,
        check=_llm3_check(
            product_type, marker_position, marker_ingredient, ingredients
        ),
        step="llm3",
        on_quality_failure=_record_quality,
    )
    return unwrap_items_reply(result.text)
