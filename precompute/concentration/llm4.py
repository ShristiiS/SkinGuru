from __future__ import annotations

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.llm_checks import ALLOWED_ZONE_LIST, check_llm4
from precompute.concentration.merge import merge_after_llm4, unwrap_items_reply
from precompute.concentration.prompts import LLM4_SYSTEM, render_llm4_user
from precompute.llm_call import run_llm_call
from tracing import traced

LLM4_MODEL = "gpt-4.1"
LLM4_MAX_TOKENS = 8000
LLM4_SCHEMA_NAME = "llm4_bands"
LLM4_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ingredient_name": {"type": "string"},
                    "display_order": {"type": "number"},
                    "zone": {
                        "type": "string",
                        "enum": list(ALLOWED_ZONE_LIST),
                    },
                    "band_L": {"type": "number"},
                    "band_U": {"type": "number"},
                },
                "required": [
                    "ingredient_name",
                    "display_order",
                    "zone",
                    "band_L",
                    "band_U",
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

    record_rerun("llm4", run, "\n".join(problems))


def _llm4_check(
    product_type,
    marker_position,
    marker_ingredient,
    classified_ingredients: list,
    ingredients: list,
):
    def check(text: str):
        node = merge_after_llm4(
            unwrap_items_reply(text),
            product_type,
            marker_position,
            marker_ingredient,
            classified_ingredients,
        )
        check_llm4(node, ingredients)
        return []

    return check


@traced("llm4_band_building")
def run_llm4(
    product_type,
    marker_position,
    marker_ingredient,
    classified_ingredients: list,
) -> str:
    """Node 12 — LLM4 Band Building."""
    result = run_llm_call(
        api="chat",
        model=LLM4_MODEL,
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": LLM4_SYSTEM},
            {
                "role": "user",
                "content": render_llm4_user(
                    product_type,
                    marker_position,
                    marker_ingredient,
                    classified_ingredients,
                ),
            },
        ],
        max_tokens=LLM4_MAX_TOKENS,
        schema_name=LLM4_SCHEMA_NAME,
        schema=LLM4_SCHEMA,
        check=_llm4_check(
            product_type,
            marker_position,
            marker_ingredient,
            classified_ingredients,
            classified_ingredients,
        ),
        step="llm4",
        on_quality_failure=_record_quality,
    )
    return unwrap_items_reply(result.text)
