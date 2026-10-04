from __future__ import annotations

from precompute.concentration.llm import chat_completions
from precompute.concentration.prompts import LLM4_SYSTEM, render_llm4_user
from tracing import traced

LLM4_MODEL = "gpt-4.1"
LLM4_MAX_TOKENS = 8000


@traced("llm4_band_building")
def run_llm4(
    product_type, marker_position, marker_ingredient, classified_ingredients: list
) -> str:
    """Node 12 — LLM4 Band Building."""
    return chat_completions(
        LLM4_SYSTEM,
        render_llm4_user(
            product_type,
            marker_position,
            marker_ingredient,
            classified_ingredients,
        ),
        LLM4_MODEL,
        LLM4_MAX_TOKENS,
    )
