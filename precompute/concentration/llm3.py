from __future__ import annotations

from precompute.concentration.llm import chat_completions
from precompute.concentration.prompts import LLM3_SYSTEM, render_llm3_user
from tracing import traced

LLM3_MODEL = "gpt-4.1-mini"
LLM3_MAX_TOKENS = 3000


@traced("llm3_classification_zones")
def run_llm3(
    product_type, marker_position, marker_ingredient, ingredients: list
) -> str:
    """Node 9 — LLM3 Classification + Zones."""
    return chat_completions(
        LLM3_SYSTEM,
        render_llm3_user(
            product_type, marker_position, marker_ingredient, ingredients
        ),
        LLM3_MODEL,
        LLM3_MAX_TOKENS,
    )
