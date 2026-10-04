from __future__ import annotations

from precompute.concentration.llm import chat_completions
from precompute.concentration.prompts import LLM1_SYSTEM, render_llm1_user
from tracing import traced

LLM1_MODEL = "gpt-4.1-mini"
LLM1_MAX_TOKENS = 1000


@traced("llm1_product_type")
def run_llm1(ingredients: list) -> str:
    """Node 3 — LLM1 Product Type + Marker."""
    return chat_completions(
        LLM1_SYSTEM,
        render_llm1_user(ingredients),
        LLM1_MODEL,
        LLM1_MAX_TOKENS,
    )
