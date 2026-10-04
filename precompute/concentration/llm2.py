from __future__ import annotations

from precompute.concentration.llm import chat_completions
from precompute.concentration.prompts import LLM2_SYSTEM, render_llm2_user
from tracing import traced

LLM2_MODEL = "gpt-4.1-mini"
LLM2_MAX_TOKENS = 5000


@traced("llm2_marker_detection")
def run_llm2(product_type, ingredients: list) -> str:
    """Node 6 — LLM2 Marker Detection."""
    return chat_completions(
        LLM2_SYSTEM,
        render_llm2_user(product_type, ingredients),
        LLM2_MODEL,
        LLM2_MAX_TOKENS,
    )
