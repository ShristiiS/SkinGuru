from __future__ import annotations

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.llm_checks import (
    ALLOWED_PRODUCT_TYPE_LIST,
    check_llm1,
)
from precompute.concentration.merge import merge_after_llm1
from precompute.concentration.prompts import LLM1_SYSTEM, render_llm1_user
from precompute.llm_call import run_llm_call
from tracing import traced

LLM1_MODEL = "gpt-4.1-mini"
LLM1_MAX_TOKENS = 1000
LLM1_SCHEMA_NAME = "llm1_product_type"
LLM1_SCHEMA = {
    "type": "object",
    "properties": {
        "product_type": {
            "type": "string",
            "enum": list(ALLOWED_PRODUCT_TYPE_LIST),
        }
    },
    "required": ["product_type"],
    "additionalProperties": False,
}


def _record_quality(run, problems) -> None:
    from precompute.flow1.product_record import record_rerun

    record_rerun("llm1", run, "\n".join(problems))


def _llm1_check(ingredients: list):
    def check(text: str):
        node = merge_after_llm1(text, ingredients)
        check_llm1(node)
        return []

    return check


@traced("llm1_product_type")
def run_llm1(ingredients: list) -> str:
    """Node 3 — LLM1 Product Type + Marker."""
    result = run_llm_call(
        api="chat",
        model=LLM1_MODEL,
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": LLM1_SYSTEM},
            {"role": "user", "content": render_llm1_user(ingredients)},
        ],
        max_tokens=LLM1_MAX_TOKENS,
        schema_name=LLM1_SCHEMA_NAME,
        schema=LLM1_SCHEMA,
        check=_llm1_check(ingredients),
        step="llm1",
        on_quality_failure=_record_quality,
    )
    return result.text
