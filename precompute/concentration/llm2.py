from __future__ import annotations

import json

from config import OPENAI_TIMEOUT_SECONDS
from precompute.concentration.llm_checks import check_llm2
from precompute.concentration.merge import merge_after_llm2
from precompute.concentration.prompts import LLM2_SYSTEM, render_llm2_user
from precompute.llm_call import run_llm_call
from tracing import traced

LLM2_MODEL = "gpt-4.1-mini"
LLM2_MAX_TOKENS = 5000


def _record_quality(run, problems) -> None:
    from precompute.flow1.product_record import record_rerun

    record_rerun("llm2", run, "\n".join(problems))


def _llm2_check(product_type, ingredients: list):
    def check(text: str):
        node = merge_after_llm2(
            json.dumps({"product_type": product_type}),
            text,
            ingredients,
        )
        check_llm2(node, ingredients)
        return []

    return check


@traced("llm2_marker_detection")
def run_llm2(product_type, ingredients: list) -> str:
    """Node 6 — LLM2 Marker Detection."""
    result = run_llm_call(
        api="chat",
        model=LLM2_MODEL,
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": LLM2_SYSTEM},
            {
                "role": "user",
                "content": render_llm2_user(product_type, ingredients),
            },
        ],
        max_tokens=LLM2_MAX_TOKENS,
        check=_llm2_check(product_type, ingredients),
        step="llm2",
        on_quality_failure=_record_quality,
    )
    return result.text
