from __future__ import annotations

from clients.supabase import (
    get_formulation_type_from_concentrations,
    get_formulation_type_from_scores,
    get_product_concern_data,
)
from precompute.flow1.agent import (
    run_concern_agent,
    run_formulation_agent,
    run_interaction_builder_agent,
    run_safety_agent,
)
from precompute.flow1.interaction_validate import validate_interaction_result
from precompute.flow1.llm import chat_completions
from precompute.flow1.prompts import (
    SYNERGY_REASONING_SYSTEM_EXPORT,
    render_synergy_reasoning_user,
    strip_leading_equals,
)
from tracing import record_prompt_values, traced


def _usable(value) -> bool:
    return value is not None and value != ""


def _first_usable(rows: list):
    if not rows or not isinstance(rows[0], dict):
        return None
    value = rows[0].get("formulation_product_type")
    if _usable(value):
        return value
    return None


@traced("pass_through")
def pass_through(product_id, estimator_result) -> dict:
    """Node 22 — Pass Through. PORT DECISION CHANGED: check both tables."""
    if estimator_result is not None:
        formulation_product_type = estimator_result["formulation_product_type"]
    else:
        formulation_product_type = _first_usable(
            get_formulation_type_from_scores(product_id)
        )
        if formulation_product_type is None:
            formulation_product_type = _first_usable(
                get_formulation_type_from_concentrations(product_id)
            )

    return {
        "product_id": product_id,
        "formulation_product_type": formulation_product_type,
    }


@traced("get_concern_data")
def get_concern_data(product_id):
    """Node 23 — Get Concern Data. One object, key order kept."""
    return get_product_concern_data(product_id)


@traced("synergy_reasoning")
def synergy_reasoning(concern_data) -> str:
    """Node 24 — Synergy Reasoning. gpt-4o-mini, reply kept as-is."""
    data = concern_data if isinstance(concern_data, dict) else {}
    record_prompt_values(
        {
            "concern_keys": data.get("concern_keys"),
            "concern_tiers": data.get("concern_tiers"),
            "concentrations": data.get("concentrations"),
        }
    )
    return chat_completions(
        strip_leading_equals(SYNERGY_REASONING_SYSTEM_EXPORT),
        render_synergy_reasoning_user(concern_data),
        "gpt-4o-mini",
        180,
    )


@traced("concern_agent")
def concern_agent(node22: dict, node23, synergy_text: str) -> str:
    """Node 26 — Concern Agent. Final text is unused."""
    return run_concern_agent(node22, node23, synergy_text)


@traced("run_concern")
def run_concern(node22: dict) -> None:
    """Nodes 23, 24, 26."""
    node23 = get_concern_data(node22["product_id"])
    synergy_text = synergy_reasoning(node23)
    concern_agent(node22, node23, synergy_text)


@traced("interaction_builder_agent")
def interaction_builder_agent(product_id):
    raw, tool_log = run_interaction_builder_agent(product_id)
    return validate_interaction_result(raw, tool_log)


@traced("run_interaction_builder")
def run_interaction_builder(product_id) -> dict:
    interaction_result = interaction_builder_agent(product_id)
    return {
        "product_id": product_id,
        "interaction_result": interaction_result,
    }


@traced("safety_agent")
def safety_agent(node22: dict, interaction_result) -> str:
    """Node 30 — Safety Agent. Final text is unused."""
    return run_safety_agent(node22, interaction_result)


@traced("run_safety")
def run_safety(node22: dict, interaction_result) -> None:
    """Node 30."""
    safety_agent(node22, interaction_result)


@traced("formulation_agent")
def formulation_agent(node22: dict) -> str:
    """Node 36 — Formulation Agent. Final text is unused."""
    return run_formulation_agent(node22)


@traced("run_formulation")
def run_formulation(node22: dict) -> None:
    """Node 36. Node 42 is a no-op; the product loop continues after this."""
    formulation_agent(node22)
