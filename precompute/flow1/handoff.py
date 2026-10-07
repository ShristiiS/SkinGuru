from __future__ import annotations

from clients.supabase import (
    get_formulation_type_from_concentrations,
    get_formulation_type_from_scores,
    get_product_concern_data,
)
from precompute.call_retry import call_with_retry
from precompute.flow1.agent import (
    run_concern_agent,
    run_formulation_agent,
    run_interaction_builder_agent,
    run_safety_agent,
)
from precompute.flow1.concern_validate import validate_concern_agent
from precompute.flow1.formulation_validate import validate_formulation_agent
from precompute.flow1.interaction_validate import (
    validate_interaction_builder_checks,
    validate_interaction_result,
)
from precompute.flow1 import llm as flow1_llm
from precompute.flow1.safety_validate import validate_safety_agent
from precompute.flow1.product_record import (
    current_product_record,
    record_rerun,
    record_step,
)
from precompute.flow1.prompts import (
    SYNERGY_REASONING_SYSTEM_EXPORT,
    render_synergy_reasoning_user,
    strip_leading_equals,
)
from precompute.flow1.run_until_ok import AGENT_MAX_RUNS, run_until_ok
from precompute.flow1.synergy_validate import (
    collect_synergy_problems,
    format_synergy_feedback,
    validate_synergy_reply,
)
from tracing import record_prompt_values, traced, trace_step


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
            call_with_retry(get_formulation_type_from_scores, product_id)
        )
        if formulation_product_type is None:
            formulation_product_type = _first_usable(
                call_with_retry(
                    get_formulation_type_from_concentrations, product_id
                )
            )

    return {
        "product_id": product_id,
        "formulation_product_type": formulation_product_type,
    }


@traced("get_concern_data")
def get_concern_data(product_id):
    """Node 23 — Get Concern Data. One object, key order kept."""
    return call_with_retry(get_product_concern_data, product_id)


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
    concern_keys = data.get("concern_keys") or []
    system = strip_leading_equals(SYNERGY_REASONING_SYSTEM_EXPORT)
    user = render_synergy_reasoning_user(concern_data)
    previous_text = None
    feedback = None
    last_error: BaseException | None = None
    for run in range(1, AGENT_MAX_RUNS + 1):
        try:
            with trace_step("synergy_reasoning_run") as fields:
                fields["debug_input"] = {"run": run}
                if feedback:
                    fields["debug_input"]["problems"] = feedback
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                if previous_text is not None and feedback is not None:
                    messages.append(
                        {"role": "assistant", "content": previous_text}
                    )
                    messages.append({"role": "user", "content": feedback})
                message = flow1_llm.chat_completions_turn(
                    messages, "gpt-4o-mini", 180
                )
                text = message.get("content") or ""
                previous_text = text
                parsed, problems = collect_synergy_problems(text, concern_keys)
                fields["debug_output"] = {
                    "text": text,
                    "problems": problems,
                }
                if problems:
                    feedback = format_synergy_feedback(problems, concern_keys)
                    fields["debug_output"]["feedback"] = feedback
                    raise ValueError("\n".join(problems))
                record = current_product_record()
                if record is not None:
                    record.synergy_parsed = parsed
                record_step("synergy_reasoning", "OK")
                return text
        except Exception as exc:
            last_error = exc
            record_rerun("synergy_reasoning", run, str(exc))
            if run == AGENT_MAX_RUNS:
                record_step(
                    "synergy_reasoning",
                    "FAILED",
                    f"run {run} of {AGENT_MAX_RUNS}: {exc}",
                )
                raise
    raise last_error


@traced("concern_agent")
def concern_agent(node22: dict, node23, synergy_text: str) -> str:
    """Node 26 — Concern Agent. Final text is unused."""
    data = node23 if isinstance(node23, dict) else {}
    record = current_product_record()
    synergy_obj = record.synergy_parsed if record is not None else None
    if synergy_obj is None:
        try:
            synergy_obj = validate_synergy_reply(
                synergy_text, data.get("concern_keys") or []
            )
        except Exception:
            synergy_obj = {}

    def one_run():
        result = run_concern_agent(
            node22, node23, synergy_text, synergy=synergy_obj
        )
        validate_concern_agent(
            result, product_id=node22["product_id"], synergy=synergy_obj
        )
        return result

    return run_until_ok(one_run, step="concern")


@traced("run_concern")
def run_concern(node22: dict) -> None:
    """Nodes 23, 24, 26. Concern fail does not stop IB / Safety / Formulation."""
    try:
        node23 = get_concern_data(node22["product_id"])
    except Exception as exc:
        record_step("synergy_reasoning", "SKIPPED", "get concern data failed")
        record_step("concern", "FAILED", str(exc))
        return
    try:
        synergy_text = synergy_reasoning(node23)
        concern_agent(node22, node23, synergy_text)
        record_step("concern", "OK")
    except Exception as exc:
        record_step("concern", "FAILED", str(exc))



@traced("interaction_builder_agent")
def interaction_builder_agent(product_id):
    def one_run():
        result = run_interaction_builder_agent(product_id)
        parsed = validate_interaction_result(result.text, result.tool_log)
        validate_interaction_builder_checks(parsed, result)
        return parsed

    parsed = run_until_ok(one_run, step="interaction_builder")
    record_step("interaction_builder", "OK")
    return parsed


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
    def one_run():
        result = run_safety_agent(node22, interaction_result)
        validate_safety_agent(result, interaction_result)
        return result

    return run_until_ok(one_run, step="safety")


@traced("run_safety")
def run_safety(node22: dict, interaction_result) -> None:
    """Node 30. Safety fail does not stop Formulation."""
    try:
        safety_agent(node22, interaction_result)
        record_step("safety", "OK")
    except Exception as exc:
        record_step("safety", "FAILED", str(exc))


@traced("formulation_agent")
def formulation_agent(node22: dict) -> str:
    """Node 36 — Formulation Agent. Final text is unused."""
    def one_run():
        result = run_formulation_agent(node22)
        validate_formulation_agent(result, node22)
        return result

    return run_until_ok(one_run, step="formulation")


@traced("run_formulation")
def run_formulation(node22: dict) -> None:
    """Node 36. Node 42 is a no-op; the product loop continues after this."""
    try:
        formulation_agent(node22)
        record_step("formulation", "OK")
    except Exception as exc:
        record_step("formulation", "FAILED", str(exc))
