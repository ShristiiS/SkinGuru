from __future__ import annotations

from precompute.concentration.calculate import (
    calculate_concentrations as _calculate_concentrations,
)
from precompute.concentration.llm1 import run_llm1
from precompute.concentration.llm2 import run_llm2
from precompute.concentration.llm3 import run_llm3
from precompute.concentration.llm4 import run_llm4
from precompute.concentration.llm_checks import (
    check_llm1,
    check_llm2,
    check_llm3,
    check_llm4,
)
from precompute.concentration.merge import (
    merge_after_llm1,
    merge_after_llm2,
    merge_after_llm3,
    merge_after_llm4,
)
from precompute.concentration.output import (
    format_estimator_output as _format_estimator_output,
)
from precompute.concentration.parse import parse_input as _parse_input
from tracing import traced, trace_step
from tracing.debug import to_jsonable

parse_input = traced("parse_input")(_parse_input)
calculate_concentrations = traced("calculate_concentrations")(
    _calculate_concentrations
)
format_estimator_output = traced("format_estimator_output")(
    _format_estimator_output
)

LLM_STEP_MAX_RUNS = 3


def _repeat_llm_step(fn, step_name: str):
    from precompute.flow1.product_record import record_rerun

    last_error: BaseException | None = None
    for run in range(1, LLM_STEP_MAX_RUNS + 1):
        try:
            with trace_step(f"{step_name}_run") as fields:
                fields["debug_input"] = {"run": run}
                result = fn()
                fields["debug_output"] = to_jsonable(result)
                return result
        except Exception as exc:
            last_error = exc
            record_rerun(step_name, run, str(exc))
            if run == LLM_STEP_MAX_RUNS:
                raise
    raise last_error


def estimate_concentrations(ingredients: list) -> dict:
    """Concentration Estimator: nodes 2 → 3 → 5 → 6 → 8 → 9 → 11 → 12 → 14 → 15 → 16."""
    node2 = parse_input(ingredients)
    parsed_ingredients = node2["ingredients"]

    def llm1_step():
        text = run_llm1(parsed_ingredients)
        node = merge_after_llm1(text, parsed_ingredients)
        check_llm1(node)
        return text, node

    llm1_text, node5 = _repeat_llm_step(llm1_step, "llm1")

    def llm2_step():
        text = run_llm2(node5["product_type"], node5["ingredients"])
        node = merge_after_llm2(llm1_text, text, parsed_ingredients)
        check_llm2(node, parsed_ingredients)
        return node

    node8 = _repeat_llm_step(llm2_step, "llm2")

    def llm3_step():
        text = run_llm3(
            node8["product_type"],
            node8["marker_position"],
            node8["marker_ingredient"],
            node8["ingredients"],
        )
        node = merge_after_llm3(
            text,
            node8["product_type"],
            node8["marker_position"],
            node8["marker_ingredient"],
        )
        check_llm3(node, parsed_ingredients)
        return node

    node11 = _repeat_llm_step(llm3_step, "llm3")

    def llm4_step():
        text = run_llm4(
            node11["product_type"],
            node11["marker_position"],
            node11["marker_ingredient"],
            node11["classified_ingredients"],
        )
        node = merge_after_llm4(
            text,
            node11["product_type"],
            node11["marker_position"],
            node11["marker_ingredient"],
            node11["classified_ingredients"],
        )
        check_llm4(node, parsed_ingredients)
        return node

    node14 = _repeat_llm_step(llm4_step, "llm4")

    node15 = calculate_concentrations(node14)
    return format_estimator_output(node15, node11["product_type"])
