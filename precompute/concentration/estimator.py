from __future__ import annotations

from precompute.concentration.calculate import (
    calculate_concentrations as _calculate_concentrations,
)
from precompute.concentration.llm1 import run_llm1
from precompute.concentration.llm2 import run_llm2
from precompute.concentration.llm3 import run_llm3
from precompute.concentration.llm4 import run_llm4
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
from tracing import traced

parse_input = traced("parse_input")(_parse_input)
calculate_concentrations = traced("calculate_concentrations")(
    _calculate_concentrations
)
format_estimator_output = traced("format_estimator_output")(
    _format_estimator_output
)


def estimate_concentrations(ingredients: list) -> dict:
    """Concentration Estimator: nodes 2 → 3 → 5 → 6 → 8 → 9 → 11 → 12 → 14 → 15 → 16."""
    node2 = parse_input(ingredients)
    parsed_ingredients = node2["ingredients"]

    llm1_text = run_llm1(parsed_ingredients)
    node5 = merge_after_llm1(llm1_text, parsed_ingredients)

    llm2_text = run_llm2(node5["product_type"], node5["ingredients"])
    node8 = merge_after_llm2(llm1_text, llm2_text, parsed_ingredients)

    llm3_text = run_llm3(
        node8["product_type"],
        node8["marker_position"],
        node8["marker_ingredient"],
        node8["ingredients"],
    )
    node11 = merge_after_llm3(
        llm3_text,
        node8["product_type"],
        node8["marker_position"],
        node8["marker_ingredient"],
    )

    llm4_text = run_llm4(
        node11["product_type"],
        node11["marker_position"],
        node11["marker_ingredient"],
        node11["classified_ingredients"],
    )
    node14 = merge_after_llm4(
        llm4_text,
        node11["product_type"],
        node11["marker_position"],
        node11["marker_ingredient"],
        node11["classified_ingredients"],
    )

    node15 = calculate_concentrations(node14)
    return format_estimator_output(node15, node11["product_type"])
