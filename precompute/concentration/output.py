from __future__ import annotations

INSTRUCTIONS = (
    "Pass only 'concentrations' to run_efficacy_analyzer. "
    "Pass only 'formulation_product_type' to run_formulation_analyzer."
)


def format_estimator_output(node15: dict, product_type) -> dict:
    """Node 16 — last node. Warnings are not included."""
    return {
        "concentrations": node15["concentrations"],
        "formulation_product_type": product_type,
        "_instructions": INSTRUCTIONS,
    }
