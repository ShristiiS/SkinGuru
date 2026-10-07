from __future__ import annotations

from clients.supabase import (
    store_approx_concentrations as _store_approx_concentrations,
    store_formulation_type as _store_formulation_type,
)
from precompute.call_retry import call_with_retry
from precompute.concentration.estimator import estimate_concentrations
from tracing import traced


@traced("estimate_concentrations")
def run_estimator(ingredients: list) -> dict:
    """Node 7 — call the Python estimator in-process. Do not call n8n."""
    return estimate_concentrations(ingredients)


@traced("store_approx_concentrations")
def store_approx_concentrations(product_id, node7: dict) -> None:
    """Node 8 — POST product_concentrations (formulation_product_type added)."""
    call_with_retry(
        _store_approx_concentrations,
        product_id,
        node7["concentrations"],
        node7["formulation_product_type"],
    )


@traced("store_formulation_type")
def store_formulation_type(product_id, node7: dict) -> None:
    """Node 9 — POST product_scores?on_conflict=product_id."""
    call_with_retry(
        _store_formulation_type, product_id, node7["formulation_product_type"]
    )
