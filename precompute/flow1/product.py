from __future__ import annotations

from clients.supabase import (
    get_precompute_product_ingredients,
    get_product_concentration_row,
)
from precompute.flow1.estimate_store import (
    run_estimator,
    store_approx_concentrations,
    store_formulation_type,
)
from precompute.flow1.handoff import (
    pass_through,
    run_concern,
    run_formulation,
    run_interaction_builder,
    run_safety,
)
from precompute.flow1.search import search_concentrations
from tracing import bind_product, log_product_totals, traced


@traced("get_product_ingredients")
def get_product_ingredients(product_id) -> dict:
    """Node 4 — Get Product Ingredients."""
    rows = get_precompute_product_ingredients(product_id)
    return {"product_id": product_id, "ingredients": rows}


@traced("check_concentration_exists")
def check_concentration_exists(product_id) -> list:
    """Node 5 — Check Concentration Exists. Empty list = n8n empty item."""
    return get_product_concentration_row(product_id)


@traced("already_exists")
def already_exists(items: list) -> bool:
    """Node 6 — `$input.all().filter(i => Object.keys(i.json).length > 0).length > 0`."""
    return any(
        isinstance(item, dict) and len(item.keys()) > 0 for item in items
    )


@traced("process_one_product")
def process_one_product(product_id) -> dict:
    """Nodes 4–36. Node 42 is a no-op; the caller loops to the next product."""
    with bind_product(str(product_id)):
        try:
            node4 = get_product_ingredients(product_id)
            rows = check_concentration_exists(product_id)
            estimator_result = None
            if not already_exists(rows):
                estimator_result = run_estimator(node4["ingredients"])
                store_approx_concentrations(product_id, estimator_result)
                store_formulation_type(product_id, estimator_result)
            search_concentrations(node4, estimator_result)
            node22 = pass_through(node4["product_id"], estimator_result)
            run_concern(node22)
            ib = run_interaction_builder(product_id)
            run_safety(node22, ib["interaction_result"])
            run_formulation(node22)
            return node22
        finally:
            log_product_totals()
