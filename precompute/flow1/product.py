from __future__ import annotations

from clients.supabase import (
    get_precompute_product_ingredients,
    get_product_concentration_row,
)
from precompute.call_retry import call_with_retry
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
from precompute.flow1.product_record import (
    bind_product_record,
    current_product_record,
    next_status_step,
    record_step,
    skip_from,
)
from precompute.flow1.search import search_concentrations
from precompute.flow1.status_block import append_status_block
from tracing import bind_product, log_product_totals, traced


def _fail_early(step: str, exc: BaseException) -> None:
    record_step(step, "FAILED", str(exc))
    following = next_status_step(step)
    if following is not None:
        skip_from(following, "previous step failed")


@traced("get_product_ingredients")
def get_product_ingredients(product_id) -> dict:
    """Node 4 — Get Product Ingredients."""
    rows = call_with_retry(get_precompute_product_ingredients, product_id)
    return {"product_id": product_id, "ingredients": rows}


@traced("check_concentration_exists")
def check_concentration_exists(product_id) -> list:
    """Node 5 — Check Concentration Exists. Empty list = n8n empty item."""
    return call_with_retry(get_product_concentration_row, product_id)


@traced("already_exists")
def already_exists(items: list) -> bool:
    """Node 6 — `$input.all().filter(i => Object.keys(i.json).length > 0).length > 0`."""
    return any(
        isinstance(item, dict) and len(item.keys()) > 0 for item in items
    )


@traced("process_one_product")
def process_one_product(product_id) -> dict:
    """Nodes 4–36. Node 42 is a no-op; the caller loops to the next product."""
    with bind_product(str(product_id)), bind_product_record(product_id):
        try:
            try:
                node4 = get_product_ingredients(product_id)
                record_step("ingredients", "OK")
            except Exception as exc:
                _fail_early("ingredients", exc)
                raise
            try:
                rows = check_concentration_exists(product_id)
                record_step("concentration_check", "OK")
            except Exception as exc:
                _fail_early("concentration_check", exc)
                raise
            estimator_result = None
            if not already_exists(rows):
                try:
                    estimator_result = run_estimator(node4["ingredients"])
                    record_step("estimator", "OK")
                except Exception as exc:
                    _fail_early("estimator", exc)
                    raise
                try:
                    store_approx_concentrations(product_id, estimator_result)
                    store_formulation_type(product_id, estimator_result)
                    record_step("stores", "OK")
                except Exception as exc:
                    _fail_early("stores", exc)
                    raise
            else:
                record_step("estimator", "OK", "already existed")
                record_step("stores", "OK")
            try:
                search_concentrations(node4, estimator_result)
                record_step("serpapi", "OK")
            except Exception as exc:
                record_step("serpapi", "FAILED", str(exc))
            try:
                node22 = pass_through(node4["product_id"], estimator_result)
            except Exception as exc:
                skip_from("synergy_reasoning", str(exc))
                return {"product_id": product_id}
            run_concern(node22)
            ib_result = None
            try:
                ib = run_interaction_builder(product_id)
                ib_result = ib["interaction_result"]
            except Exception as exc:
                record_step("interaction_builder", "FAILED", str(exc))
                record_step("safety", "SKIPPED", "interaction builder failed")
            if ib_result is not None:
                run_safety(node22, ib_result)
            run_formulation(node22)
            return node22
        finally:
            record = current_product_record()
            if record is not None:
                append_status_block(record)
            log_product_totals()
