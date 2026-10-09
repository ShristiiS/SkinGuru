from __future__ import annotations

from clients.supabase import (
    get_precompute_product_ingredients,
    get_product_concern_data,
    get_product_concern_score_keys,
    get_product_concentration_row,
    get_product_safety_flag_row,
    get_product_scores_computation_row,
)
from precompute.call_retry import call_with_retry
from precompute.flow1.estimate_store import (
    run_estimator,
    store_approx_concentrations,
    store_formulation_type,
)
from precompute.concentration.llm_checks import is_valid_product_type
from precompute.flow1.handoff import (
    NO_VALID_PRODUCT_TYPE,
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


def _concern_keys_from_data(data) -> list | None:
    if not isinstance(data, dict):
        return None
    keys = data.get("concern_keys")
    if not isinstance(keys, list) or not keys:
        return None
    return [str(key) for key in keys]


def _nonempty_rows(rows) -> bool:
    return any(isinstance(item, dict) and len(item.keys()) > 0 for item in rows or [])


def _concern_complete(product_id) -> bool:
    data = call_with_retry(get_product_concern_data, product_id)
    expected = _concern_keys_from_data(data)
    if expected is None:
        return False
    rows = call_with_retry(get_product_concern_score_keys, product_id)
    have = {
        row.get("concern_key")
        for row in rows or []
        if isinstance(row, dict)
    }
    return set(expected).issubset(have)


def _safety_complete(product_id) -> bool:
    rows = call_with_retry(get_product_safety_flag_row, product_id)
    return _nonempty_rows(rows)


def _formulation_complete(product_id) -> bool:
    rows = call_with_retry(get_product_scores_computation_row, product_id)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("formulation_score") is None:
            continue
        if is_valid_product_type(row.get("formulation_product_type")):
            return True
    return False


@traced("check_already_computed")
def missing_precompute_parts(product_id) -> list[str]:
    """Return which of concern/safety/formulation are not fully stored."""
    missing = []
    try:
        if not _concern_complete(product_id):
            missing.append("concern")
    except Exception:
        missing.append("concern")
    try:
        if not _safety_complete(product_id):
            missing.append("safety")
    except Exception:
        missing.append("safety")
    try:
        if not _formulation_complete(product_id):
            missing.append("formulation")
    except Exception:
        missing.append("formulation")
    return missing


@traced("process_one_product")
def process_one_product(product_id) -> dict:
    """Nodes 4–36. Node 42 is a no-op; the caller loops to the next product."""
    with bind_product(str(product_id)), bind_product_record(product_id):
        try:
            missing = missing_precompute_parts(product_id)
            record = current_product_record()
            if record is not None:
                record.missing = list(missing)
            if not missing:
                if record is not None:
                    record.skipped = True
                    record.missing = []
                return {"product_id": product_id, "already_computed": True}
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
                node22 = pass_through(
                    node4["product_id"],
                    estimator_result,
                    node4["ingredients"],
                )
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
            if is_valid_product_type(node22.get("formulation_product_type")):
                run_formulation(node22)
            else:
                record_step("formulation", "FAILED", NO_VALID_PRODUCT_TYPE)
            return node22
        finally:
            record = current_product_record()
            if record is not None:
                append_status_block(record)
            log_product_totals()
