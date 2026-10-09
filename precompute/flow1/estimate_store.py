from __future__ import annotations

import time

from clients.supabase import (
    get_product_concentration_fields,
    get_product_concentration_row,
    patch_product_concentrations as _patch_product_concentrations,
    store_approx_concentrations as _store_approx_concentrations,
    store_formulation_type as _store_formulation_type,
)
from config import FLOW1_CALL_RETRY_ATTEMPTS, FLOW1_CALL_RETRY_WAIT_SECONDS
from precompute.call_retry import call_with_retry
from precompute.concentration.estimator import estimate_concentrations
from precompute.concentration.llm_checks import is_valid_product_type
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


def concentrations_are_present(value) -> bool:
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    return False


def _concentration_row_exists(product_id) -> bool:
    rows = call_with_retry(get_product_concentration_row, product_id)
    return any(isinstance(item, dict) and len(item.keys()) > 0 for item in rows)


def confirmed_concentration_type(product_id):
    rows = call_with_retry(get_product_concentration_fields, product_id)
    if not rows or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    if not concentrations_are_present(row.get("concentrations")):
        return None
    product_type = row.get("formulation_product_type")
    if not is_valid_product_type(product_type):
        return None
    return product_type


@traced("save_estimator_concentrations")
def save_estimator_concentrations(product_id, node7: dict) -> None:
    """Fallback save: PATCH existing product_concentrations row, else POST. No scores write."""
    concentrations = node7["concentrations"]
    product_type = node7["formulation_product_type"]
    if _concentration_row_exists(product_id):
        call_with_retry(
            _patch_product_concentrations,
            product_id,
            concentrations,
            product_type,
        )
        return
    call_with_retry(
        _store_approx_concentrations,
        product_id,
        concentrations,
        product_type,
    )


@traced("save_concentrations_with_verify")
def save_concentrations_with_verify(product_id, node7: dict):
    """Save, re-GET, retry save max 3 / 5s if concentrations or type not confirmed."""
    last_type = None
    for run in range(1, FLOW1_CALL_RETRY_ATTEMPTS + 1):
        try:
            save_estimator_concentrations(product_id, node7)
            last_type = confirmed_concentration_type(product_id)
        except Exception:
            last_type = None
        if last_type is not None:
            return last_type
        if run < FLOW1_CALL_RETRY_ATTEMPTS:
            time.sleep(FLOW1_CALL_RETRY_WAIT_SECONDS)
    return last_type
