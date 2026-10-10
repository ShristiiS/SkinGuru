from clients.supabase import (
    get_matched_product_ingredients,
    upsert_product_ingredient,
)
from ingestion.product_record import record_not_processed, record_rerun
from ingestion.products import ProductWriteResult
from precompute.call_retry import call_with_retry
from tracing import trace_step, traced

STORE_MAX_RUNS = 3
STORE_MISSING_REASON = "missing from re-GET"


def _dedupe_key(ing: dict):
    ingredient_id = ing.get("ingredient_id")
    if ingredient_id is None:
        return None
    return (ing.get("product_id"), ingredient_id)


def _display_order_sort_key(ing: dict):
    order = ing.get("display_order")
    if order is None:
        return (1, 0)
    return (0, order)


def _dedupe_note(kept: dict, skipped: list[dict]) -> str:
    orders = ", ".join(str(row.get("display_order")) for row in skipped)
    labels = ", ".join(
        f"'{row.get('as_listed_on_label')}'" for row in skipped
    )
    return (
        f"{kept.get('ingredient_name')}: kept display_order "
        f"{kept.get('display_order')}, skipped {orders} ({labels})"
    )


def dedupe_product_ingredients(ingredients: list[dict]) -> list[dict]:
    """Keep the lowest display_order per (product_id, ingredient_id). Deliberate change from n8n."""
    groups: dict = {}
    for ing in ingredients:
        key = _dedupe_key(ing)
        if key is None:
            continue
        groups.setdefault(key, []).append(ing)

    seen = set()
    result = []
    notes = []
    for ing in ingredients:
        key = _dedupe_key(ing)
        if key is None:
            result.append(ing)
            continue
        if key in seen:
            continue
        seen.add(key)
        group = groups[key]
        if len(group) == 1:
            result.append(ing)
            continue
        kept = min(group, key=_display_order_sort_key)
        skipped = [row for row in group if row is not kept]
        skipped.sort(key=_display_order_sort_key)
        merged = {
            **kept,
            "is_key_ingredient": any(
                row.get("is_key_ingredient") is True for row in group
            ),
        }
        result.append(merged)
        notes.append(_dedupe_note(kept, skipped))

    with trace_step("dedupe_product_ingredients") as fields:
        fields["debug_input"] = {"count": len(ingredients)}
        fields["debug_output"] = notes
    return result


def _upsert_row(ing: dict) -> None:
    call_with_retry(
        upsert_product_ingredient,
        {
            "p_product_id": ing["product_id"],
            "p_ingredient_id": ing["ingredient_id"],
            "p_ingredient_name": ing["ingredient_name"],
            "p_as_listed_on_label": ing["as_listed_on_label"],
            "p_is_key_ingredient": ing["is_key_ingredient"],
            "p_display_order": ing["display_order"],
            "p_matched": ing["matched"],
        },
    )


def _missing_matched(sent: list[dict], stored: list[dict]) -> list[dict]:
    found_ids = {row.get("ingredient_id") for row in stored}
    return [ing for ing in sent if ing["ingredient_id"] not in found_ids]


def _record_store_miss(name) -> dict:
    item = {"name": name, "reason": STORE_MISSING_REASON}
    record_not_processed(name, STORE_MISSING_REASON)
    with trace_step("ingredient_not_processed") as fields:
        fields["debug_input"] = {"canonical_name": name}
        fields["debug_output"] = {
            "result": "NOT PROCESSED",
            "reason": STORE_MISSING_REASON,
        }
    return item


@traced("store_product_ingredients")
def store_product_ingredients(
    ingredients: list[dict], written: ProductWriteResult
) -> tuple[list[dict], list[dict]]:
    """Node 25 store + node 28 re-GET. Missing matched ids: re-store those rows, max 3 runs."""
    ingredients = dedupe_product_ingredients(ingredients)
    sent_matched = [ing for ing in ingredients if ing.get("matched") is True]
    pending = list(ingredients)
    stored: list[dict] = []
    missing: list[dict] = list(sent_matched)
    for run in range(1, STORE_MAX_RUNS + 1):
        with trace_step("store_ingredients_run") as fields:
            fields["debug_input"] = {"run": run, "count": len(pending)}
            for ing in pending:
                _upsert_row(ing)
            stored = get_product_ingredients(written)
            missing = _missing_matched(sent_matched, stored)
            fields["debug_output"] = {
                "missing": [ing.get("ingredient_name") for ing in missing]
            }
            if not missing:
                return stored, []
            record_rerun("store_ingredients", run, STORE_MISSING_REASON)
            pending = missing
    not_processed = [
        _record_store_miss(ing.get("ingredient_name")) for ing in missing
    ]
    return stored, not_processed


@traced("get_product_id")
def get_product_id(written: ProductWriteResult):
    """Node 26 — Insert Product id if insert ran, else CHECK PRODUCT EXISTS id."""
    return written.product_id()


@traced("save_product_id")
def save_product_id(static_product_ids: list, product_id) -> None:
    """Node 27 — append to workflow static data; caller passes items through."""
    static_product_ids.append(product_id)


@traced("get_product_ingredients")
def get_product_ingredients(written: ProductWriteResult) -> list[dict]:
    """Node 28 — re-GET matched product_ingredients (main path, not skip)."""
    return call_with_retry(
        get_matched_product_ingredients, written.check_or_insert_id()
    )
