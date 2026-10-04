from clients.supabase import (
    get_matched_product_ingredients,
    upsert_product_ingredient,
)
from ingestion.products import ProductWriteResult
from tracing import traced


@traced("store_product_ingredients")
def store_product_ingredients(ingredients: list[dict]) -> None:
    """Node 25 — one upsert_product_ingredient RPC per merged row."""
    for ing in ingredients:
        upsert_product_ingredient(
            {
                "p_product_id": ing["product_id"],
                "p_ingredient_id": ing["ingredient_id"],
                "p_ingredient_name": ing["ingredient_name"],
                "p_as_listed_on_label": ing["as_listed_on_label"],
                "p_is_key_ingredient": ing["is_key_ingredient"],
                "p_display_order": ing["display_order"],
                "p_matched": ing["matched"],
            }
        )


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
    return get_matched_product_ingredients(written.check_or_insert_id())
