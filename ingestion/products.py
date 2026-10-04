from dataclasses import dataclass
from typing import Any

from clients.supabase import (
    get_product_row_by_nykaa_url,
    insert_product,
    update_product_by_nykaa_url,
)
from tracing import traced

# Nodes 10–11 write these 9 fields; Insert also writes nykaa_url.
_PRODUCT_WRITE_FIELDS = (
    "product_name",
    "brand",
    "price",
    "rating",
    "total_reviews",
    "images",
    "description",
    "how_to_use",
    "key_ingredients",
)


@dataclass
class ProductWriteResult:
    """Carries both ID sources so later nodes can use the Insert-then-Check fallback."""

    action: str
    insert_executed: bool
    check_product_exists: dict
    insert_product: dict | None
    update_product: dict | None

    def product_id(self) -> Any:
        # Nodes 26 / 29 / 33: Insert Product id if that node ran, else CHECK PRODUCT EXISTS id.
        if self.insert_executed:
            return self.insert_product["id"]
        return self.check_product_exists["id"]

    def product_id_try_insert_else_check(self) -> Any:
        # Nodes 29 / 33: try Insert Product's id, else CHECK PRODUCT EXISTS's id.
        try:
            return self.insert_product["id"]
        except (TypeError, KeyError):
            return self.check_product_exists["id"]

    def extract_input_id(self) -> Any:
        # Node 12: $input.first().json.id from whichever of Insert/Update ran.
        if self.insert_executed:
            return self.insert_product["id"]
        return self.update_product["id"]

    def check_or_insert_id(self) -> Any:
        # Node 28: CHECK PRODUCT EXISTS id || Insert Product id
        check_id = self.check_product_exists.get("id")
        if check_id is not None:
            return check_id
        return self.insert_product["id"]


def _write_fields(parsed: dict) -> dict:
    return {field: parsed.get(field) for field in _PRODUCT_WRITE_FIELDS}


@traced("upsert_parsed_product")
def upsert_parsed_product(parsed: dict) -> ProductWriteResult:
    """Nodes 8–11: CHECK PRODUCT EXISTS → Update or Insert."""
    existing = get_product_row_by_nykaa_url(parsed["nykaa_url"])

    # Node 9 — PRODUCT EXISTS?  Object.keys($json).length > 0
    if len(existing.keys()) > 0:
        updated_rows = update_product_by_nykaa_url(
            parsed["nykaa_url"],
            _write_fields(parsed),
        )
        return ProductWriteResult(
            action="updated",
            insert_executed=False,
            check_product_exists=existing,
            insert_product=None,
            update_product=updated_rows[0] if updated_rows else None,
        )

    inserted_rows = insert_product(
        {**_write_fields(parsed), "nykaa_url": parsed["nykaa_url"]}
    )
    insert_row = inserted_rows[0] if inserted_rows else None
    return ProductWriteResult(
        action="inserted",
        insert_executed=True,
        check_product_exists=existing,
        insert_product=insert_row,
        update_product=None,
    )
