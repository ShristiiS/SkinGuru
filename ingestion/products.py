import time
from dataclasses import dataclass
from typing import Any

from clients.supabase import (
    get_product_row_by_nykaa_url,
    insert_product,
    update_product_by_nykaa_url,
)
from config import FLOW1_CALL_RETRY_ATTEMPTS, FLOW1_CALL_RETRY_WAIT_SECONDS
from precompute.call_retry import call_with_retry, is_retryable_call
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


def _require_product_row(row, action: str) -> dict:
    if not isinstance(row, dict) or row.get("id") is None:
        raise RuntimeError(f"{action} product returned empty")
    return row


def _insert_product_with_retry(fields: dict) -> list[dict]:
    """Insert with call retry; re-GET by nykaa_url before a retry and use the row if it exists."""
    nykaa_url = fields["nykaa_url"]
    last_error: BaseException | None = None
    attempts = FLOW1_CALL_RETRY_ATTEMPTS
    for attempt in range(attempts):
        try:
            return insert_product(fields)
        except Exception as exc:
            last_error = exc
            if not is_retryable_call(exc) or attempt == attempts - 1:
                raise
            existing = call_with_retry(get_product_row_by_nykaa_url, nykaa_url)
            if existing:
                return [existing]
            time.sleep(FLOW1_CALL_RETRY_WAIT_SECONDS)
    raise last_error


@traced("upsert_parsed_product")
def upsert_parsed_product(parsed: dict) -> ProductWriteResult:
    """Nodes 8–11: CHECK PRODUCT EXISTS → Update or Insert."""
    existing = call_with_retry(get_product_row_by_nykaa_url, parsed["nykaa_url"])

    # Node 9 — PRODUCT EXISTS?  Object.keys($json).length > 0
    if len(existing.keys()) > 0:
        updated_rows = call_with_retry(
            update_product_by_nykaa_url,
            parsed["nykaa_url"],
            _write_fields(parsed),
        )
        update_row = updated_rows[0] if updated_rows else None
        return ProductWriteResult(
            action="updated",
            insert_executed=False,
            check_product_exists=existing,
            insert_product=None,
            update_product=_require_product_row(update_row, "update"),
        )

    inserted_rows = _insert_product_with_retry(
        {**_write_fields(parsed), "nykaa_url": parsed["nykaa_url"]}
    )
    insert_row = inserted_rows[0] if inserted_rows else None
    return ProductWriteResult(
        action="inserted",
        insert_executed=True,
        check_product_exists=existing,
        insert_product=_require_product_row(insert_row, "insert"),
        update_product=None,
    )
