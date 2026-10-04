import httpx

from clients.supabase import get_processed_product_ids_by_urls
from config import PRECOMPUTE_TIMEOUT_SECONDS, PRECOMPUTE_WEBHOOK_URL
from ingestion.catalog import PRODUCT_URLS
from tracing import record_http, traced


@traced("get_all_product_ids")
def get_all_product_ids() -> dict:
    """Node 35 — re-query catalog URLs with catalog_processed=true. Not saved_product_ids."""
    product_ids = get_processed_product_ids_by_urls(PRODUCT_URLS)
    return {"product_ids": product_ids}


@traced("call_precomputation")
def call_precomputation(product_ids: list) -> str:
    """Node 36 — POST {product_ids} to precompute. Plain text response, do not JSON-parse."""
    response = httpx.post(
        PRECOMPUTE_WEBHOOK_URL,
        json={"product_ids": product_ids},
        timeout=PRECOMPUTE_TIMEOUT_SECONDS,
    )
    record_http(
        response.status_code,
        url=PRECOMPUTE_WEBHOOK_URL,
        method="POST",
        request_body={"product_ids": product_ids},
        response_body=response.text,
    )
    response.raise_for_status()
    return response.text


@traced("post_loop_handoff")
def post_loop_handoff() -> dict:
    """Nodes 35–36. Runs once after the per-product loop."""
    fetched = get_all_product_ids()
    precompute_text = call_precomputation(fetched["product_ids"])
    return {
        "product_ids": fetched["product_ids"],
        "precompute_response": precompute_text,
    }
