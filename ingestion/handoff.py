import json

from clients.supabase import get_processed_product_ids_by_urls
from ingestion.catalog import PRODUCT_URLS
from precompute.call_retry import call_with_retry
from precompute.flow1.run import accept_precompute
from tracing import current_run_id, traced


@traced("get_all_product_ids")
def get_all_product_ids() -> dict:
    """Node 35 — re-query catalog URLs with catalog_processed=true. Not saved_product_ids."""
    product_ids = call_with_retry(get_processed_product_ids_by_urls, PRODUCT_URLS)
    return {"product_ids": product_ids}


@traced("call_precomputation")
def call_precomputation(product_ids: list) -> str:
    """Node 36 — start Python pre-compute in-process. String reply, do not JSON-parse."""
    reply = accept_precompute(product_ids, current_run_id())
    return json.dumps(reply)


@traced("post_loop_handoff")
def post_loop_handoff() -> dict:
    """Nodes 35–36. Runs once after the per-product loop."""
    fetched = get_all_product_ids()
    precompute_text = call_precomputation(fetched["product_ids"])
    return {
        "product_ids": fetched["product_ids"],
        "precompute_response": precompute_text,
    }
