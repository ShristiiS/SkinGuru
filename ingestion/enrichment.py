import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from clients.supabase import check_enrichment_status, mark_product_processed
from config import (
    FLOW1_CALL_RETRY_ATTEMPTS,
    FLOW1_CALL_RETRY_WAIT_SECONDS,
    ORCHESTRATOR_TIMEOUT_SECONDS,
    ORCHESTRATOR_WEBHOOK_URL,
    PMC_PUBCHEM_TIMEOUT_SECONDS,
    PMC_WEBHOOK_URL,
    PUBCHEM_WEBHOOK_URL,
)
from ingestion.product_record import record_orchestrator_not_processed
from ingestion.products import ProductWriteResult
from precompute.call_retry import call_with_retry, is_retryable_call
from tracing import (
    PRODUCT_URL_HEADER,
    RUN_ID_HEADER,
    current_product_url,
    current_run_id,
    record_http,
    trace_step,
    traced,
)

WEBHOOK_TIMEOUT_REASON = "timed out"


def call_long_webhook_with_retry(fn, /, *args, **kwargs):
    """3 tries, 5s. Retry network/429/5xx. Never timeout (job may still be running)."""
    last_error: BaseException | None = None
    attempts = FLOW1_CALL_RETRY_ATTEMPTS
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except httpx.TimeoutException:
            raise RuntimeError(WEBHOOK_TIMEOUT_REASON) from None
        except Exception as exc:
            last_error = exc
            if not is_retryable_call(exc) or attempt == attempts - 1:
                raise
            time.sleep(FLOW1_CALL_RETRY_WAIT_SECONDS)
    raise last_error


@traced("get_matched_ingredient_ids")
def get_matched_ingredient_ids(stored_rows: list[dict], written: ProductWriteResult) -> dict:
    """Node 29 — unique ingredient_ids; product_id via try Insert else CHECK."""
    ingredient_ids = list(
        dict.fromkeys(row["ingredient_id"] for row in stored_rows)
    )
    return {
        "product_id": written.product_id_try_insert_else_check(),
        "ingredient_ids": ingredient_ids,
        "total_matched": len(ingredient_ids),
    }


@traced("categorize_enrichment_status")
def categorize_enrichment_status(rpc_rows: list[dict], product_id) -> dict:
    """Node 31 — split RPC rows using only the fields this node reads."""
    needs_enrichment = []
    fully_enriched = []
    for row in rpc_rows:
        if row.get("needs_enrichment"):
            needs_enrichment.append(
                {
                    "ingredient_id": row.get("ingredient_id"),
                    "canonical_name": row.get("canonical_name"),
                    "llm_enriched": row.get("llm_enriched"),
                    "concerns_processed": row.get("concerns_processed"),
                    "functional_processed": row.get("functional_processed"),
                }
            )
        else:
            fully_enriched.append(
                {
                    "ingredient_id": row.get("ingredient_id"),
                    "canonical_name": row.get("canonical_name"),
                }
            )
    return {
        "product_id": product_id,
        "needs_enrichment": needs_enrichment,
        "fully_enriched": fully_enriched,
        "all_enriched": len(needs_enrichment) == 0,
    }


def _post_json(url: str, body, timeout_seconds: float):
    step_name = (
        "europe_pmc_on_demand" if url == PMC_WEBHOOK_URL else "pubchem_on_demand"
    )
    with trace_step(step_name):
        response = httpx.post(
            url,
            json=body,
            timeout=timeout_seconds,
        )
        record_http(
            response.status_code,
            url=url,
            method="POST",
            request_body=body,
            response_body=response.text,
        )
        response.raise_for_status()
        return response


def _post_json_retry(url: str, body, timeout_seconds: float):
    return call_long_webhook_with_retry(_post_json, url, body, timeout_seconds)


@traced("call_pmc_and_pubchem")
def _call_pmc_and_pubchem(needs_enrichment: list) -> list:
    """Node 32 false: PMC + PubChem in parallel, body = JSON.stringify(needs_enrichment)."""
    urls = (PMC_WEBHOOK_URL, PUBCHEM_WEBHOOK_URL)
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                contextvars.copy_context().run,
                _post_json_retry,
                url,
                needs_enrichment,
                PMC_PUBCHEM_TIMEOUT_SECONDS,
            )
            for url in urls
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


@traced("recommendation_orchestrator_http")
def _call_recommendation_orchestrator(needs_enrichment: list):
    """Node 32 false: body = Categorize Enrichment Status.needs_enrichment.

    Chunk F: HTTP POST to this service's POST /recommendation-orchestrator
    (same process, new request). Not n8n, and not asyncio.run().

    Safe only while POST /ingestion/run is a sync def (FastAPI threadpool):
    that thread blocks on httpx.post while uvicorn serves the orchestrator on
    the event loop. If /ingestion/run becomes async def, this hop must switch
    to async httpx or the self-POST can deadlock (event loop blocked, second
    request never accepted).
    """
    headers = {}
    run_id = current_run_id()
    product_url = current_product_url()
    if run_id:
        headers[RUN_ID_HEADER] = run_id
    if product_url:
        headers[PRODUCT_URL_HEADER] = product_url
    def _post_orchestrator():
        response = httpx.post(
            ORCHESTRATOR_WEBHOOK_URL,
            json=needs_enrichment,
            headers=headers,
            timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
        )
        record_http(
            response.status_code,
            url=ORCHESTRATOR_WEBHOOK_URL,
            method="POST",
            request_body=needs_enrichment,
            response_body=response.text,
        )
        response.raise_for_status()
        return response

    return call_long_webhook_with_retry(_post_orchestrator).json()


def orchestrator_not_processed_items(response) -> list[dict]:
    if not isinstance(response, dict):
        return []
    items = []
    for row in response.get("not_processed") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("ingredient_name") or row.get("name")
        reason = row.get("error") or row.get("reason") or ""
        if name:
            items.append({"name": name, "reason": str(reason)})
    return items


def catalog_mark_allowed(
    subflow1_not_processed, orchestrator_response=None
) -> tuple[bool, str | None]:
    """True only when Sub-flow 1 and Orchestrator have no NOT PROCESSED leftovers."""
    bits = []
    seen = []
    for item in subflow1_not_processed or []:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason")
        if reason and reason not in seen:
            seen.append(reason)
            bits.append(reason)
    if orchestrator_not_processed_items(orchestrator_response):
        bits.append("orchestrator not_processed")
    if bits:
        return False, "; ".join(bits)
    return True, None


@traced("apply_enrichment_branch")
def apply_enrichment_branch(
    stored_rows: list[dict],
    written: ProductWriteResult,
    subflow1_not_processed=None,
) -> dict:
    """Nodes 29–33. True → mark processed. False → PMC + PubChem HTTP, then orchestrator HTTP.

    Do not PATCH catalog_processed if Sub-flow 1 or Orchestrator has NOT PROCESSED.
    """
    matched = get_matched_ingredient_ids(stored_rows, written)
    rpc_rows = call_with_retry(check_enrichment_status, matched["ingredient_ids"])
    categorized = categorize_enrichment_status(rpc_rows, matched["product_id"])
    subflow1_np = list(subflow1_not_processed or [])

    if categorized["all_enriched"] is True:
        allowed, why = catalog_mark_allowed(subflow1_np, None)
        if allowed:
            call_with_retry(
                mark_product_processed, written.product_id_try_insert_else_check()
            )
        return {
            **categorized,
            "catalog_processed": allowed,
            "catalog_processed_reason": why,
            "subflow2_triggered": False,
        }

    _call_pmc_and_pubchem(categorized["needs_enrichment"])
    orchestrator_response = _call_recommendation_orchestrator(
        categorized["needs_enrichment"]
    )
    for item in orchestrator_not_processed_items(orchestrator_response):
        record_orchestrator_not_processed(item["name"], item["reason"])
    allowed, why = catalog_mark_allowed(subflow1_np, orchestrator_response)
    if allowed:
        call_with_retry(
            mark_product_processed, written.product_id_try_insert_else_check()
        )
    return {
        **categorized,
        "catalog_processed": allowed,
        "catalog_processed_reason": why,
        "subflow2_triggered": True,
        "orchestrator_response": orchestrator_response,
    }
