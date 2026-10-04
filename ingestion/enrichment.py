import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from clients.supabase import check_enrichment_status, mark_product_processed
from config import (
    ORCHESTRATOR_TIMEOUT_SECONDS,
    ORCHESTRATOR_WEBHOOK_URL,
    PMC_PUBCHEM_TIMEOUT_SECONDS,
    PMC_WEBHOOK_URL,
    PUBCHEM_WEBHOOK_URL,
)
from ingestion.products import ProductWriteResult
from tracing import (
    PRODUCT_URL_HEADER,
    RUN_ID_HEADER,
    current_product_url,
    current_run_id,
    record_http,
    trace_step,
    traced,
)


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


@traced("call_pmc_and_pubchem")
def _call_pmc_and_pubchem(needs_enrichment: list) -> list:
    """Node 32 false: PMC + PubChem in parallel, body = JSON.stringify(needs_enrichment)."""
    urls = (PMC_WEBHOOK_URL, PUBCHEM_WEBHOOK_URL)
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                contextvars.copy_context().run,
                _post_json,
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
    return response.json()


@traced("apply_enrichment_branch")
def apply_enrichment_branch(stored_rows: list[dict], written: ProductWriteResult) -> dict:
    """Nodes 29–33. True → mark processed. False → PMC + PubChem HTTP, then orchestrator HTTP."""
    matched = get_matched_ingredient_ids(stored_rows, written)
    rpc_rows = check_enrichment_status(matched["ingredient_ids"])
    categorized = categorize_enrichment_status(rpc_rows, matched["product_id"])

    if categorized["all_enriched"] is True:
        mark_product_processed(written.product_id_try_insert_else_check())
        return {
            **categorized,
            "catalog_processed": True,
            "subflow2_triggered": False,
        }

    _call_pmc_and_pubchem(categorized["needs_enrichment"])
    orchestrator_response = _call_recommendation_orchestrator(
        categorized["needs_enrichment"]
    )
    mark_product_processed(written.product_id_try_insert_else_check())
    return {
        **categorized,
        "catalog_processed": True,
        "subflow2_triggered": True,
        "orchestrator_response": orchestrator_response,
    }
