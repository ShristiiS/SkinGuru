import asyncio
import logging

from config import (
    FUNCTIONAL_CATEGORY_TIMEOUT_SECONDS,
    FUNCTIONAL_INTER_BATCH_WAIT_SECONDS,
)
from orchestrator.llm import log_batch_progress, prepare_llm_batches
from orchestrator.n8n import post_n8n_webhook
from tracing import traced

logger = logging.getLogger(__name__)

FUNCTIONAL_CATEGORY_PATH = "functional-category-on-demand"


@traced("get_ids_for_functional")
def get_ids_for_functional(parsed_items: list[dict]) -> dict:
    """Node 22 — Get IDs for Functional 2. Original INPUT PARSER list only.

    PORT DECISION #1: this live node is the source for retry ID lookup,
    not the dead Get IDs for Functional2original node.
    """
    return {
        "ids": [
            {"name": item["canonical_name"], "id": item["ingredient_id"]}
            for item in parsed_items
        ]
    }


@traced("expand_functional_items")
def expand_functional_items(batch: dict) -> list[dict]:
    """Node 25 — {ingredient_id, ingredient_name, batch_number}."""
    return [
        {
            "ingredient_id": item["id"],
            "ingredient_name": item["name"],
            "batch_number": batch["batch_number"],
        }
        for item in batch["ingredients"]
    ]


@traced("handle_functional_errors")
def handle_functional_errors(
    ingredient_name: str, ingredient_id, batch_number: int, response
) -> dict:
    """Node 27 — same as LLM handle, plus ingredient_id for retry."""
    if not isinstance(response, dict):
        response = {}
    if response.get("error") or not response.get("status"):
        return {
            "ingredient_name": ingredient_name,
            "ingredient_id": ingredient_id,
            "status": "failed",
            "error": response.get("error") or "Unknown error",
            "batch_number": batch_number,
        }
    return {
        "ingredient_name": ingredient_name,
        "ingredient_id": ingredient_id,
        "status": response.get("status") or "completed",
        "batch_number": batch_number,
    }


@traced("prepare_functional_retry")
def prepare_functional_retry(progress: dict, all_ids: list[dict]) -> list[dict]:
    """Node 30 — >3 skip; else look up failed names on live node 22 ids."""
    failed_names = progress.get("failed_ingredients") or []
    if len(failed_names) > 3:
        logger.warning(
            "Skipping Functional retry: %s failures in batch (more than 3)",
            len(failed_names),
        )
        return []
    failed_set = set(failed_names)
    return [
        {
            "ingredient_id": item["id"],
            "ingredient_name": item["name"],
            "is_retry": True,
        }
        for item in all_ids
        if item["name"] in failed_set
    ]


@traced("log_functional_retry")
def log_functional_retry(handled: list[dict]) -> dict:
    """Node 32 — simpler retry log: !error vs rest. Not the LLM Batch logger."""
    successful = sum(1 for item in handled if not item.get("error"))
    still_failed = len(handled) - successful
    logger.info(
        "Functional retry: retry_successful=%s retry_failed=%s",
        successful,
        still_failed,
    )
    return {"retry_successful": successful, "retry_failed": still_failed}


def _response_json(response) -> dict:
    try:
        payload = response.json()
    except Exception:
        return {}
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    return payload


@traced("call_functional")
async def _call_functional(body: dict) -> dict:
    try:
        response = await post_n8n_webhook(
            FUNCTIONAL_CATEGORY_PATH,
            body,
            FUNCTIONAL_CATEGORY_TIMEOUT_SECONDS,
        )
        return _response_json(response)
    except Exception as exc:
        return {"error": str(exc)}


@traced("functional_initial_parallel")
async def _call_initial_parallel(expanded: list[dict]) -> list[dict]:
    """Node 26 — body is {ingredient_id, ingredient_name}."""
    raws = await asyncio.gather(
        *[
            _call_functional(
                {
                    "ingredient_id": item["ingredient_id"],
                    "ingredient_name": item["ingredient_name"],
                }
            )
            for item in expanded
        ]
    )
    return [
        handle_functional_errors(
            item["ingredient_name"],
            item["ingredient_id"],
            item["batch_number"],
            raw,
        )
        for item, raw in zip(expanded, raws)
    ]


@traced("functional_retry_parallel")
async def _call_retry_parallel(retry_items: list[dict], batch_number: int) -> list[dict]:
    """Node 31 — body is {ingredient_id} only."""
    raws = await asyncio.gather(
        *[
            _call_functional({"ingredient_id": item["ingredient_id"]})
            for item in retry_items
        ]
    )
    return [
        handle_functional_errors(
            item["ingredient_name"],
            item["ingredient_id"],
            batch_number,
            raw,
        )
        for item, raw in zip(retry_items, raws)
    ]


@traced("run_functional_category")
async def run_functional_category(parsed_items: list[dict]) -> dict:
    """Nodes 22–36. Batches of 5 from the original INPUT PARSER list."""
    ids_payload = get_ids_for_functional(parsed_items)
    all_ids = ids_payload["ids"]
    batches = prepare_llm_batches(all_ids)
    batch_summaries = []

    for batch in batches:
        expanded = expand_functional_items(batch)
        handled = await _call_initial_parallel(expanded)
        progress = log_batch_progress(handled)

        retry_items = []
        retry_log = None
        if progress and progress["failed"] > 0:
            retry_items = prepare_functional_retry(progress, all_ids)
            if retry_items:
                retried = await _call_retry_parallel(
                    retry_items, progress["batch_number"]
                )
                retry_log = log_functional_retry(retried)

        batch_summaries.append(
            {
                "batch_number": batch["batch_number"],
                "progress": progress,
                "retried": bool(retry_items),
                "retry_log": retry_log,
            }
        )
        await asyncio.sleep(FUNCTIONAL_INTER_BATCH_WAIT_SECONDS)

    return {
        "functional_total_batches": len(batches),
        "functional_batches": batch_summaries,
    }
