import asyncio
import logging
import math

from config import (
    LLM_BATCH_SIZE,
    LLM_ENRICHMENT_TIMEOUT_SECONDS,
    LLM_INTER_BATCH_WAIT_SECONDS,
)
from orchestrator.n8n import post_n8n_webhook
from tracing import traced

logger = logging.getLogger(__name__)

LLM_ENRICHMENT_PATH = "llm-enrichment-on-demand"


@traced("prepare_llm_batches")
def prepare_llm_batches(ingredient_ids: list[dict]) -> list[dict]:
    """Node 7 — chunk ingredient_ids into batches of 5."""
    total = len(ingredient_ids)
    total_batches = math.ceil(total / LLM_BATCH_SIZE) if total else 0
    batches = []
    for index in range(0, total, LLM_BATCH_SIZE):
        chunk = ingredient_ids[index : index + LLM_BATCH_SIZE]
        batches.append(
            {
                "batch_number": index // LLM_BATCH_SIZE + 1,
                "total_batches": total_batches,
                "ingredients": chunk,
            }
        )
    return batches


@traced("expand_to_items")
def expand_to_items(batch: dict) -> list[dict]:
    """Node 9 — un-chunk one batch into per-ingredient items."""
    return [
        {
            "ingredient_name": item["name"],
            "ingredient_id": item["id"],
            "batch_number": batch["batch_number"],
            "total_batches": batch["total_batches"],
        }
        for item in batch["ingredients"]
    ]


@traced("handle_llm_errors")
def handle_llm_errors(ingredient_name: str, batch_number: int, response) -> dict:
    """Node 11 — error field or missing status → failed."""
    if not isinstance(response, dict):
        response = {}
    if response.get("error") or not response.get("status"):
        return {
            "ingredient_name": ingredient_name,
            "status": "failed",
            "error": response.get("error") or "Unknown error",
            "batch_number": batch_number,
        }
    return {
        "ingredient_name": ingredient_name,
        "status": response.get("status") or "completed",
        "batch_number": batch_number,
    }


@traced("log_batch_progress")
def log_batch_progress(handled: list[dict]) -> dict | None:
    """Nodes 12 and 16 (identical). PORT DECISION #7: skip entirely if zero items."""
    if not handled:
        return None

    successful = 0
    failed_ingredients = []
    for item in handled:
        has_error = bool(item.get("error"))
        status = item.get("status")
        if status == "completed" or not has_error:
            successful += 1
        if has_error or status == "failed":
            failed_ingredients.append(item.get("ingredient_name"))

    total = len(handled)
    failed = total - successful
    batch_number = handled[0].get("batch_number")
    logger.info(
        "LLM Batch %s: successful=%s failed=%s failed_ingredients=%s",
        batch_number,
        successful,
        failed,
        failed_ingredients,
    )
    return {
        "batch_number": batch_number,
        "successful": successful,
        "failed": failed,
        "failed_ingredients": failed_ingredients,
    }


@traced("prepare_llm_retry")
def prepare_llm_retry(progress: dict) -> list[dict]:
    """Node 14 — retry only if 1–3 failed; >3 skip (return [])."""
    failed_ingredients = progress.get("failed_ingredients") or []
    if len(failed_ingredients) > 3:
        logger.warning(
            "Skipping LLM retry: %s failures in batch (more than 3)",
            len(failed_ingredients),
        )
        return []
    return [
        {
            "ingredient_name": name,
            "is_retry": True,
            "retry_attempt": 1,
        }
        for name in failed_ingredients
    ]


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


@traced("call_llm_enrichment")
async def _call_llm_enrichment(ingredient_name: str) -> dict:
    """Nodes 10 / 15 — body is {ingredient_name} only."""
    try:
        response = await post_n8n_webhook(
            LLM_ENRICHMENT_PATH,
            {"ingredient_name": ingredient_name},
            LLM_ENRICHMENT_TIMEOUT_SECONDS,
        )
        return _response_json(response)
    except Exception as exc:
        return {"error": str(exc)}


@traced("llm_batch_parallel")
async def _call_batch_parallel(names_and_batches: list[tuple[str, int]]) -> list[dict]:
    tasks = [
        _call_llm_enrichment(name) for name, _batch_number in names_and_batches
    ]
    raw_responses = await asyncio.gather(*tasks)
    return [
        handle_llm_errors(name, batch_number, raw)
        for (name, batch_number), raw in zip(names_and_batches, raw_responses)
    ]


@traced("run_llm_enrichment")
async def run_llm_enrichment(worklist: dict) -> dict:
    """Nodes 7–20. Sequential batches of up to 5 parallel name-only n8n calls."""
    batches = prepare_llm_batches(worklist["ingredient_ids"])
    batch_summaries = []

    for batch in batches:
        expanded = expand_to_items(batch)
        handled = await _call_batch_parallel(
            [(item["ingredient_name"], item["batch_number"]) for item in expanded]
        )
        progress = log_batch_progress(handled)

        retry_items = []
        retry_progress = None
        if progress and progress["failed"] > 0:
            retry_items = prepare_llm_retry(progress)
            if retry_items:
                retried = await _call_batch_parallel(
                    [
                        (item["ingredient_name"], progress["batch_number"])
                        for item in retry_items
                    ]
                )
                retry_progress = log_batch_progress(retried)
            else:
                # PORT DECISION #7: zero retry items → skip Log LLM Retry.
                retry_progress = None

        batch_summaries.append(
            {
                "batch_number": batch["batch_number"],
                "progress": progress,
                "retried": bool(retry_items),
                "retry_progress": retry_progress,
            }
        )
        # Node 18 Wait — 1s default; not specified in the n8n export.
        await asyncio.sleep(LLM_INTER_BATCH_WAIT_SECONDS)

    return {
        "llm_total_batches": len(batches),
        "llm_batches": batch_summaries,
    }
