import asyncio
import logging
import math

from config import (
    LLM_BATCH_SIZE,
    LLM_ENRICHMENT_TIMEOUT_SECONDS,
    LLM_INTER_BATCH_WAIT_SECONDS,
    ORCHESTRATOR_MAX_ATTEMPTS,
)
from orchestrator.n8n import post_n8n_webhook
from orchestrator.retry import (
    STEP_LLM,
    AttemptHadFailures,
    classify_reply,
    record_and_collect_not_processed,
    record_attempt,
    wait_before_retry,
)
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
    """Error field, missing status, or failed/error status → failed."""
    failed, error = classify_reply(response)
    if failed:
        return {
            "ingredient_name": ingredient_name,
            "status": "failed",
            "error": error,
            "batch_number": batch_number,
        }
    if not isinstance(response, dict):
        response = {}
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
    """Sequential batches of up to 5; retry only remaining failures, up to 6 attempts."""
    batches = prepare_llm_batches(worklist["ingredient_ids"])
    batch_summaries = []
    leftovers = []

    for batch in batches:
        remaining = expand_to_items(batch)
        attempt_logs = []
        last_error_by_name = {}

        for attempt in range(1, ORCHESTRATOR_MAX_ATTEMPTS + 1):
            if not remaining:
                break
            await wait_before_retry(attempt)
            handled = await _call_batch_parallel(
                [(item["ingredient_name"], item["batch_number"]) for item in remaining]
            )
            passed = []
            failed = []
            still = []
            for item, row in zip(remaining, handled):
                name = item["ingredient_name"]
                if row.get("status") == "failed" or row.get("error"):
                    error = row.get("error") or "Unknown error"
                    failed.append({"ingredient_name": name, "error": error})
                    last_error_by_name[name] = error
                    still.append(item)
                else:
                    passed.append(name)
            try:
                record_attempt(
                    STEP_LLM,
                    attempt,
                    sent=[item["ingredient_name"] for item in remaining],
                    passed=passed,
                    failed=failed,
                )
            except AttemptHadFailures:
                pass
            remaining = still
            attempt_logs.append(
                {"attempt": attempt, "passed": passed, "failed": failed}
            )

        for item in remaining:
            name = item["ingredient_name"]
            leftovers.append(
                {
                    "ingredient_name": name,
                    "step": STEP_LLM,
                    "error": last_error_by_name.get(name) or "Unknown error",
                }
            )

        batch_summaries.append(
            {
                "batch_number": batch["batch_number"],
                "attempts": attempt_logs,
            }
        )
        # Node 18 Wait — 1s default; not specified in the n8n export.
        await asyncio.sleep(LLM_INTER_BATCH_WAIT_SECONDS)

    return {
        "llm_total_batches": len(batches),
        "llm_batches": batch_summaries,
        "not_processed": record_and_collect_not_processed(leftovers),
    }
