import asyncio

from config import (
    FUNCTIONAL_CATEGORY_TIMEOUT_SECONDS,
    FUNCTIONAL_INTER_BATCH_WAIT_SECONDS,
    ORCHESTRATOR_MAX_ATTEMPTS,
)
from orchestrator.enrichment_status import verify_and_resend
from orchestrator.llm import prepare_llm_batches
from orchestrator.n8n import post_n8n_webhook
from orchestrator.retry import (
    STEP_FUNCTIONAL,
    AttemptHadFailures,
    classify_reply,
    record_and_collect_not_processed,
    record_attempt,
    wait_before_retry,
)
from tracing import traced

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
    """Error field, missing status, or failed/error status → failed."""
    failed, error = classify_reply(response)
    if failed:
        return {
            "ingredient_name": ingredient_name,
            "ingredient_id": ingredient_id,
            "status": "failed",
            "error": error,
            "batch_number": batch_number,
        }
    if not isinstance(response, dict):
        response = {}
    return {
        "ingredient_name": ingredient_name,
        "ingredient_id": ingredient_id,
        "status": response.get("status") or "completed",
        "batch_number": batch_number,
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


async def _resend_functional(chunk: list[dict]) -> list[dict]:
    retry_items = [
        {
            "ingredient_id": item["id"],
            "ingredient_name": item["name"],
        }
        for item in chunk
    ]
    return await _call_retry_parallel(retry_items, 1)


@traced("run_functional_category")
async def run_functional_category(parsed_items: list[dict]) -> dict:
    """Batches of 5; retry only remaining failures, up to 6 attempts."""
    ids_payload = get_ids_for_functional(parsed_items)
    all_ids = ids_payload["ids"]
    batches = prepare_llm_batches(all_ids)
    batch_summaries = []
    leftovers = []
    attempts_used = {item["name"]: 0 for item in all_ids}
    n8n_failed_names = set()

    for batch in batches:
        remaining = expand_functional_items(batch)
        attempt_logs = []
        last_error_by_name = {}

        for attempt in range(1, ORCHESTRATOR_MAX_ATTEMPTS + 1):
            if not remaining:
                break
            await wait_before_retry(attempt)
            if attempt == 1:
                handled = await _call_initial_parallel(remaining)
            else:
                handled = await _call_retry_parallel(
                    remaining, remaining[0]["batch_number"]
                )
            passed = []
            failed = []
            still = []
            for item, row in zip(remaining, handled):
                name = item["ingredient_name"]
                attempts_used[name] = attempts_used.get(name, 0) + 1
                if row.get("status") == "failed" or row.get("error"):
                    error = row.get("error") or "Unknown error"
                    failed.append({"ingredient_name": name, "error": error})
                    last_error_by_name[name] = error
                    still.append(item)
                else:
                    passed.append(name)
            try:
                record_attempt(
                    STEP_FUNCTIONAL,
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
            n8n_failed_names.add(name)
            leftovers.append(
                {
                    "ingredient_name": name,
                    "step": STEP_FUNCTIONAL,
                    "error": last_error_by_name.get(name) or "Unknown error",
                }
            )

        batch_summaries.append(
            {
                "batch_number": batch["batch_number"],
                "attempts": attempt_logs,
            }
        )
        await asyncio.sleep(FUNCTIONAL_INTER_BATCH_WAIT_SECONDS)

    leftovers.extend(
        await verify_and_resend(
            items=all_ids,
            attempts_used=attempts_used,
            n8n_failed_names=n8n_failed_names,
            flag="functional_processed",
            step=STEP_FUNCTIONAL,
            resend=_resend_functional,
        )
    )

    return {
        "functional_total_batches": len(batches),
        "functional_batches": batch_summaries,
        "not_processed": record_and_collect_not_processed(leftovers),
    }
