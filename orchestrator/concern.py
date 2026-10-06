import asyncio
import logging

from clients.supabase import mark_concerns_analyzed
from config import (
    CONCERN_ANALYSIS_TIMEOUT_SECONDS,
    CONCERN_ANALYSIS_WAIT_SECONDS,
    ORCHESTRATOR_MAX_ATTEMPTS,
)
from orchestrator.n8n import post_n8n_webhook
from orchestrator.retry import (
    STEP_CONCERN,
    AttemptHadFailures,
    classify_reply,
    record_and_collect_not_processed,
    record_attempt,
    wait_before_retry,
)
from tracing import traced

logger = logging.getLogger(__name__)

CONCERN_PATH = "concern-analysis-on-demand"


def _trigger_body(ingredient_ids: list[dict]) -> list[dict]:
    return [
        {
            "canonical_name": item["name"],
            "ingredient_id": item["id"],
            "evidence_count": 0,
        }
        for item in ingredient_ids
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


@traced("run_concern_analysis")
async def run_concern_analysis(worklist: dict) -> dict:
    """One n8n trigger for the whole list; mark only after the trigger succeeds."""
    ingredient_ids = worklist["ingredient_ids"]
    body = _trigger_body(ingredient_ids)
    names = [item["name"] for item in ingredient_ids]
    last_error = "Unknown error"
    triggered = False

    for attempt in range(1, ORCHESTRATOR_MAX_ATTEMPTS + 1):
        await wait_before_retry(attempt)
        error = None
        try:
            response = await post_n8n_webhook(
                CONCERN_PATH,
                body,
                CONCERN_ANALYSIS_TIMEOUT_SECONDS,
            )
            body_failed, body_error = classify_reply(_response_json(response))
            if body_failed:
                error = body_error
        except Exception as exc:
            error = str(exc)

        if error is None:
            triggered = True
            try:
                record_attempt(
                    STEP_CONCERN,
                    attempt,
                    sent=names,
                    passed=names,
                    failed=[],
                )
            except AttemptHadFailures:
                pass
            break

        last_error = error
        failed = [{"ingredient_name": name, "error": error} for name in names]
        try:
            record_attempt(
                STEP_CONCERN,
                attempt,
                sent=names,
                passed=[],
                failed=failed,
            )
        except AttemptHadFailures:
            pass

    if not triggered:
        leftovers = [
            {
                "ingredient_name": name,
                "step": STEP_CONCERN,
                "error": last_error,
            }
            for name in names
        ]
        return {
            "concern_triggered": False,
            "concerns_marked": False,
            "not_processed": record_and_collect_not_processed(leftovers),
        }

    marked = True
    try:
        await asyncio.to_thread(mark_concerns_analyzed, names)
    except Exception as exc:
        marked = False
        logger.warning("Mark Concerns Analyzed failed: %s", exc)

    await asyncio.sleep(CONCERN_ANALYSIS_WAIT_SECONDS)
    return {
        "concern_triggered": True,
        "concerns_marked": marked,
        "not_processed": [],
    }
