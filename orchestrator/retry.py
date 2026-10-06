import asyncio

from config import ORCHESTRATOR_RETRY_WAIT_SECONDS
from tracing import trace_step

STEP_LLM = "llm_enrichment"
STEP_FUNCTIONAL = "functional_category"
STEP_CONCERN = "concern_analysis"

ATTEMPT_STEP = {
    STEP_LLM: "llm_enrichment_attempt",
    STEP_FUNCTIONAL: "functional_category_attempt",
    STEP_CONCERN: "concern_analysis_attempt",
}

_FAILED_STATUSES = {"failed", "error"}


class AttemptHadFailures(Exception):
    """Raised inside an attempt span so the debug block is status=error."""

    def __init__(self, failed: list[dict]):
        self.failed = failed
        super().__init__(f"{len(failed)} failed")


class NotProcessedRecorded(Exception):
    """Raised inside a not_processed span so the debug block is status=error."""


def classify_reply(response) -> tuple[bool, str]:
    """Return (is_failed, error_message). Passed replies have error_message ''."""
    if not isinstance(response, dict):
        return True, "Unknown error"
    error = response.get("error")
    status = response.get("status")
    if error or not status or str(status).lower() in _FAILED_STATUSES:
        if error:
            return True, str(error)
        if status:
            return True, str(status)
        return True, "Unknown error"
    return False, ""


async def wait_before_retry(attempt: int) -> None:
    """No wait before attempt 1; 5s before attempts 2–6."""
    if attempt <= 1:
        return
    await asyncio.sleep(ORCHESTRATOR_RETRY_WAIT_SECONDS)


def record_attempt(
    step: str,
    attempt: int,
    sent,
    passed: list,
    failed: list[dict],
) -> None:
    """Write one attempt block. Raises AttemptHadFailures if anyone failed."""
    with trace_step(ATTEMPT_STEP[step]) as fields:
        fields["debug_input"] = {"attempt": attempt, "sent": sent}
        fields["debug_output"] = {
            "attempt": attempt,
            "passed": passed,
            "failed": failed,
        }
        if failed:
            raise AttemptHadFailures(failed)


def record_not_processed(ingredient_name, step: str, error: str) -> None:
    """Write one NOT PROCESSED block with status=error. Always raises."""
    payload = {
        "ingredient_name": ingredient_name,
        "step": step,
        "error": error,
        "result": "NOT PROCESSED",
    }
    with trace_step("not_processed") as fields:
        fields["debug_input"] = payload
        fields["debug_output"] = payload
        raise NotProcessedRecorded(
            f"NOT PROCESSED ingredient_name={ingredient_name} "
            f"step={step} error={error}"
        )


def not_processed_item(ingredient_name, step: str, error: str) -> dict:
    return {
        "ingredient_name": ingredient_name,
        "step": step,
        "error": error,
    }


def record_and_collect_not_processed(
    leftovers: list[dict],
) -> list[dict]:
    """leftovers items: {ingredient_name, step, error}. Trace each; return the list."""
    collected = []
    for item in leftovers:
        collected.append(
            not_processed_item(item["ingredient_name"], item["step"], item["error"])
        )
        try:
            record_not_processed(
                item["ingredient_name"], item["step"], item["error"]
            )
        except NotProcessedRecorded:
            pass
    return collected
