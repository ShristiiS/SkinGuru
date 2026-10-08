import asyncio

from orchestrator.concern import run_concern_analysis
from orchestrator.functional import run_functional_category
from orchestrator.llm import run_llm_enrichment
from orchestrator.parser import _iso_timestamp, parse_and_prepare
from orchestrator.status_block import format_status_block
from tracing import current_product_url, trace_step, traced
from tracing.writer import write_status_block


def _record_cancel_error(task: asyncio.Task, exc: BaseException) -> None:
    payload = {"task": task.get_name(), "error": str(exc)}
    with trace_step("cancel_task") as fields:
        fields["debug_input"] = payload
        fields["debug_output"] = payload
        raise RuntimeError(str(exc))


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        try:
            _record_cancel_error(task, exc)
        except Exception:
            pass


@traced("count_prepare_total")
def _ingredients_processed(worklist: dict, parsed_items: list[dict]):
    """PORT DECISION #2: Count & Prepare List total_count; on error, INPUT PARSER (undefined)."""
    try:
        return worklist["total_count"]
    except Exception:
        try:
            return parsed_items[0]["total_count"]
        except Exception:
            return None


def _merge_not_processed(*results: dict) -> list[dict]:
    merged = []
    for result in results:
        merged.extend(result.get("not_processed") or [])
    return merged


@traced("run_orchestrator")
async def run_orchestrator(payload) -> dict:
    """PORT DECISIONS #5+#2: concurrent branches; completed even when some ingredients fail."""
    parsed_items, worklist = parse_and_prepare(payload)

    concern_task = asyncio.create_task(
        run_concern_analysis(worklist), name="concern-analysis"
    )
    llm_task = asyncio.create_task(
        run_llm_enrichment(worklist), name="llm-enrichment"
    )
    functional_task = None

    try:
        while not llm_task.done():
            unfinished = {t for t in (concern_task, llm_task) if not t.done()}
            await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)

        llm_result = llm_task.result()

        functional_task = asyncio.create_task(
            run_functional_category(parsed_items), name="functional-category"
        )

        pending = {t for t in (concern_task, functional_task) if not t.done()}
        while pending:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            pending = {t for t in (concern_task, functional_task) if not t.done()}

        functional_result = functional_task.result()
        concern_result = concern_task.result()
    except BaseException:
        await _cancel_task(llm_task)
        await _cancel_task(functional_task)
        raise

    not_processed = _merge_not_processed(
        llm_result, functional_result, concern_result
    )
    write_status_block(
        format_status_block(
            parsed_items,
            not_processed,
            current_product_url(),
        )
    )
    return {
        "status": "completed",
        "ingredients_processed": _ingredients_processed(worklist, parsed_items),
        "timestamp": _iso_timestamp(),
        "not_processed": not_processed,
    }
