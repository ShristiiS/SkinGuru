import asyncio

from orchestrator.concern import ConcernAnalysisError, run_concern_analysis
from orchestrator.functional import run_functional_category
from orchestrator.llm import run_llm_enrichment
from orchestrator.parser import _iso_timestamp, parse_and_prepare
from tracing import traced


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _task_exception(task: asyncio.Task):
    if not task.done():
        return None
    try:
        return task.exception()
    except asyncio.CancelledError:
        return None


def _raise_if_concern_failed(concern_task: asyncio.Task) -> None:
    exc = _task_exception(concern_task)
    if exc is None:
        return
    if isinstance(exc, ConcernAnalysisError):
        raise exc
    raise ConcernAnalysisError(str(exc)) from exc


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


@traced("run_orchestrator")
async def run_orchestrator(payload) -> dict:
    """PORT DECISIONS #5+#6+#2: concurrent branches, concern-failure cancel, completed response."""
    parsed_items, worklist = parse_and_prepare(payload)

    concern_task = asyncio.create_task(
        run_concern_analysis(worklist), name="concern-analysis"
    )
    llm_task = asyncio.create_task(
        run_llm_enrichment(worklist), name="llm-enrichment"
    )
    functional_task = None

    try:
        # Start Functional the moment LLM finishes; abort if Concern fails first.
        while not llm_task.done():
            unfinished = {t for t in (concern_task, llm_task) if not t.done()}
            await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            if concern_task.done():
                _raise_if_concern_failed(concern_task)
                if not llm_task.done():
                    await llm_task

        _raise_if_concern_failed(concern_task)
        llm_task.result()

        functional_task = asyncio.create_task(
            run_functional_category(parsed_items), name="functional-category"
        )

        pending = {t for t in (concern_task, functional_task) if not t.done()}
        while pending:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            _raise_if_concern_failed(concern_task)
            pending = {t for t in (concern_task, functional_task) if not t.done()}

        functional_task.result()
        concern_task.result()

    except ConcernAnalysisError:
        await _cancel_task(llm_task)
        await _cancel_task(functional_task)
        raise

    except BaseException:
        await _cancel_task(llm_task)
        await _cancel_task(functional_task)
        raise

    return {
        "status": "completed",
        "ingredients_processed": _ingredients_processed(worklist, parsed_items),
        "timestamp": _iso_timestamp(),
    }
