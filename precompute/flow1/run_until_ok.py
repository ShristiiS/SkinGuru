from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from precompute.flow1.agent import AgentRunResult
from precompute.flow1.product_record import record_rerun, record_step
from precompute.llm_call import format_llm_feedback
from tracing import trace_step
from tracing.debug import to_jsonable

T = TypeVar("T")

AGENT_MAX_RUNS = 3


def _debug_output(result):
    if isinstance(result, AgentRunResult):
        return {
            "text": result.text,
            "max_calls_hit": result.max_calls_hit,
            "tool_log": to_jsonable(result.tool_log),
        }
    return to_jsonable(result)


def _feedback_from_error(exc: BaseException) -> str:
    problems = str(exc).split("\n")
    return format_llm_feedback(problems)


def run_until_ok(
    fn: Callable[..., T],
    *,
    step: str,
    max_runs: int = AGENT_MAX_RUNS,
) -> T:
    """Run fn up to max_runs times. Rerun on exception or validation raise."""
    last_error: BaseException | None = None
    feedback = None
    for run in range(1, max_runs + 1):
        try:
            with trace_step(f"{step}_run") as fields:
                fields["debug_input"] = {"run": run}
                if feedback:
                    fields["debug_input"]["feedback"] = feedback
                result = fn(feedback)
                fields["debug_output"] = _debug_output(result)
                return result
        except Exception as exc:
            last_error = exc
            record_rerun(step, run, str(exc))
            feedback = _feedback_from_error(exc)
            if run == max_runs:
                record_step(step, "FAILED", f"run {run} of {max_runs}: {exc}")
                raise
    raise last_error
