from __future__ import annotations

from precompute.flow1.product_record import (
    STATUS_STEPS,
    ProductRecord,
    product_status,
)
from tracing.writer import write_status_block

_MAX_RUNS = 3
_LLM_LABELS = {
    "llm1": "LLM1",
    "llm2": "LLM2",
    "llm3": "LLM3",
    "llm4": "LLM4",
}
_AGENT_STEPS = frozenset(
    {
        "synergy_reasoning",
        "concern",
        "interaction_builder",
        "safety",
        "formulation",
    }
)
_STATUS_PAD = {"OK": 6, "FAILED": 2, "SKIPPED": 2}


def _step_status(record: ProductRecord, name: str) -> str:
    return (record.steps.get(name) or {}).get("status") or "SKIPPED"


def _step_reason(record: ProductRecord, name: str):
    return (record.steps.get(name) or {}).get("reason")


def _reruns_for(record: ProductRecord, step: str) -> list:
    return [entry for entry in record.reruns if entry.get("step") == step]


def _ok_retry_note(reruns: list, label: str | None = None) -> str | None:
    if not reruns:
        return None
    ordered = sorted(reruns, key=lambda entry: entry.get("run") or 0)
    last_fail = ordered[-1].get("run") or 0
    success_run = last_fail + 1
    parts = [
        f"run {entry.get('run')}: {entry.get('reason')}" for entry in ordered
    ]
    head = f"run {success_run} of {_MAX_RUNS}"
    if label:
        head = f"{label} {head}"
    return f"({head} — {'; '.join(parts)})"


def _estimator_extra(record: ProductRecord, status: str) -> str | None:
    reason = _step_reason(record, "estimator")
    if status == "SKIPPED" and reason:
        return reason
    if status == "OK" and reason == "already existed":
        return "(already existed)"
    notes = []
    for step, label in _LLM_LABELS.items():
        reruns = _reruns_for(record, step)
        if not reruns:
            continue
        if status == "OK":
            note = _ok_retry_note(reruns, label)
            if note:
                notes.append(note[1:-1])
        else:
            last = sorted(reruns, key=lambda entry: entry.get("run") or 0)[-1]
            notes.append(
                f"{label} run {last.get('run')} of {_MAX_RUNS}: {last.get('reason')}"
            )
    if status == "OK" and reason:
        if notes:
            return "(" + reason + "; " + "; ".join(notes) + ")"
        return f"({reason})"
    if notes:
        if status == "OK":
            return "(" + "; ".join(notes) + ")"
        return "; ".join(notes)
    if status == "FAILED" and reason:
        return reason
    return None


def _serpapi_extra(record: ProductRecord, status: str) -> str | None:
    bits = []
    reason = _step_reason(record, "serpapi")
    if status in ("FAILED", "SKIPPED") and reason:
        bits.append(reason)
    if record.not_processed:
        listed = "; ".join(
            f"{item.get('name')} ({item.get('reason')})"
            for item in record.not_processed
        )
        bits.append(f"NOT PROCESSED: {listed}")
    return " ".join(bits) if bits else None


def _agent_extra(record: ProductRecord, name: str, status: str) -> str | None:
    reason = _step_reason(record, name)
    if status == "FAILED" and reason:
        return reason
    if status == "SKIPPED" and reason:
        return reason
    if status == "OK":
        return _ok_retry_note(_reruns_for(record, name))
    return None


def _stores_estimator_simple(record: ProductRecord, name: str, status: str):
    reason = _step_reason(record, name)
    if status == "FAILED" and reason:
        return reason
    if status == "SKIPPED" and reason:
        return reason
    return None


def extra_for(record: ProductRecord, name: str, status: str) -> str | None:
    if name == "estimator":
        return _estimator_extra(record, status)
    if name == "serpapi":
        return _serpapi_extra(record, status)
    if name in _AGENT_STEPS:
        return _agent_extra(record, name, status)
    if name == "stores":
        return _stores_estimator_simple(record, name, status)
    if name in ("ingredients", "concentration_check"):
        return _stores_estimator_simple(record, name, status)
    return None


def format_status_line(name: str, status: str, extra: str | None = None) -> str:
    line = f"  {name:<23}{status}"
    if not extra:
        return line
    pad = _STATUS_PAD.get(status, 2)
    return line + (" " * pad) + extra


def format_status_block(record: ProductRecord) -> str:
    overall = product_status(record)
    lines = [f"PRODUCT {record.product_id} — {overall}"]
    for name in STATUS_STEPS:
        status = _step_status(record, name)
        extra = extra_for(record, name, status)
        lines.append(format_status_line(name, status, extra))
    return "\n".join(lines) + "\n"


def append_status_block(record: ProductRecord) -> None:
    write_status_block(format_status_block(record))
