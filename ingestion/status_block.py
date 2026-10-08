from __future__ import annotations

from ingestion.match import JOIN_MISS_REASON
from ingestion.product_record import (
    STATUS_STEPS,
    ProductRecord,
    product_status,
)
from ingestion.store import STORE_MISSING_REASON
from tracing.writer import write_status_block

_MAX_RUNS = 3
_STATUS_PAD = {
    "OK": 6,
    "FAILED": 2,
    "SKIPPED": 2,
    "NOT SET": 2,
    "NOT PROCESSED": 2,
}


def _step_status(record: ProductRecord, name: str) -> str:
    return (record.steps.get(name) or {}).get("status") or "SKIPPED"


def _step_reason(record: ProductRecord, name: str):
    return (record.steps.get(name) or {}).get("reason")


def _reruns_for(record: ProductRecord, step: str) -> list:
    return [entry for entry in record.reruns if entry.get("step") == step]


def _ok_retry_note(reruns: list) -> str | None:
    if not reruns:
        return None
    ordered = sorted(reruns, key=lambda entry: entry.get("run") or 0)
    last_fail = ordered[-1].get("run") or 0
    success_run = last_fail + 1
    parts = [
        f"run {entry.get('run')}: {entry.get('reason')}" for entry in ordered
    ]
    return f"(run {success_run} of {_MAX_RUNS} — {'; '.join(parts)})"


def _failed_extra(record: ProductRecord, name: str) -> str | None:
    reruns = _reruns_for(record, name)
    reason = _step_reason(record, name)
    if reruns:
        last = sorted(reruns, key=lambda entry: entry.get("run") or 0)[-1]
        return f"run {last.get('run')} of {_MAX_RUNS}: {last.get('reason')}"
    return reason


def _np_list(items: list) -> str | None:
    if not items:
        return None
    listed = "; ".join(
        f"{item.get('name')} ({item.get('reason')})" for item in items
    )
    return listed


def _not_processed_for_step(record: ProductRecord, name: str) -> list:
    if name == "match_ingredients":
        return [
            item
            for item in record.not_processed
            if item.get("reason") == JOIN_MISS_REASON
        ]
    if name == "store_ingredients":
        return [
            item
            for item in record.not_processed
            if item.get("reason") == STORE_MISSING_REASON
        ]
    if name == "orchestrator":
        return list(record.orchestrator_not_processed)
    return []


def extra_for(record: ProductRecord, name: str, status: str) -> str | None:
    bits = []
    if status == "FAILED":
        extra = _failed_extra(record, name)
        if extra:
            bits.append(extra)
    elif status == "SKIPPED":
        reason = _step_reason(record, name)
        if reason:
            bits.append(reason)
    elif status == "NOT SET":
        reason = _step_reason(record, name)
        if reason:
            bits.append(reason)
    elif status == "OK":
        note = _ok_retry_note(_reruns_for(record, name))
        if note:
            bits.append(note)
        if name == "inci_normalizer" and record.inci_dropped:
            bits.append("dropped: " + ", ".join(str(n) for n in record.inci_dropped))
    elif status == "NOT PROCESSED":
        extra = _failed_extra(record, name)
        reruns = _reruns_for(record, name)
        if reruns and extra:
            bits.append(extra)

    np_extra = _np_list(_not_processed_for_step(record, name))
    if np_extra:
        if status == "NOT PROCESSED":
            bits.append(np_extra)
        else:
            bits.append(f"NOT PROCESSED: {np_extra}")
    return " ".join(bits) if bits else None


def format_status_line(name: str, status: str, extra: str | None = None) -> str:
    line = f"  {name:<23}{status}"
    if not extra:
        return line
    pad = _STATUS_PAD.get(status, 2)
    return line + (" " * pad) + extra


def format_status_block(record: ProductRecord) -> str:
    if record.skipped:
        return f"PRODUCT {record.product_url} — SKIPPED (already processed)\n"
    overall = product_status(record)
    lines = [f"PRODUCT {record.product_url} — {overall}"]
    for name in STATUS_STEPS:
        status = _step_status(record, name)
        extra = extra_for(record, name, status)
        lines.append(format_status_line(name, status, extra))
    return "\n".join(lines) + "\n"


def append_status_block(record: ProductRecord) -> None:
    write_status_block(format_status_block(record))
