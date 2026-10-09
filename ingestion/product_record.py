from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_current: ContextVar["ProductRecord | None"] = ContextVar(
    "ingestion_product_record", default=None
)
_records: list["ProductRecord"] = []

STATUS_STEPS = (
    "skip_check",
    "scraper",
    "product_upsert",
    "inci_normalizer",
    "match_ingredients",
    "store_ingredients",
    "enrichment",
    "orchestrator",
    "catalog_processed",
)


@dataclass
class ProductRecord:
    """Per-URL Sub-flow 1 notes for the status block."""

    product_url: str
    skipped: bool = False
    not_processed: list = field(default_factory=list)
    orchestrator_not_processed: list = field(default_factory=list)
    reruns: list = field(default_factory=list)
    steps: dict = field(default_factory=dict)
    inci_dropped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def current_product_record() -> ProductRecord | None:
    return _current.get()


def get_product_records() -> list[ProductRecord]:
    return list(_records)


def clear_product_records() -> None:
    _records.clear()


@contextmanager
def bind_product_record(product_url: str):
    record = ProductRecord(product_url=product_url)
    _records.append(record)
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)


def record_rerun(step: str, run: int, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    entry = {"step": step, "run": run, "reason": str(reason)}
    record.reruns.append(entry)


def record_not_processed(name, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    record.not_processed.append({"name": name, "reason": str(reason)})


def record_orchestrator_not_processed(name, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    record.orchestrator_not_processed.append(
        {"name": name, "reason": str(reason)}
    )


def record_inci_dropped(names: list[str]) -> None:
    record = current_product_record()
    if record is None:
        return
    record.inci_dropped.extend(names)


def record_warning(message: str) -> None:
    record = current_product_record()
    if record is None:
        return
    record.warnings.append(str(message))


def record_step(step: str, status: str, reason: str | None = None) -> None:
    record = current_product_record()
    if record is None:
        return
    entry = record.steps.setdefault(step, {})
    entry["status"] = status
    if reason is not None:
        entry["reason"] = reason


def skip_from(step: str, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    started = False
    for name in STATUS_STEPS:
        if name == step:
            started = True
            continue
        if not started:
            continue
        if name == "catalog_processed":
            continue
        existing = record.steps.get(name, {}).get("status")
        if existing in ("OK", "FAILED", "SKIPPED", "NOT SET", "NOT PROCESSED"):
            continue
        record_step(name, "SKIPPED", reason)


def product_status(record: ProductRecord) -> str:
    if record.skipped:
        return "SKIPPED"
    steps = record.steps
    if any(
        (steps.get(name) or {}).get("status") == "FAILED"
        for name in STATUS_STEPS
        if name != "catalog_processed"
    ):
        return "FAILED"
    if record.not_processed or record.orchestrator_not_processed:
        return "PARTIAL"
    if (steps.get("catalog_processed") or {}).get("status") == "NOT SET":
        return "PARTIAL"
    if any(
        (steps.get(name) or {}).get("status") == "NOT PROCESSED"
        for name in STATUS_STEPS
    ):
        return "PARTIAL"
    return "OK"
