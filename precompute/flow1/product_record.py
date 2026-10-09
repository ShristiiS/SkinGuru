from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_current: ContextVar["ProductRecord | None"] = ContextVar(
    "flow1_product_record", default=None
)
_records: list["ProductRecord"] = []


@dataclass
class ProductRecord:
    """Per-product reliability notes for the Chunk 6 status block."""

    product_id: object
    not_processed: list = field(default_factory=list)
    reruns: list = field(default_factory=list)
    steps: dict = field(default_factory=dict)
    synergy_parsed: object = None
    skipped: bool = False
    missing: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "not_processed": list(self.not_processed),
            "reruns": list(self.reruns),
            "steps": dict(self.steps),
        }


def current_product_record() -> ProductRecord | None:
    return _current.get()


def get_product_records() -> list[ProductRecord]:
    return list(_records)


def clear_product_records() -> None:
    _records.clear()


@contextmanager
def bind_product_record(product_id):
    record = ProductRecord(product_id=product_id)
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
    step_entry = record.steps.setdefault(step, {})
    step_entry.setdefault("runs", []).append(entry)


def record_not_processed(name, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    record.not_processed.append({"name": name, "reason": str(reason)})


def record_step(step: str, status: str, reason: str | None = None) -> None:
    record = current_product_record()
    if record is None:
        return
    entry = record.steps.setdefault(step, {})
    entry["status"] = status
    if reason is not None:
        entry["reason"] = reason


STATUS_STEPS = (
    "ingredients",
    "concentration_check",
    "estimator",
    "stores",
    "serpapi",
    "synergy_reasoning",
    "concern",
    "interaction_builder",
    "safety",
    "formulation",
)
_EARLY_FAIL_STEPS = frozenset(
    {"ingredients", "concentration_check", "estimator", "stores"}
)


def skip_from(step: str, reason: str) -> None:
    record = current_product_record()
    if record is None:
        return
    started = False
    for name in STATUS_STEPS:
        if name == step:
            started = True
        if not started:
            continue
        existing = record.steps.get(name, {}).get("status")
        if existing in ("OK", "FAILED", "SKIPPED"):
            continue
        record_step(name, "SKIPPED", reason)


def next_status_step(step: str) -> str | None:
    try:
        index = STATUS_STEPS.index(step)
    except ValueError:
        return None
    if index + 1 >= len(STATUS_STEPS):
        return None
    return STATUS_STEPS[index + 1]


def product_status(record: ProductRecord) -> str:
    steps = record.steps
    if any(
        (steps.get(name) or {}).get("status") == "FAILED"
        for name in _EARLY_FAIL_STEPS
    ):
        return "FAILED"
    if record.not_processed:
        return "PARTIAL"
    for name in STATUS_STEPS:
        status = (steps.get(name) or {}).get("status")
        if status in ("FAILED", "SKIPPED"):
            return "PARTIAL"
    return "OK"
