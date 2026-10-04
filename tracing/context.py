from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

RUN_ID_HEADER = "X-Run-Id"
PRODUCT_URL_HEADER = "X-Product-Url"

_run_id: ContextVar[str | None] = ContextVar("trace_run_id", default=None)
_product_url: ContextVar[str | None] = ContextVar("trace_product_url", default=None)
_step_stack: ContextVar[tuple[str, ...]] = ContextVar("trace_step_stack", default=())
_span_fields: ContextVar[dict | None] = ContextVar("trace_span_fields", default=None)
_product_totals: ContextVar[_ProductTotals | None] = ContextVar(
    "trace_product_totals", default=None
)


@dataclass
class _ProductTotals:
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    costs: list = field(default_factory=list)


def current_run_id() -> str | None:
    return _run_id.get()


def current_product_url() -> str | None:
    return _product_url.get()


def current_parent_step() -> str | None:
    stack = _step_stack.get()
    return stack[-1] if stack else None


def current_span_fields() -> dict | None:
    return _span_fields.get()


def current_product_totals() -> _ProductTotals | None:
    return _product_totals.get()


@contextmanager
def start_run(run_id: str | None = None):
    value = (run_id or "").strip() or str(uuid.uuid4())
    token = _run_id.set(value)
    try:
        yield value
    finally:
        _run_id.reset(token)


@contextmanager
def bind_product(product_url: str | None):
    url_token = _product_url.set(product_url or None)
    totals_token = _product_totals.set(_ProductTotals())
    try:
        yield
    finally:
        _product_url.reset(url_token)
        _product_totals.reset(totals_token)


@contextmanager
def push_step(step_name: str):
    parent = current_parent_step()
    stack_token = _step_stack.set(_step_stack.get() + (step_name,))
    fields_token = _span_fields.set({})
    try:
        yield parent
    finally:
        _span_fields.reset(fields_token)
        _step_stack.reset(stack_token)
