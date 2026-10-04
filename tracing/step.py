from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from config import PRICING
from tracing.context import (
    current_product_totals,
    current_product_url,
    current_run_id,
    current_span_fields,
    push_step,
)
from tracing.debug import (
    bound_input,
    error_payload,
    format_step_block,
    request_body_of,
    response_body_of,
    to_jsonable,
)
from tracing.writer import redact_secrets, write_debug_block, write_product_totals, write_trace

logger = logging.getLogger("tracing")


def _iso(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


def _rates_for(model: str | None) -> dict:
    if not model:
        return {}
    if model in PRICING:
        return PRICING[model] or {}
    for key, rates in PRICING.items():
        if key and model.startswith(str(key)):
            return rates or {}
    return {}


def compute_cost(model, prompt_tokens, completion_tokens):
    rates = _rates_for(model)
    prompt_rate = rates.get("prompt_per_1m")
    completion_rate = rates.get("completion_per_1m")
    if (
        prompt_rate is None
        or completion_rate is None
        or prompt_tokens is None
        or completion_tokens is None
    ):
        return None
    return (prompt_tokens / 1_000_000) * prompt_rate + (
        completion_tokens / 1_000_000
    ) * completion_rate


def record_http(
    status_code,
    url: str | None = None,
    method: str | None = None,
    request_body=None,
    response_body=None,
    **_extra,
) -> None:
    try:
        fields = current_span_fields()
        if fields is None:
            return
        fields["http_status"] = status_code
        if url:
            fields["url"] = url
        fields.setdefault("debug_http", []).append(
            {
                "method": method,
                "url": url,
                "request_body": to_jsonable(request_body),
                "response_status": status_code,
                "response_body": to_jsonable(response_body),
            }
        )
    except Exception:
        pass


def record_http_response(response, url: str | None = None, request_body=None) -> None:
    """Record method/URL/bodies from an httpx response. Never raises."""
    try:
        request = getattr(response, "request", None)
        body = request_body
        if body is None:
            body = request_body_of(request)
        record_http(
            getattr(response, "status_code", None),
            url=url or (str(request.url) if request is not None else None),
            method=getattr(request, "method", None) if request is not None else None,
            request_body=body,
            response_body=response_body_of(response),
        )
    except Exception:
        try:
            record_http(getattr(response, "status_code", None), url=url)
        except Exception:
            pass


def record_warnings(warnings) -> None:
    fields = current_span_fields()
    if fields is None:
        return
    fields["warnings"] = warnings


def record_llm_usage(model, usage: dict | None = None) -> None:
    fields = current_span_fields()
    if fields is None:
        return
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    fields["model"] = model
    fields["prompt_tokens"] = prompt_tokens
    fields["completion_tokens"] = completion_tokens
    fields["total_tokens"] = total_tokens
    fields["cost_usd"] = compute_cost(model, prompt_tokens, completion_tokens)


def record_llm_exchange(message: dict | None = None) -> None:
    """Record reply text and tool calls. Never records prompts or system messages."""
    try:
        fields = current_span_fields()
        if fields is None:
            return
        message = message or {}
        fields["llm_reply"] = message.get("content") or ""
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            parsed = arguments
            if isinstance(arguments, str):
                try:
                    import json

                    parsed = json.loads(arguments)
                except Exception:
                    parsed = arguments
            tool_calls.append(
                {
                    "name": function.get("name"),
                    "arguments": to_jsonable(parsed),
                    "id": call.get("id"),
                }
            )
        fields["llm_tool_calls"] = tool_calls
    except Exception:
        pass


def record_llm_retry(attempt, error) -> None:
    try:
        fields = current_span_fields()
        if fields is None:
            return
        fields.setdefault("llm_retries", []).append(
            {
                "attempt": attempt + 1,
                "error": redact_secrets(error),
            }
        )
    except Exception:
        pass


def record_prompt_values(values) -> None:
    try:
        fields = current_span_fields()
        if fields is None:
            return
        fields["prompt_values"] = to_jsonable(values)
    except Exception:
        pass


def _add_product_totals(step_name: str, latency_ms: int, fields: dict) -> None:
    totals = current_product_totals()
    if totals is None:
        return
    if step_name == "process_one_product":
        totals.latency_ms = latency_ms
    prompt_tokens = fields.get("prompt_tokens")
    completion_tokens = fields.get("completion_tokens")
    total_tokens = fields.get("total_tokens")
    if prompt_tokens:
        totals.prompt_tokens += prompt_tokens
    if completion_tokens:
        totals.completion_tokens += completion_tokens
    if total_tokens:
        totals.total_tokens += total_tokens
    cost = fields.get("cost_usd")
    if cost is not None:
        totals.costs.append(cost)


def _print_step_line(step_name: str, latency_ms: int, status: str, fields: dict) -> None:
    tokens = fields.get("total_tokens")
    prompt_tokens = fields.get("prompt_tokens")
    completion_tokens = fields.get("completion_tokens")
    token_part = "-"
    if tokens is not None or prompt_tokens is not None or completion_tokens is not None:
        token_part = f"{prompt_tokens}/{completion_tokens}/{tokens}"
    extra = ""
    if "warnings" in fields:
        extra = f" warnings={fields.get('warnings')}"
    print(
        redact_secrets(
            f"TRACE step={step_name} latency_ms={latency_ms} "
            f"tokens={token_part} cost_usd={fields.get('cost_usd')} "
            f"status={status}{extra}"
        ),
        flush=True,
    )


def log_product_totals() -> None:
    totals = current_product_totals()
    url = current_product_url() or "-"
    if totals is None:
        return
    cost = sum(totals.costs) if totals.costs else None
    tokens = totals.total_tokens or None
    print(
        f"TRACE product_totals url={url} latency_ms={totals.latency_ms} "
        f"tokens={tokens} cost_usd={cost}",
        flush=True,
    )
    write_product_totals(totals.latency_ms, tokens, cost)


def _write_debug(step_name, parent_step, started, latency_ms, status, fields) -> None:
    try:
        write_debug_block(
            format_step_block(
                _iso(started),
                current_product_url(),
                step_name,
                parent_step,
                status,
                latency_ms,
                fields or {},
            )
        )
    except Exception as exc:
        logger.warning("debug trace write failed: %s", redact_secrets(exc))


@contextmanager
def trace_step(step_name: str):
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    status = "ok"
    error_message = None
    with push_step(step_name) as parent_step:
        fields = current_span_fields()
        if fields is None:
            fields = {}
        try:
            yield fields
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            try:
                if fields.get("debug_error") is None:
                    fields["debug_error"] = error_payload(exc)
            except Exception:
                pass
            raise
        finally:
            ended = datetime.now(timezone.utc)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            meta = {}
            if "http_status" in fields:
                meta["http_status"] = fields["http_status"]
            if fields.get("url"):
                meta["url"] = fields["url"]
            write_trace(
                {
                    "run_id": current_run_id(),
                    "product_url": current_product_url(),
                    "step_name": step_name,
                    "parent_step": parent_step,
                    "started_at": _iso(started),
                    "ended_at": _iso(ended),
                    "latency_ms": latency_ms,
                    "status": status,
                    "error_message": error_message,
                    "model": fields.get("model"),
                    "prompt_tokens": fields.get("prompt_tokens"),
                    "completion_tokens": fields.get("completion_tokens"),
                    "total_tokens": fields.get("total_tokens"),
                    "cost_usd": fields.get("cost_usd"),
                    "meta": meta or None,
                }
            )
            _write_debug(step_name, parent_step, started, latency_ms, status, fields)
            _add_product_totals(step_name, latency_ms, fields)
            _print_step_line(step_name, latency_ms, status, fields)


def traced(step_name: str):
    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with trace_step(step_name) as fields:
                    try:
                        fields["debug_input"] = bound_input(fn, args, kwargs)
                    except Exception:
                        pass
                    result = await fn(*args, **kwargs)
                    try:
                        fields["debug_output"] = to_jsonable(result)
                    except Exception:
                        pass
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with trace_step(step_name) as fields:
                try:
                    fields["debug_input"] = bound_input(fn, args, kwargs)
                except Exception:
                    pass
                result = fn(*args, **kwargs)
                try:
                    fields["debug_output"] = to_jsonable(result)
                except Exception:
                    pass
                return result

        return wrapper

    return decorator
