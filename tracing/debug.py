from __future__ import annotations

import inspect
import json
import traceback

from tracing.writer import redact_secrets

_SECRET_KEY_NAMES = {
    "authorization",
    "apikey",
    "api_key",
    "x-api-key",
    "openai_api_key",
    "serpapi_key",
    "supabase_key",
    "supabase_service_role_key",
    "supabase_anon_key",
    "service_role_key",
}


def to_jsonable(value, _seen=None):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return repr(value)
    if _seen is None:
        _seen = set()
    ident = id(value)
    if ident in _seen:
        return "<recursive>"
    if isinstance(value, dict):
        _seen.add(ident)
        return {str(key): to_jsonable(item, _seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        _seen.add(ident)
        return [to_jsonable(item, _seen) for item in value]
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    if inspect.isclass(value):
        return f"<class {value.__name__}>"
    if inspect.ismodule(value):
        return f"<module {getattr(value, '__name__', '?')}>"
    return repr(value)


def redact_structured(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEY_NAMES:
                redacted[key] = "***"
            else:
                redacted[key] = redact_structured(item)
        return redacted
    if isinstance(value, list):
        return [redact_structured(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


def pretty_json(value) -> str:
    ready = redact_structured(to_jsonable(value))
    return json.dumps(ready, indent=2, ensure_ascii=False, default=str)


def bound_input(fn, args, kwargs):
    try:
        signature = inspect.signature(fn)
        bound = signature.bind_partial(*args, **kwargs)
        return {name: to_jsonable(item) for name, item in bound.arguments.items()}
    except Exception:
        return {"args": to_jsonable(args), "kwargs": to_jsonable(kwargs)}


def error_payload(exc: BaseException) -> dict:
    return {
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def request_body_of(request):
    if request is None:
        return None
    content = getattr(request, "content", None)
    if not content:
        return None
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except Exception:
            return repr(content)
    else:
        text = str(content)
    try:
        return json.loads(text)
    except Exception:
        return text


def response_body_of(response):
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        pass
    try:
        return response.text
    except Exception:
        return None


def format_step_block(
    started_at,
    product_id,
    step_name,
    parent_step,
    status,
    latency_ms,
    fields: dict,
) -> str:
    lines = [
        "=" * 80,
        (
            f"{started_at}  product={product_id or '-'}  "
            f"step={step_name}  parent={parent_step or '-'}  "
            f"status={status}  latency_ms={latency_ms}"
        ),
        "-" * 80,
        "INPUT",
        pretty_json(fields.get("debug_input")),
        "",
        "OUTPUT",
        pretty_json(fields.get("debug_output")),
        "",
        "ERROR",
        pretty_json(fields.get("debug_error")),
    ]
    http_calls = fields.get("debug_http")
    if http_calls:
        lines.extend(["", "HTTP", pretty_json(http_calls)])
    llm = _llm_section(fields)
    if llm is not None:
        lines.extend(["", "LLM", pretty_json(llm)])
    agent = fields.get("agent_debug")
    if agent:
        lines.extend(["", "AGENT", pretty_json(agent)])
    lines.append("=" * 80)
    lines.append("")
    return redact_secrets("\n".join(lines))


def _llm_section(fields: dict):
    if (
        fields.get("llm_reply") is None
        and not fields.get("llm_tool_calls")
        and not fields.get("llm_retries")
        and fields.get("model") is None
    ):
        return None
    prompt_values = fields.get("prompt_values")
    if prompt_values is None and fields.get("debug_input") is not None:
        prompt_values = fields.get("debug_input")
    return {
        "model": fields.get("model"),
        "prompt_values": prompt_values,
        "reply": fields.get("llm_reply"),
        "tool_calls": fields.get("llm_tool_calls") or [],
        "prompt_tokens": fields.get("prompt_tokens"),
        "completion_tokens": fields.get("completion_tokens"),
        "total_tokens": fields.get("total_tokens"),
        "cost_usd": fields.get("cost_usd"),
        "retries": fields.get("llm_retries") or [],
    }
