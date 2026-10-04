from __future__ import annotations

import httpx

from config import OPENAI_API_URL, OPENAI_RESPONSES_URL, require_openai_config
from tracing import record_http, record_llm_exchange, record_llm_retry, record_llm_usage

_MAX_RETRIES = 2


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _post_messages(
    messages: list,
    model: str,
    timeout_seconds: float,
    max_tokens=None,
    max_completion_tokens=None,
    tools=None,
) -> dict:
    api_key = require_openai_config()
    body = {
        "model": model,
        "messages": messages,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if tools is not None:
        body["tools"] = tools
    response = httpx.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_seconds,
    )
    record_http(response.status_code, url=OPENAI_API_URL)
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0]["message"]
    record_llm_usage(payload.get("model") or model, payload.get("usage") or {})
    record_llm_exchange(message)
    return message


def chat_completions_turn(
    messages: list,
    model: str,
    timeout_seconds: float,
    max_tokens=None,
    max_completion_tokens=None,
    tools=None,
) -> dict:
    """One Chat Completions turn. No temperature. Retry API errors twice."""
    last_error: BaseException | None = None
    attempts = 1 + _MAX_RETRIES
    for attempt in range(attempts):
        try:
            return _post_messages(
                messages,
                model,
                timeout_seconds,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                tools=tools,
            )
        except Exception as exc:
            last_error = exc
            record_llm_retry(attempt, exc)
            if not _is_retryable(exc) or attempt == attempts - 1:
                raise
    raise last_error


def chat_completions(
    system: str,
    user: str,
    model: str,
    timeout_seconds: float,
    max_tokens=None,
    max_completion_tokens=None,
) -> str:
    """Flow 1 OpenAI Chat Completions. No temperature. Retry API errors twice."""
    message = chat_completions_turn(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model,
        timeout_seconds,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    return message.get("content") or ""


def _responses_usage(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _normalize_responses_payload(payload: dict) -> dict:
    texts = []
    tool_calls = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    texts.append(part.get("text") or "")
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
    message = {
        "content": "".join(texts),
        "tool_calls": tool_calls,
        "output": payload.get("output") or [],
    }
    return message


def _post_responses(
    input_items: list,
    model: str,
    timeout_seconds: float,
    tools=None,
    max_output_tokens=None,
) -> dict:
    api_key = require_openai_config()
    body = {
        "model": model,
        "input": input_items,
    }
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if tools is not None:
        body["tools"] = tools
    response = httpx.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_seconds,
    )
    record_http(response.status_code, url=OPENAI_RESPONSES_URL)
    response.raise_for_status()
    payload = response.json()
    message = _normalize_responses_payload(payload)
    record_llm_usage(payload.get("model") or model, _responses_usage(payload.get("usage")))
    record_llm_exchange(message)
    return message


def responses_turn(
    input_items: list,
    model: str,
    timeout_seconds: float,
    tools=None,
    max_output_tokens=None,
) -> dict:
    """One Responses API turn. No temperature. Retry API errors twice."""
    last_error: BaseException | None = None
    attempts = 1 + _MAX_RETRIES
    for attempt in range(attempts):
        try:
            return _post_responses(
                input_items,
                model,
                timeout_seconds,
                tools=tools,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            last_error = exc
            record_llm_retry(attempt, exc)
            if not _is_retryable(exc) or attempt == attempts - 1:
                raise
    raise last_error
