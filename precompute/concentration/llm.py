from __future__ import annotations

import time

import httpx

from config import (
    FLOW1_CALL_RETRY_WAIT_SECONDS,
    OPENAI_API_URL,
    OPENAI_TIMEOUT_SECONDS,
    require_openai_config,
)
from tracing import record_http, record_llm_exchange, record_llm_retry, record_llm_usage

_MAX_RETRIES = 2


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _post_chat(system: str, user: str, model: str, max_tokens: int) -> str:
    api_key = require_openai_config()
    response = httpx.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    record_http(response.status_code, url=OPENAI_API_URL)
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"] or ""
    record_llm_usage(payload.get("model") or model, payload.get("usage") or {})
    record_llm_exchange({"content": text})
    return text


def chat_completions(system: str, user: str, model: str, max_tokens: int) -> str:
    """OpenAI Chat Completions. No temperature. Retry API errors up to 2 times."""
    last_error: BaseException | None = None
    attempts = 1 + _MAX_RETRIES
    for attempt in range(attempts):
        try:
            return _post_chat(system, user, model, max_tokens)
        except Exception as exc:
            last_error = exc
            record_llm_retry(attempt, exc)
            if not _is_retryable(exc) or attempt == attempts - 1:
                raise
            time.sleep(FLOW1_CALL_RETRY_WAIT_SECONDS)
    raise last_error
