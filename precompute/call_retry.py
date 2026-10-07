from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import httpx

from config import FLOW1_CALL_RETRY_ATTEMPTS, FLOW1_CALL_RETRY_WAIT_SECONDS

T = TypeVar("T")


def is_retryable_call(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def call_with_retry(fn: Callable[..., T], /, *args, **kwargs) -> T:
    """3 tries (original + 2), 5s apart. Retry timeout/network/429/5xx only."""
    last_error: BaseException | None = None
    attempts = FLOW1_CALL_RETRY_ATTEMPTS
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if not is_retryable_call(exc) or attempt == attempts - 1:
                raise
            time.sleep(FLOW1_CALL_RETRY_WAIT_SECONDS)
    raise last_error
