from __future__ import annotations

from dataclasses import dataclass

import httpx

from config import (
    OPENAI_API_URL,
    OPENAI_RESPONSES_URL,
    require_openai_config,
)
from precompute.call_retry import call_with_retry
from tracing import (
    record_http,
    record_llm_exchange,
    record_llm_retry,
    record_llm_usage,
    trace_step,
)

QUALITY_MAX_RUNS = 3
CUT_OFF_PROBLEM = "reply was cut off (hit max tokens)"
FEEDBACK_HEADER = "Your previous reply had these problems. Fix all of them:"


class SchemaRequestRejected(RuntimeError):
    """OpenAI 400 on a structured-output request. Do not retry or strip schema."""


@dataclass
class LlmCallResult:
    text: str
    message: dict
    payload: dict


def format_llm_feedback(problems, extra_lines=None) -> str:
    lines = [FEEDBACK_HEADER]
    for problem in problems:
        lines.append(f" - {problem}")
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


def _response_format(schema_name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def _responses_text_format(schema_name: str, schema: dict) -> dict:
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema,
        }
    }


def _openai_error_text(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text
    if body:
        return body
    return str(exc)


def _responses_usage(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _normalize_responses_payload(payload: dict) -> dict:
    texts = []
    refusals = []
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
                elif part.get("type") == "refusal":
                    refusals.append(
                        part.get("refusal") or part.get("text") or ""
                    )
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
        "refusal": "".join(refusals) or None,
        "tool_calls": tool_calls,
        "output": payload.get("output") or [],
    }
    return message


def _chat_cut_off(payload: dict) -> bool:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    return choices[0].get("finish_reason") == "length"


def _responses_cut_off(payload: dict) -> bool:
    if payload.get("status") != "incomplete":
        return False
    details = payload.get("incomplete_details") or {}
    return details.get("reason") == "max_output_tokens"


def _chat_refusal(message: dict):
    value = message.get("refusal")
    if value:
        return str(value)
    return None


def _run_check(check, text: str) -> list:
    if check is None:
        return []
    try:
        result = check(text)
    except Exception as exc:
        return [str(exc)]
    if not result:
        return []
    if isinstance(result, list):
        return [str(item) for item in result]
    return [str(result)]


def _post_chat(
    messages: list,
    model: str,
    timeout_seconds: float,
    max_tokens=None,
    max_completion_tokens=None,
    tools=None,
    temperature=None,
    schema_name=None,
    schema=None,
) -> dict:
    api_key = require_openai_config()
    body = {
        "model": model,
        "messages": list(messages),
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if tools is not None:
        body["tools"] = tools
    if temperature is not None:
        body["temperature"] = temperature
    if schema_name is not None and schema is not None:
        body["response_format"] = _response_format(schema_name, schema)
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
    return payload


def _post_responses(
    input_items: list,
    model: str,
    timeout_seconds: float,
    tools=None,
    max_output_tokens=None,
    schema_name=None,
    schema=None,
) -> dict:
    api_key = require_openai_config()
    body = {
        "model": model,
        "input": list(input_items),
    }
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if tools is not None:
        body["tools"] = tools
    if schema_name is not None and schema is not None:
        body["text"] = _responses_text_format(schema_name, schema)
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
    record_llm_usage(
        payload.get("model") or model, _responses_usage(payload.get("usage"))
    )
    record_llm_exchange(message)
    return payload


def _post_with_network_retry(post_fn):
    attempt = {"n": 0}

    def once():
        try:
            return post_fn()
        except Exception as exc:
            record_llm_retry(attempt["n"], exc)
            attempt["n"] += 1
            raise

    return call_with_retry(once)


def run_llm_call(
    *,
    api: str,
    model: str,
    timeout_seconds: float,
    messages=None,
    input_items=None,
    max_tokens=None,
    max_completion_tokens=None,
    max_output_tokens=None,
    tools=None,
    temperature=None,
    schema_name=None,
    schema=None,
    check=None,
    step: str = "llm_call",
    append_failed_reply: bool = True,
    feedback_extra=None,
    on_quality_failure=None,
) -> LlmCallResult:
    """One shared Chat Completions / Responses call with cut-off, refusal, check, 3 quality runs."""
    if api == "chat":
        conversation = list(messages or [])
    elif api == "responses":
        conversation = list(input_items or [])
    else:
        raise ValueError(f"unknown llm api: {api!r}")

    last_problems: list = []
    last_feedback = None
    for run in range(1, QUALITY_MAX_RUNS + 1):
        with trace_step(f"{step}_run") as fields:
            fields["debug_input"] = {
                "step": step,
                "attempt": f"{run}/{QUALITY_MAX_RUNS}",
                "schema_name": schema_name,
            }
            try:
                if api == "chat":
                    payload = _post_with_network_retry(
                        lambda: _post_chat(
                            conversation,
                            model,
                            timeout_seconds,
                            max_tokens=max_tokens,
                            max_completion_tokens=max_completion_tokens,
                            tools=tools,
                            temperature=temperature,
                            schema_name=schema_name,
                            schema=schema,
                        )
                    )
                    message = payload["choices"][0]["message"]
                    text = message.get("content") or ""
                    finish_or_status = payload["choices"][0].get("finish_reason")
                    cut_off = _chat_cut_off(payload)
                    refusal = _chat_refusal(message)
                else:
                    payload = _post_with_network_retry(
                        lambda: _post_responses(
                            conversation,
                            model,
                            timeout_seconds,
                            tools=tools,
                            max_output_tokens=max_output_tokens,
                            schema_name=schema_name,
                            schema=schema,
                        )
                    )
                    message = _normalize_responses_payload(payload)
                    text = message.get("content") or ""
                    finish_or_status = payload.get("status")
                    cut_off = _responses_cut_off(payload)
                    refusal = message.get("refusal")
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    error_text = _openai_error_text(exc)
                    fields["debug_output"] = {
                        "step": step,
                        "attempt": f"{run}/{QUALITY_MAX_RUNS}",
                        "schema_name": schema_name,
                        "http_status": 400,
                        "problems": [error_text],
                    }
                    raise SchemaRequestRejected(error_text) from exc
                raise

            problems = []
            if cut_off:
                problems.append(CUT_OFF_PROBLEM)
            if refusal:
                problems.append(f"model refused: {refusal}")
            if not problems:
                problems.extend(_run_check(check, text))

            feedback = format_llm_feedback(problems, feedback_extra) if problems else None
            fields["debug_output"] = {
                "step": step,
                "attempt": f"{run}/{QUALITY_MAX_RUNS}",
                "schema_name": schema_name,
                "finish_reason": finish_or_status,
                "status": finish_or_status,
                "problems": problems,
                "feedback": feedback,
            }
            last_problems = problems
            last_feedback = feedback
            if not problems:
                return LlmCallResult(text=text, message=message, payload=payload)
            if on_quality_failure is not None:
                on_quality_failure(run, problems)
            if run == QUALITY_MAX_RUNS:
                break
            if append_failed_reply:
                conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": feedback})

    raise ValueError("\n".join(last_problems) if last_problems else last_feedback)
