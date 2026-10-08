"""Match check_enrichment_status rows to the Orchestrator's {name, id} items."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from clients.supabase import check_enrichment_status
from config import LLM_BATCH_SIZE, ORCHESTRATOR_MAX_ATTEMPTS
from orchestrator.retry import AttemptHadFailures, record_attempt, wait_before_retry
from precompute.call_retry import call_with_retry

REASON_N8N_NOT_SAVED = "n8n replied but not saved"


async def supabase_call_retry(fn, /, *args):
    return await asyncio.to_thread(call_with_retry, fn, *args)


def items_missing_flag(items: list[dict], rows: list[dict], flag: str) -> list[dict]:
    """Items whose RPC row is missing or whose flag is not true."""
    by_id = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "ingredient_id" in row:
                by_id[row["ingredient_id"]] = row
    missing = []
    for item in items:
        row = by_id.get(item["id"])
        if row is None or not row.get(flag):
            missing.append(item)
    return missing


def _not_saved_leftovers(items: list[dict], step: str) -> list[dict]:
    return [
        {
            "ingredient_name": item["name"],
            "step": step,
            "error": REASON_N8N_NOT_SAVED,
        }
        for item in items
    ]


def _split_budget(missing: list[dict], attempts_used: dict) -> tuple[list[dict], list[dict]]:
    resendable = []
    exhausted = []
    for item in missing:
        if attempts_used.get(item["name"], 0) >= ORCHESTRATOR_MAX_ATTEMPTS:
            exhausted.append(item)
        else:
            resendable.append(item)
    return resendable, exhausted


def _http_error_leftovers(items: list[dict], step: str, error: str) -> list[dict]:
    return [
        {
            "ingredient_name": item["name"],
            "step": step,
            "error": error,
        }
        for item in items
    ]


async def verify_and_resend(
    *,
    items: list[dict],
    attempts_used: dict,
    n8n_failed_names: set,
    flag: str,
    step: str,
    resend: Callable[[list[dict]], Awaitable[list[dict]]],
) -> list[dict]:
    """After n8n, resend unsaved flags using leftover of 6 attempts per ingredient."""
    passed_items = [item for item in items if item["name"] not in n8n_failed_names]
    if not passed_items:
        return []
    all_ids = [item["id"] for item in items]
    try:
        rows = await supabase_call_retry(check_enrichment_status, all_ids)
    except Exception as exc:
        return _http_error_leftovers(passed_items, step, str(exc))

    leftovers = []
    missing = items_missing_flag(passed_items, rows, flag)
    resendable, exhausted = _split_budget(missing, attempts_used)
    leftovers.extend(_not_saved_leftovers(exhausted, step))

    while resendable:
        await wait_before_retry(2)
        for offset in range(0, len(resendable), LLM_BATCH_SIZE):
            chunk = resendable[offset : offset + LLM_BATCH_SIZE]
            handled = await resend(chunk)
            passed = []
            failed = []
            for item, row in zip(chunk, handled):
                name = item["name"]
                attempts_used[name] = attempts_used.get(name, 0) + 1
                if not isinstance(row, dict):
                    row = {}
                if row.get("status") == "failed" or row.get("error"):
                    failed.append(
                        {
                            "ingredient_name": name,
                            "error": row.get("error") or "Unknown error",
                        }
                    )
                else:
                    passed.append(name)
            try:
                record_attempt(
                    step,
                    max(attempts_used[item["name"]] for item in chunk),
                    sent=[item["name"] for item in chunk],
                    passed=passed,
                    failed=failed,
                )
            except AttemptHadFailures:
                pass
        try:
            rows = await supabase_call_retry(check_enrichment_status, all_ids)
        except Exception as exc:
            leftovers.extend(_http_error_leftovers(resendable, step, str(exc)))
            return leftovers
        missing = items_missing_flag(passed_items, rows, flag)
        resendable, exhausted = _split_budget(missing, attempts_used)
        leftovers.extend(_not_saved_leftovers(exhausted, step))

    return leftovers
