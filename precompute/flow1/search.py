from __future__ import annotations

import re

import httpx

from clients.supabase import (
    _encode_uri_component,
    get_concentration_status,
    patch_ingredient_concentration,
)
from config import SERPAPI_TIMEOUT_SECONDS, require_serpapi_config
from precompute.call_retry import call_with_retry
from precompute.flow1.llm import chat_completions
from precompute.flow1.product_record import record_not_processed, record_rerun
from precompute.flow1.prompts import (
    EXTRACT_CONCENTRATION_SYSTEM,
    render_extract_concentration_user,
)
from tracing import record_http, record_prompt_values, traced, trace_step

EXTRACT_MODEL = "gpt-4o-mini"
EXTRACT_TIMEOUT_SECONDS = 180
EXTRACT_MAX_RUNS = 3
_EXTRACT_PERCENT = re.compile(r"^\d+(?:\.\d+)?%$")
_EXTRACT_RANGE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?%$")
SERPAPI_QUERY_SUFFIX = " effective concentration percentage skincare cosmetic"
_ESTIMATOR_MISSING = (
    "Couldn't get data for node 'concentration estimator trigger'"
)


def _node11_id_empty(row: dict) -> bool:
    """`($json.id?.toString() ?? '') === ''`."""
    value = row.get("id") if isinstance(row, dict) else None
    if value is None:
        return True
    return str(value) == ""


def _node12_valid(row: dict) -> bool:
    """`id != null && id !== ''`."""
    if not isinstance(row, dict):
        return False
    value = row.get("id")
    return value is not None and value != ""


def mask_serpapi_url(url: str) -> str:
    marker = "api_key="
    start = url.find(marker)
    if start < 0:
        return url
    start += len(marker)
    end = url.find("&", start)
    if end < 0:
        return url[:start] + "***"
    return url[:start] + "***" + url[end:]


@traced("get_concentration_status")
def fetch_concentration_status(node4: dict) -> list:
    """Node 10 — p_names are node 4 ingredient_name values, in node 4 order."""
    p_names = [item["ingredient_name"] for item in node4["ingredients"]]
    return call_with_retry(get_concentration_status, p_names)


@traced("prepare_loop_items")
def prepare_loop_items(rows: list, product_id, estimator_result) -> list:
    """Node 12 — Prepare Loop Items."""
    valid = [row for row in rows if _node12_valid(row)]
    if not valid:
        if estimator_result is None:
            raise RuntimeError(_ESTIMATOR_MISSING)
        return [
            {
                "product_id": product_id,
                "concentrations": estimator_result["concentrations"],
            }
        ]
    return [
        {
            "id": row["id"],
            "canonical_name": row["canonical_name"],
            "needs_search": True,
        }
        for row in valid
    ]


@traced("serpapi_search")
def serpapi_search(canonical_name) -> dict:
    """Node 14 — SerpAPI. api_key is masked in traces and errors."""
    api_key = require_serpapi_config()
    query = str(canonical_name) + SERPAPI_QUERY_SUFFIX
    url = (
        "https://serpapi.com/search.json"
        f"?api_key={api_key}"
        f"&q={_encode_uri_component(query)}"
        "&num=3"
    )
    masked = mask_serpapi_url(url)

    def _get():
        response = httpx.get(url, timeout=SERPAPI_TIMEOUT_SECONDS)
        recorded_body = None
        try:
            recorded_body = response.json()
        except Exception:
            try:
                recorded_body = response.text
            except Exception:
                recorded_body = None
        record_http(
            response.status_code,
            url=masked,
            method="GET",
            response_body=recorded_body,
        )
        if response.status_code == 429 or response.status_code >= 500:
            response.raise_for_status()
        return response

    try:
        response = call_with_retry(_get)
    except httpx.TimeoutException:
        raise RuntimeError("SerpAPI timeout") from None
    except httpx.NetworkError:
        raise RuntimeError("SerpAPI network error") from None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise RuntimeError(f"SerpAPI HTTP {status}") from None
    if response.is_error:
        raise RuntimeError(f"SerpAPI HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected SerpAPI response")
    return payload


@traced("extract_concentration")
def extract_concentration(canonical_name, serp_payload: dict) -> str:
    """Node 15 — Extract Concentration. gpt-4o-mini, no temperature, no max tokens."""
    organic_results = serp_payload.get("organic_results")
    if organic_results is None:
        snippets = None
    else:
        snippets = [
            item.get("snippet") if isinstance(item, dict) else None
            for item in organic_results[:3]
        ]
    record_prompt_values(
        {"canonical_name": canonical_name, "search_snippets": snippets}
    )
    return chat_completions(
        EXTRACT_CONCENTRATION_SYSTEM,
        render_extract_concentration_user(canonical_name, organic_results),
        EXTRACT_MODEL,
        EXTRACT_TIMEOUT_SECONDS,
    )


def extract_text_is_allowed(text) -> bool:
    concentration = text.strip() if isinstance(text, str) else ""
    compared = concentration.rstrip(".")
    if not compared or compared.lower() == "not specified":
        return True
    if _EXTRACT_PERCENT.fullmatch(compared):
        return True
    if _EXTRACT_RANGE.fullmatch(compared):
        return True
    return False


@traced("prepare_concentration_update")
def prepare_concentration_update(text, item: dict) -> dict:
    """Node 17 — trailing '.' stripped only for the not-specified comparison."""
    concentration = text.strip() if isinstance(text, str) else ""
    compared = concentration.rstrip(".")
    if not compared or compared.lower() == "not specified":
        return {
            "skip_update": True,
            "id": item["id"],
            "canonical_name": item["canonical_name"],
        }
    return {
        "skip_update": False,
        "id": item["id"],
        "canonical_name": item["canonical_name"],
        "concentration_effective": concentration,
    }


@traced("patch_not_specified")
def patch_not_specified(ingredient_id) -> None:
    """Node 19."""
    call_with_retry(patch_ingredient_concentration, ingredient_id, "not specified")


@traced("patch_concentration")
def patch_concentration(ingredient_id, concentration_effective) -> None:
    """Node 20."""
    call_with_retry(
        patch_ingredient_concentration, ingredient_id, concentration_effective
    )


def _record_ingredient_not_processed(name, reason: str) -> None:
    record_not_processed(name, reason)
    with trace_step("ingredient_not_processed") as fields:
        fields["debug_input"] = {"canonical_name": name}
        fields["debug_output"] = {"result": "NOT PROCESSED", "reason": reason}


@traced("process_one_ingredient")
def process_one_ingredient(item: dict) -> None:
    """Nodes 14–21 for one loop item."""
    name = item["canonical_name"]
    serp_payload = serpapi_search(name)
    text = None
    for run in range(1, EXTRACT_MAX_RUNS + 1):
        try:
            with trace_step("extract_run") as fields:
                fields["debug_input"] = {"run": run, "canonical_name": name}
                text = extract_concentration(name, serp_payload)
                fields["debug_output"] = text
                if not extract_text_is_allowed(text):
                    raise ValueError("extract output invalid")
            break
        except ValueError as exc:
            record_rerun("extract", run, str(exc))
            if run == EXTRACT_MAX_RUNS:
                _record_ingredient_not_processed(
                    name, "extract output invalid after 3 runs"
                )
                return
    prepared = prepare_concentration_update(text, item)
    if prepared["skip_update"]:
        patch_not_specified(prepared["id"])
    else:
        patch_concentration(prepared["id"], prepared["concentration_effective"])


@traced("search_concentrations")
def search_concentrations(node4: dict, estimator_result) -> None:
    """Nodes 10–21. Node 22 is invoked once by the caller afterwards."""
    rows = fetch_concentration_status(node4)
    if not rows:
        return
    to_prepare = [row for row in rows if not _node11_id_empty(row)]
    if not to_prepare:
        return
    items = prepare_loop_items(
        to_prepare, node4["product_id"], estimator_result
    )
    to_search = [item for item in items if item.get("canonical_name")]
    if not to_search:
        return
    for item in to_search:
        try:
            process_one_ingredient(item)
        except Exception as exc:
            _record_ingredient_not_processed(item.get("canonical_name"), str(exc))
            continue
