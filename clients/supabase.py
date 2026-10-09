from urllib.parse import quote

import httpx

from config import require_supabase_config
from tracing import record_http_response, traced


def _encode_uri_component(value: str) -> str:
    # JS encodeURIComponent equivalent used by the n8n REST calls.
    return quote(value, safe="-_.!~*'()")


def _headers(key: str, prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _rest(path_and_query: str) -> str:
    supabase_url, _key = require_supabase_config()
    return f"{supabase_url.rstrip('/')}/rest/v1/{path_and_query}"


def _capture(response):
    record_http_response(response, url=str(response.request.url))
    return response


@traced("supabase.get_products_by_nykaa_url")
def get_products_by_nykaa_url(url: str) -> list[dict]:
    """Node 4 query: GET products?nykaa_url=eq.{encodeURIComponent(url)}&select=id,catalog_processed"""
    _, key = require_supabase_config()
    encoded = _encode_uri_component(url)
    endpoint = _rest(
        f"products?nykaa_url=eq.{encoded}&select=id,catalog_processed"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected products response: {payload!r}")
    return payload


@traced("supabase.get_product_row_by_nykaa_url")
def get_product_row_by_nykaa_url(nykaa_url: str) -> dict:
    """Node 8 — CHECK PRODUCT EXISTS. Empty dict if no row (Object.keys length 0)."""
    _, key = require_supabase_config()
    encoded = _encode_uri_component(nykaa_url)
    endpoint = _rest(f"products?nykaa_url=eq.{encoded}")
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected products response: {payload!r}")
    if payload:
        return payload[0]
    return {}


@traced("supabase.update_product_by_nykaa_url")
def update_product_by_nykaa_url(nykaa_url: str, fields: dict) -> list[dict]:
    """Node 10 — Update Product. PATCH matched on nykaa_url, return representation."""
    _, key = require_supabase_config()
    encoded = _encode_uri_component(nykaa_url)
    endpoint = _rest(f"products?nykaa_url=eq.{encoded}")
    response = _capture(
        httpx.patch(
            endpoint,
            headers=_headers(key, prefer="return=representation"),
            json=fields,
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected update response: {payload!r}")
    return payload


@traced("supabase.insert_product")
def insert_product(fields: dict) -> list[dict]:
    """Node 11 — Insert Product. POST with return=representation."""
    _, key = require_supabase_config()
    endpoint = _rest("products")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key, prefer="return=representation"),
            json=fields,
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected insert response: {payload!r}")
    return payload


@traced("supabase.match_ingredients")
def match_ingredients(names: list[str]) -> list[dict]:
    """Node 17 — POST /rpc/match_ingredients with {names: unique list}. Matching stays in Supabase."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/match_ingredients")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json={"names": names},
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected match_ingredients response: {payload!r}")
    return payload


@traced("supabase.insert_new_ingredient")
def insert_new_ingredient(payload: dict) -> dict:
    """Node 21 — POST /ingredients?on_conflict=canonical_name upsert."""
    _, key = require_supabase_config()
    endpoint = _rest("ingredients?on_conflict=canonical_name")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(
                key,
                prefer="return=representation,resolution=merge-duplicates",
            ),
            json=payload,
            timeout=30.0,
        )
    )
    response.raise_for_status()
    body = response.json()
    if isinstance(body, list):
        if not body:
            raise RuntimeError("Insert New Ingredient returned an empty representation")
        return body[0]
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected ingredient insert response: {body!r}")
    return body


@traced("supabase.upsert_product_ingredient")
def upsert_product_ingredient(params: dict) -> None:
    """Node 25 — POST /rpc/upsert_product_ingredient, one row per call."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/upsert_product_ingredient")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json=params,
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.get_matched_product_ingredients")
def get_matched_product_ingredients(product_id) -> list[dict]:
    """Node 28 — GET product_ingredients for this product where matched=true."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_ingredients?product_id=eq.{product_id}"
        f"&matched=eq.true&select=ingredient_id,ingredient_name"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected product_ingredients response: {payload!r}")
    return payload


@traced("supabase.check_enrichment_status")
def check_enrichment_status(ingredient_ids: list) -> list[dict]:
    """Node 30 — POST /rpc/check_enrichment_status."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/check_enrichment_status")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json={"ingredient_ids": ingredient_ids},
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected check_enrichment_status response: {payload!r}")
    return payload


@traced("supabase.mark_product_processed")
def mark_product_processed(product_id) -> None:
    """Node 33 — PATCH products?id=eq.{id} {catalog_processed: true}."""
    _, key = require_supabase_config()
    endpoint = _rest(f"products?id=eq.{product_id}")
    response = _capture(
        httpx.patch(
            endpoint,
            headers=_headers(key, prefer="return=minimal"),
            json={"catalog_processed": True},
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.get_processed_product_ids_by_urls")
def get_processed_product_ids_by_urls(urls: list[str]) -> list:
    """Node 35 — GET products?nykaa_url=in.(catalog URLs)&catalog_processed=eq.true&select=id."""
    _, key = require_supabase_config()
    quoted = ",".join(f'"{url}"' for url in urls)
    endpoint = _rest("products")
    response = _capture(
        httpx.get(
            endpoint,
            headers=_headers(key),
            params={
                "nykaa_url": f"in.({quoted})",
                "catalog_processed": "eq.true",
                "select": "id",
            },
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected products id response: {payload!r}")
    return [row["id"] for row in payload]


@traced("supabase.mark_concerns_analyzed")
def mark_concerns_analyzed(p_names: list) -> None:
    """Node 5 — POST /rpc/mark_concerns_analyzed (public Supabase URL, service role)."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/mark_concerns_analyzed")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json={"p_names": p_names},
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.get_precompute_product_ingredients")
def get_precompute_product_ingredients(product_id) -> list[dict]:
    """Flow 1 node 4 — GET product_ingredients matched rows with display_order."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_ingredients?product_id=eq.{product_id}"
        f"&matched=eq.true&select=ingredient_id,ingredient_name,display_order"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected product_ingredients response: {payload!r}")
    return payload


@traced("supabase.get_product_concentration_row")
def get_product_concentration_row(product_id) -> list[dict]:
    """Flow 1 node 5 — GET product_concentrations?select=product_id&limit=1."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_concentrations?product_id=eq.{product_id}"
        f"&select=product_id&limit=1"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected product_concentrations response: {payload!r}"
        )
    return payload


@traced("supabase.get_product_concentration_fields")
def get_product_concentration_fields(product_id) -> list[dict]:
    """GET product_concentrations concentrations + formulation_product_type. Not node 5."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_concentrations?product_id=eq.{product_id}"
        f"&select=concentrations,formulation_product_type&limit=1"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected product_concentrations response: {payload!r}"
        )
    return payload


@traced("supabase.patch_product_concentrations")
def patch_product_concentrations(
    product_id, concentrations, formulation_product_type
) -> None:
    """PATCH product_concentrations?product_id=eq.{id}. No on_conflict."""
    _, key = require_supabase_config()
    endpoint = _rest(f"product_concentrations?product_id=eq.{product_id}")
    response = _capture(
        httpx.patch(
            endpoint,
            headers=_headers(key, prefer="return=representation"),
            json={
                "product_id": product_id,
                "concentrations": concentrations,
                "formulation_product_type": formulation_product_type,
            },
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.store_approx_concentrations")
def store_approx_concentrations(
    product_id, concentrations, formulation_product_type
) -> None:
    """Flow 1 node 8 — POST product_concentrations, no on_conflict."""
    _, key = require_supabase_config()
    endpoint = _rest("product_concentrations")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key, prefer="resolution=merge-duplicates"),
            json={
                "product_id": product_id,
                "concentrations": concentrations,
                "formulation_product_type": formulation_product_type,
            },
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.store_formulation_type")
def store_formulation_type(product_id, formulation_product_type) -> None:
    """Flow 1 node 9 — POST product_scores?on_conflict=product_id."""
    _, key = require_supabase_config()
    endpoint = _rest("product_scores?on_conflict=product_id")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key, prefer="resolution=merge-duplicates"),
            json={
                "product_id": product_id,
                "formulation_product_type": formulation_product_type,
            },
            timeout=30.0,
        )
    )
    response.raise_for_status()


@traced("supabase.get_formulation_type_from_scores")
def get_formulation_type_from_scores(product_id) -> list:
    """Flow 1 node 22 — GET product_scores?select=formulation_product_type."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_scores?product_id=eq.{product_id}"
        f"&select=formulation_product_type"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected product_scores response: {payload!r}")
    return payload


@traced("supabase.get_formulation_type_from_concentrations")
def get_formulation_type_from_concentrations(product_id) -> list:
    """Flow 1 node 22 — GET product_concentrations?select=formulation_product_type."""
    _, key = require_supabase_config()
    endpoint = _rest(
        f"product_concentrations?product_id=eq.{product_id}"
        f"&select=formulation_product_type"
    )
    response = _capture(httpx.get(endpoint, headers=_headers(key), timeout=30.0))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected product_concentrations response: {payload!r}"
        )
    return payload


@traced("supabase.get_concentration_status")
def get_concentration_status(p_names: list) -> list:
    """Flow 1 node 10 — POST /rpc/get_concentration_status."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/get_concentration_status")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json={"p_names": p_names},
            timeout=30.0,
        )
    )
    response.raise_for_status()
    payload = response.json()
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected get_concentration_status response: {payload!r}"
        )
    return payload


@traced("supabase.get_product_concern_data")
def get_product_concern_data(product_id):
    """Flow 1 node 23 — POST /rpc/get_product_concern_data. One JSON object."""
    _, key = require_supabase_config()
    endpoint = _rest("rpc/get_product_concern_data")
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json={"p_product_id": product_id},
            timeout=30.0,
        )
    )
    response.raise_for_status()
    return response.json()


def _post_rpc_text(path: str, body: dict) -> str:
    _, key = require_supabase_config()
    endpoint = _rest(path)
    response = _capture(
        httpx.post(
            endpoint,
            headers=_headers(key),
            json=body,
            timeout=30.0,
        )
    )
    response.raise_for_status()
    return response.text


@traced("supabase.calculate_product_concern_scores")
def calculate_product_concern_scores(p_product_id) -> str:
    """Concern Agent tool calculate_concern_scores."""
    return _post_rpc_text(
        "rpc/calculate_product_concern_scores",
        {"p_product_id": p_product_id},
    )


@traced("supabase.store_product_concern_scores")
def store_product_concern_scores(p_data: str) -> str:
    """Concern Agent tool store_concern_results. p_data is the model's string."""
    return _post_rpc_text(
        "rpc/store_product_concern_scores",
        {"p_data": p_data},
    )


@traced("supabase.get_active_ingredients_for_interactions")
def get_active_ingredients_for_interactions(product_id) -> str:
    """Interaction Builder tool get_active_ingredients_for_interactions."""
    return _post_rpc_text(
        "rpc/get_active_ingredients_for_interactions_by_product",
        {"p_product_id": product_id},
    )


@traced("supabase.get_predefined_interactions_for_interaction_builder")
def get_predefined_interactions_for_interaction_builder(product_id) -> str:
    """Interaction Builder tool get_predefined_interactions."""
    return _post_rpc_text(
        "rpc/get_predefined_interactions_by_product",
        {"p_product_id": product_id},
    )


@traced("supabase.get_ingredient_irritation_flags")
def get_ingredient_irritation_flags(product_id) -> str:
    """Interaction Builder tool get_ingredient_irritation_flags."""
    return _post_rpc_text(
        "rpc/get_ingredient_irritation_flags_by_product",
        {"p_product_id": product_id},
    )


@traced("supabase.save_llm_interaction")
def save_llm_interaction(
    ingredient_a,
    ingredient_b,
    raw_type,
    severity,
    reason,
    is_irritation_related=None,
) -> str:
    """Interaction Builder tool save_llm_interaction."""
    return _post_rpc_text(
        "rpc/save_llm_interaction",
        {
            "p_ingredient_a": ingredient_a,
            "p_ingredient_b": ingredient_b,
            "p_raw_type": raw_type,
            "p_severity": severity,
            "p_reason": reason,
            "p_is_irritation_related": is_irritation_related,
        },
    )


@traced("supabase.calculate_product_safety_flags")
def calculate_product_safety_flags(p_product_id, p_interaction_result) -> str:
    """Safety Agent tool calculate_and_store_safety_flags."""
    return _post_rpc_text(
        "rpc/calculate_product_safety_flags",
        {
            "p_product_id": p_product_id,
            "p_interaction_result": p_interaction_result,
        },
    )


@traced("supabase.store_product_safety_flags")
def store_product_safety_flags(p_data) -> str:
    """Safety Agent tool store_safety_flags. p_data is JSON.parse'd."""
    return _post_rpc_text(
        "rpc/store_product_safety_flags",
        {"p_data": p_data},
    )


@traced("supabase.get_product_ingredients_with_functions")
def get_product_ingredients_with_functions(p_product_id) -> str:
    """Formulation Agent tool get_product_ingredients_with_functions."""
    return _post_rpc_text(
        "rpc/get_product_ingredients_with_functions",
        {"p_product_id": p_product_id},
    )


@traced("supabase.count_botanicals")
def count_botanicals(p_product_id) -> str:
    """Formulation Agent tool count_botanicals."""
    return _post_rpc_text(
        "rpc/count_botanicals",
        {"p_product_id": p_product_id},
    )


@traced("supabase.calculate_formulation_score")
def calculate_formulation_score(
    p_product_id, p_product_type, p_botanical_count
) -> str:
    """Formulation Agent tool calculate_formulation_score."""
    return _post_rpc_text(
        "rpc/calculate_formulation_score",
        {
            "p_product_id": p_product_id,
            "p_product_type": p_product_type,
            "p_botanical_count": p_botanical_count,
        },
    )


@traced("supabase.store_product_formulation_score")
def store_product_formulation_score(p_data) -> str:
    """Formulation Agent tool store_formulation_score. p_data is JSON.parse'd."""
    return _post_rpc_text(
        "rpc/store_product_formulation_score",
        {"p_data": p_data},
    )


@traced("supabase.patch_ingredient_concentration")
def patch_ingredient_concentration(ingredient_id, concentration_effective) -> None:
    """Flow 1 nodes 19/20 — PATCH ingredients?id=eq.{id}."""
    _, key = require_supabase_config()
    endpoint = _rest(f"ingredients?id=eq.{ingredient_id}")
    response = _capture(
        httpx.patch(
            endpoint,
            headers=_headers(key),
            json={
                "concentration_effective": concentration_effective,
                "concentration_searched": True,
            },
            timeout=30.0,
        )
    )
    response.raise_for_status()

