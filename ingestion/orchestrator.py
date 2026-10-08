from ingestion.aliases import map_common_aliases
from ingestion.catalog import product_urls_catalog
from ingestion.enrichment import (
    apply_enrichment_branch,
    catalog_mark_allowed,
    orchestrator_not_processed_items,
)
from ingestion.extract import extract_ingredients
from ingestion.handoff import post_loop_handoff
from ingestion.key_ingredients import mark_key_ingredients
from ingestion.llm_inci import llm_inci_normalizer
from ingestion.match import match_and_upsert_ingredients
from ingestion.normalize import normalize_ingredients
from ingestion.product_record import (
    bind_product_record,
    current_product_record,
    record_not_processed,
    record_step,
    skip_from,
)
from ingestion.products import upsert_parsed_product
from ingestion.scrape import scrape_and_parse
from ingestion.skip import check_if_already_processed
from ingestion.status_block import append_status_block
from ingestion.store import (
    get_product_id,
    save_product_id,
    store_product_ingredients,
)
from tracing import bind_product, log_product_totals, start_run, trace_step


def _ensure_not_processed(items: list) -> None:
    record = current_product_record()
    if record is None:
        return
    existing = {(row.get("name"), row.get("reason")) for row in record.not_processed}
    for item in items or []:
        key = (item.get("name"), item.get("reason"))
        if key not in existing:
            record_not_processed(item.get("name"), item.get("reason") or "")


def run_subflow1() -> dict:
    """Sub-flow 1: catalog loop (nodes 2–34) then post-loop handoff (nodes 35–36)."""
    with start_run():
        with trace_step("run_subflow1"):
            return _run_subflow1_body()


def _process_one_url(url: str, saved_product_ids: list, step: list) -> dict:
    step[0] = "skip_check"
    check = check_if_already_processed(url)
    record_step("skip_check", "OK")

    # Node 5 — Skip or Process.
    if check["already_processed"] is True:
        # True branch → Replace Me (node 34) → next loop item.
        # Skip does not mark keys, store, save id, or re-GET ingredients.
        record = current_product_record()
        if record is not None:
            record.skipped = True
        with trace_step("skip_product"):
            return {**check, "action": "skipped"}

    step[0] = "scraper"
    parsed = scrape_and_parse(url)
    record_step("scraper", "OK")

    step[0] = "product_upsert"
    written = upsert_parsed_product(parsed)
    record_step("product_upsert", "OK")

    step[0] = "inci_normalizer"
    extracted = extract_ingredients(
        parsed["ingredients"],
        written.extract_input_id(),
    )
    if not extracted:
        raise RuntimeError("no ingredients after extract")
    regex_normalized = normalize_ingredients(extracted)
    llm_rows = llm_inci_normalizer(regex_normalized)
    aliased = map_common_aliases(llm_rows)
    record_step("inci_normalizer", "OK")

    step[0] = "match_ingredients"
    matched_batch = match_and_upsert_ingredients(aliased)
    join_not_processed = list(matched_batch.get("not_processed") or [])
    _ensure_not_processed(join_not_processed)
    record_step(
        "match_ingredients",
        "NOT PROCESSED" if join_not_processed else "OK",
    )
    marked = mark_key_ingredients(
        matched_batch["merged"],
        parsed.get("key_ingredients"),
    )

    step[0] = "store_ingredients"
    stored, store_not_processed = store_product_ingredients(marked, written)
    _ensure_not_processed(store_not_processed)
    record_step(
        "store_ingredients",
        "NOT PROCESSED" if store_not_processed else "OK",
    )
    not_processed = join_not_processed + list(store_not_processed)
    product_id = get_product_id(written)
    save_product_id(saved_product_ids, product_id)

    step[0] = "enrichment"
    enrichment = apply_enrichment_branch(stored, written, not_processed)
    record_step("enrichment", "OK")
    if enrichment.get("subflow2_triggered"):
        record_step("orchestrator", "OK")
    else:
        record_step("orchestrator", "SKIPPED")

    orch_items = orchestrator_not_processed_items(
        enrichment.get("orchestrator_response")
    )
    allowed, why = catalog_mark_allowed(not_processed, enrichment.get("orchestrator_response"))
    marked_catalog = enrichment.get("catalog_processed") is True and allowed
    if marked_catalog:
        record_step("catalog_processed", "OK")
    else:
        record_step(
            "catalog_processed",
            "NOT SET",
            why or enrichment.get("catalog_processed_reason"),
        )

    result = {
        **check,
        "action": written.action,
        "product_id": product_id,
        "nykaa_url": parsed["nykaa_url"],
        "product_name": parsed["product_name"],
        "extracted_count": len(extracted),
        "after_llm_count": len(llm_rows),
        "after_alias_count": len(aliased),
        "unique_ingredients": matched_batch["unique_ingredients"],
        "matched_count": matched_batch["matched_count"],
        "unmatched_count": matched_batch["unmatched_count"],
        "joined_new_count": matched_batch["joined_new_count"],
        "dropped_join_count": matched_batch["dropped_join_count"],
        "not_processed": not_processed + orch_items,
        "key_ingredient_count": sum(
            1 for row in marked if row.get("is_key_ingredient") is True
        ),
        "stored_row_count": len(marked),
        "refetched_matched_count": len(stored),
        "all_enriched": enrichment["all_enriched"],
        "subflow2_triggered": enrichment["subflow2_triggered"],
        "needs_enrichment_count": len(enrichment["needs_enrichment"]),
    }
    if marked_catalog:
        result["catalog_processed"] = True
    return result


def _run_subflow1_body() -> dict:
    items = product_urls_catalog()
    results = []
    # Node 27 — $getWorkflowStaticData('global').product_ids (unused downstream).
    saved_product_ids = []

    # Node 3 — Process One Product (one URL at a time).
    for item in items:
        url = item["url"]
        step = ["skip_check"]
        with bind_product(url), bind_product_record(url) as record:
            try:
                with trace_step("process_one_product"):
                    results.append(_process_one_url(url, saved_product_ids, step))
            except Exception as exc:
                record_step(step[0], "FAILED", str(exc))
                skip_from(step[0], "previous step failed")
                record_step("catalog_processed", "NOT SET", str(exc))
                results.append(
                    {
                        "url": url,
                        "already_processed": False,
                        "action": "failed",
                        "step": step[0],
                        "reason": str(exc),
                    }
                )
            finally:
                append_status_block(record)
                log_product_totals()

    return {
        "products": results,
        "saved_product_ids": saved_product_ids,
        "handoff": post_loop_handoff(),
    }
