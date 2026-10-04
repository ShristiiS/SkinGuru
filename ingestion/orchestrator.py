from ingestion.aliases import map_common_aliases
from ingestion.catalog import product_urls_catalog
from ingestion.enrichment import apply_enrichment_branch
from ingestion.extract import extract_ingredients
from ingestion.handoff import post_loop_handoff
from ingestion.key_ingredients import mark_key_ingredients
from ingestion.llm_inci import llm_inci_normalizer
from ingestion.match import match_and_upsert_ingredients
from ingestion.normalize import normalize_ingredients
from ingestion.products import upsert_parsed_product
from ingestion.scrape import call_python_scraper, parse_scraper_data
from ingestion.skip import check_if_already_processed
from ingestion.store import (
    get_product_id,
    get_product_ingredients,
    save_product_id,
    store_product_ingredients,
)
from tracing import bind_product, log_product_totals, start_run, trace_step


def run_subflow1() -> dict:
    """Sub-flow 1: catalog loop (nodes 2–34) then post-loop handoff (nodes 35–36)."""
    with start_run():
        with trace_step("run_subflow1"):
            return _run_subflow1_body()


def _run_subflow1_body() -> dict:
    items = product_urls_catalog()
    results = []
    # Node 27 — $getWorkflowStaticData('global').product_ids (unused downstream).
    saved_product_ids = []

    # Node 3 — Process One Product (one URL at a time).
    for item in items:
        url = item["url"]
        with bind_product(url):
            try:
                with trace_step("process_one_product"):
                    check = check_if_already_processed(url)

                    # Node 5 — Skip or Process.
                    if check["already_processed"] is True:
                        # True branch → Replace Me (node 34) → next loop item.
                        # Skip does not mark keys, store, save id, or re-GET ingredients.
                        with trace_step("skip_product"):
                            results.append({**check, "action": "skipped"})
                        continue

                    # Nodes 6–7 — Call Python Scraper → Parse Scraper Data.
                    scraped = call_python_scraper(url)
                    parsed = parse_scraper_data(scraped, url)

                    # Nodes 8–11 — CHECK PRODUCT EXISTS → Update or Insert.
                    written = upsert_parsed_product(parsed)

                    # Node 12 — Extract Ingredients (id from Insert/Update output).
                    extracted = extract_ingredients(
                        parsed["ingredients"],
                        written.extract_input_id(),
                    )
                    # Node 13 — Normalize Ingredients (runs; LLM must not read this field).
                    regex_normalized = normalize_ingredients(extracted)
                    # Node 14 — LLM INCI Normalizer (raw ingredient_name).
                    llm_rows, llm_parse_fallback = llm_inci_normalizer(regex_normalized)
                    # Node 15 — Map Common Aliases.
                    aliased = map_common_aliases(llm_rows)
                    # Nodes 16–23 — unique-name RPC match, remap all occurrences, upsert unmatched, merge.
                    matched_batch = match_and_upsert_ingredients(aliased)
                    # Node 24 — Mark Key Ingredients.
                    marked = mark_key_ingredients(
                        matched_batch["merged"],
                        parsed.get("key_ingredients"),
                    )
                    # Node 25 — Store Product Ingredients (one RPC per row).
                    store_product_ingredients(marked)
                    # Nodes 26–27 — Get Product ID, Save Product ID (static data, unused later).
                    product_id = get_product_id(written)
                    save_product_id(saved_product_ids, product_id)
                    # Node 28 — Get Product Ingredients (main path re-GET, matched=true).
                    stored = get_product_ingredients(written)
                    # Nodes 29–33 — enrichment status; mark processed or fire Sub-flow 2 webhooks.
                    enrichment = apply_enrichment_branch(stored, written)

                    results.append(
                        {
                            **check,
                            "action": written.action,
                            "product_id": product_id,
                            "nykaa_url": parsed["nykaa_url"],
                            "product_name": parsed["product_name"],
                            "extracted_count": len(extracted),
                            "after_llm_count": len(llm_rows),
                            "after_alias_count": len(aliased),
                            "llm_parse_fallback": llm_parse_fallback,
                            "unique_ingredients": matched_batch["unique_ingredients"],
                            "matched_count": matched_batch["matched_count"],
                            "unmatched_count": matched_batch["unmatched_count"],
                            "joined_new_count": matched_batch["joined_new_count"],
                            "dropped_join_count": matched_batch["dropped_join_count"],
                            "key_ingredient_count": sum(
                                1
                                for row in marked
                                if row.get("is_key_ingredient") is True
                            ),
                            "stored_row_count": len(marked),
                            "refetched_matched_count": len(stored),
                            "all_enriched": enrichment["all_enriched"],
                            "catalog_processed": enrichment["catalog_processed"],
                            "subflow2_triggered": enrichment["subflow2_triggered"],
                            "needs_enrichment_count": len(
                                enrichment["needs_enrichment"]
                            ),
                        }
                    )
            finally:
                log_product_totals()

    return {
        "products": results,
        "saved_product_ids": saved_product_ids,
        "handoff": post_loop_handoff(),
    }
