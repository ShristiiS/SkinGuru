import logging
import re

from clients.supabase import insert_new_ingredient, match_ingredients
from tracing import traced

logger = logging.getLogger(__name__)

# DELIBERATE EXCEPTION to "port as-is" (the only one in Sub-flow 1):
# n8n node 20 stored canonical_name with punctuation intact, while node 22
# stripped [.,;!] from as_listed_on_label before joining. Labels like "Bang!"
# were inserted as "BANG!" then failed the join and were silently dropped.
# Node 20 now strips the same punctuation before building canonical_name so
# the node 22 join always succeeds. Nodes 22's comparison is unchanged.
_LABEL_PUNCT = re.compile(r"[.,;!]")


def _strip_label_punctuation(value: str) -> str:
    """Uppercase, remove [.,;!], trim — the node 22 label-side transform."""
    return _LABEL_PUNCT.sub("", (value or "").upper()).strip()


@traced("prepare_batch_match")
def prepare_batch_match(items: list[dict]) -> dict:
    """Node 16 — collect every occurrence, dedupe normalized_name for the RPC only."""
    normalized_names = [item["normalized_name"] for item in items]
    ingredient_names = list(dict.fromkeys(normalized_names))
    return {
        "all_ingredients": items,
        "ingredient_names": ingredient_names,
        "total_ingredients": len(items),
        "unique_ingredients": len(ingredient_names),
    }


@traced("find_matching_ingredients")
def find_matching_ingredients(unique_names: list[str]) -> list[dict]:
    """Node 17 — call match_ingredients RPC; do not reimplement matching."""
    return match_ingredients(unique_names)


@traced("map_ingredient_matches")
def map_ingredient_matches(all_ingredients: list[dict], rpc_rows: list[dict]) -> list[dict]:
    """Node 18 — map RPC hits onto every original occurrence, not just uniques."""
    by_canonical = {
        row["canonical_name"]: {
            "ingredient_id": row["id"],
            "ingredient_name": row["canonical_name"],
        }
        for row in rpc_rows
    }
    mapped = []
    for ing in all_ingredients:
        hit = by_canonical.get(ing["normalized_name"])
        if hit:
            mapped.append(
                {
                    "product_id": ing["product_id"],
                    "ingredient_id": hit["ingredient_id"],
                    "ingredient_name": hit["ingredient_name"],
                    "as_listed_on_label": ing["original_name"],
                    "is_key_ingredient": False,
                    "display_order": ing["display_order"],
                    "matched": True,
                }
            )
        else:
            mapped.append(
                {
                    "product_id": ing["product_id"],
                    "ingredient_id": None,
                    "ingredient_name": None,
                    "as_listed_on_label": ing["original_name"],
                    "is_key_ingredient": False,
                    "display_order": ing["display_order"],
                    "matched": False,
                }
            )
    return mapped


@traced("split_matched_unmatched")
def split_matched_unmatched(mapped: list[dict]) -> tuple[list[dict], list[dict]]:
    """Node 19 — Ingredient Found?  $json.matched === true."""
    matched = [item for item in mapped if item["matched"] is True]
    unmatched = [item for item in mapped if item["matched"] is not True]
    return matched, unmatched


@traced("prepare_new_ingredient")
def prepare_new_ingredient(item: dict) -> dict:
    """Node 20 — auto-ingredient payload for one unmatched occurrence.

    DELIBERATE EXCEPTION: strip [.,;!] here (same transform as node 22's
    label comparison) so canonical_name joins back instead of being dropped.
    """
    return {
        "canonical_name": _strip_label_punctuation(
            item.get("as_listed_on_label") or "UNKNOWN"
        ),
        "description": "Auto-added from product - needs review",
        "status": "auto",
        "last_updated_by": "auto_system",
    }


@traced("insert_unmatched_ingredients")
def insert_unmatched_ingredients(unmatched: list[dict]) -> list[dict]:
    """Node 21 — one upsert POST per unmatched item (n8n item-looping)."""
    inserted = []
    for item in unmatched:
        inserted.append(insert_new_ingredient(prepare_new_ingredient(item)))
    return inserted


@traced("map_new_ingredient_id")
def map_new_ingredient_id(inserted_rows: list[dict], mapped_items: list[dict]) -> list[dict]:
    """Node 22 — join each new row back to Map Ingredient Matches; log+drop if none."""
    remapped = []
    for new_ing in inserted_rows:
        canonical = (new_ing.get("canonical_name") or "").upper().strip()
        orig = None
        for item in mapped_items:
            label = _strip_label_punctuation(item.get("as_listed_on_label") or "")
            if label == canonical:
                orig = item
                break
        if orig is None:
            logger.error(
                "No original ingredient found for canonical_name=%r",
                new_ing.get("canonical_name"),
            )
            continue
        remapped.append(
            {
                **orig,
                "ingredient_id": new_ing["id"],
                "ingredient_name": new_ing["canonical_name"],
                "matched": True,
            }
        )
    return remapped


@traced("merge_matched_and_new")
def merge_matched_and_new(matched: list[dict], newly_inserted: list[dict]) -> list[dict]:
    """Node 23 — Merge1 default: combine matched branch + mapped-new branch."""
    return matched + newly_inserted


@traced("match_and_upsert_ingredients")
def match_and_upsert_ingredients(aliased_items: list[dict]) -> dict:
    """Nodes 16–23 as one pipeline, stopping before Mark Key Ingredients."""
    batch = prepare_batch_match(aliased_items)
    rpc_rows = find_matching_ingredients(batch["ingredient_names"])
    mapped = map_ingredient_matches(batch["all_ingredients"], rpc_rows)
    matched, unmatched = split_matched_unmatched(mapped)
    inserted_rows = insert_unmatched_ingredients(unmatched)
    newly_inserted = map_new_ingredient_id(inserted_rows, mapped)
    merged = merge_matched_and_new(matched, newly_inserted)
    return {
        "total_ingredients": batch["total_ingredients"],
        "unique_ingredients": batch["unique_ingredients"],
        "unique_names": batch["ingredient_names"],
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "inserted_count": len(inserted_rows),
        "joined_new_count": len(newly_inserted),
        "dropped_join_count": len(inserted_rows) - len(newly_inserted),
        "merged": merged,
    }
