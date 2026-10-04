from __future__ import annotations


def parse_input(rows: list[dict]) -> dict:
    """Node 2 — Parse Input. Caller already has Flow 1 node 4 rows (a list)."""
    kept = [
        item
        for item in rows
        if item.get("ingredient_name") and item["ingredient_name"].strip() != ""
    ]
    kept.sort(key=lambda item: item["display_order"])
    ingredients = [
        {
            "ingredient_name": item["ingredient_name"].strip().upper(),
            "display_order": item["display_order"],
        }
        for item in kept
    ]
    return {"ingredients": ingredients}
