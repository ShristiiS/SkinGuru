from tracing import traced

ALIASES = {
    "PURIFIED WATER": "AQUA",
    "WATER": "AQUA",
    "EAU": "AQUA",
    "DEIONIZED WATER": "AQUA",
    "DISTILLED WATER": "AQUA",
    "PERFUME": "FRAGRANCE",
    "PARFUM": "FRAGRANCE",
    "FRAGRANCE": "FRAGRANCE",
}


@traced("map_common_aliases")
def map_common_aliases(items: list[dict]) -> list[dict]:
    """Node 15 — Map Common Aliases. Writes the mapped value to both name fields."""
    mapped_items = []
    for item in items:
        source = item.get("ingredient_name") or item.get("normalized_name") or ""
        lookup = source.upper().strip()
        mapped = ALIASES.get(lookup, lookup)
        mapped_items.append(
            {
                **item,
                "ingredient_name": mapped,
                "normalized_name": mapped,
            }
        )
    return mapped_items
