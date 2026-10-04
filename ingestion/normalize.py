import re

from tracing import traced


@traced("normalize_ingredients")
def normalize_ingredients(items: list[dict]) -> list[dict]:
    """Node 13 — Normalize Ingredients. Produces normalized_name; does not change ingredient_name."""
    normalized = []
    for item in items:
        name = item["ingredient_name"]
        name = re.sub(r"\s*\([^)]*\)", "", name)
        name = re.sub(r"\d+%\s*", "", name)
        name = re.sub(r"\s+and\s*$", "", name, flags=re.I)
        name = re.sub(r"[.,;!]", "", name)
        name = name.strip().upper()
        normalized.append(
            {
                **item,
                "normalized_name": name,
                "original_name": item["ingredient_name"],
            }
        )
    return normalized
