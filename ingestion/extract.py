import re

from tracing import traced

_FULL_INGREDINETS = re.compile(r"^Full Ingredinets:\s*", re.I)
_FILLER = re.compile(
    r"^(and |it |this |the |for |with |also |helps |reduces |prevents |is a |is an )",
    re.I,
)
_MARKETING = re.compile(
    r"acne marks|dark spots|pigmentation|melasma|anti-aging|recurrence|discoloration|sunburn|even after|skin tone",
    re.I,
)


@traced("extract_ingredients")
def extract_ingredients(raw_ingredients: list, product_id) -> list[dict]:
    """Node 12 — Extract Ingredients. product_id is Insert/Update's returned id."""
    if not isinstance(raw_ingredients, list):
        raise TypeError("ingredients must be an array")

    survivors = []
    for raw in raw_ingredients:
        ing = _FULL_INGREDINETS.sub("", raw, count=1).strip()
        if len(ing) > 100:
            continue
        if len(ing) < 3:
            continue
        if _FILLER.search(ing):
            continue
        if _MARKETING.search(ing):
            continue
        survivors.append(ing)

    return [
        {
            "product_id": product_id,
            "ingredient_name": name,
            "display_order": index,
        }
        for index, name in enumerate(survivors, start=1)
    ]
