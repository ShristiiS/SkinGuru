import re

from tracing import traced

_KEY_NAME = re.compile(r"^([^:]+):")
_PARENS = re.compile(r"\s*\([^)]*\)")


@traced("mark_key_ingredients")
def mark_key_ingredients(ingredients: list[dict], key_ingredients) -> list[dict]:
    """Node 24 — Mark Key Ingredients.

    Empty key_ingredients → pass through unchanged.
    Else: name before first ':', strip parens, uppercase, then
    ingredient_name.includes(keyName) substring match.
    """
    if not isinstance(key_ingredients, list) or len(key_ingredients) == 0:
        return ingredients

    key_names = []
    for desc in key_ingredients:
        if not isinstance(desc, str):
            continue
        match = _KEY_NAME.search(desc)
        if not match:
            continue
        name = match.group(1).strip()
        name = _PARENS.sub("", name).strip().upper()
        key_names.append(name)

    marked = []
    for ing in ingredients:
        ingredient_name = ing.get("ingredient_name") or ""
        is_key = any(key_name in ingredient_name for key_name in key_names)
        marked.append({**ing, "is_key_ingredient": is_key})
    return marked
