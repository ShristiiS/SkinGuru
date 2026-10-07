from __future__ import annotations

ALLOWED_PRODUCT_TYPES = frozenset(
    {
        "CLEANSER",
        "SUNSCREEN",
        "EMULSION",
        "GEL",
        "WATER_SERUM",
        "ANHYDROUS",
    }
)


def check_llm1(node5: dict) -> None:
    product_type = node5.get("product_type")
    if product_type not in ALLOWED_PRODUCT_TYPES:
        raise ValueError(f"LLM1 product_type invalid: {product_type!r}")


def check_llm2(node8: dict, ingredients: list) -> None:
    orders = [item["display_order"] for item in ingredients]
    marker_position = node8.get("marker_position")
    if marker_position not in orders:
        raise ValueError(
            f"LLM2 marker_position {marker_position!r} is not in the ingredient list"
        )


def _names_one_per_ingredient(rows, ingredients: list, label: str) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{label} result is not a list")
    expected = [item["ingredient_name"] for item in ingredients]
    names = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("ingredient_name"):
            raise ValueError(f"{label} row missing ingredient_name")
        names.append(row["ingredient_name"])
    if len(names) != len(expected):
        raise ValueError(
            f"{label} expected {len(expected)} rows, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{label} duplicate ingredient names")
    if set(names) != set(expected):
        raise ValueError(f"{label} missing or invented ingredient names")


def check_llm3(node11: dict, ingredients: list) -> None:
    _names_one_per_ingredient(
        node11.get("classified_ingredients"), ingredients, "LLM3"
    )


def check_llm4(node14: dict, ingredients: list) -> None:
    rows = node14.get("ingredients_with_bands")
    _names_one_per_ingredient(rows, ingredients, "LLM4")
    for row in rows:
        try:
            if row["band_L"] > row["band_U"]:
                raise ValueError(
                    f"LLM4 band_L > band_U for {row.get('ingredient_name')!r}"
                )
        except TypeError as exc:
            raise ValueError("LLM4 band_L/band_U are not comparable") from exc
