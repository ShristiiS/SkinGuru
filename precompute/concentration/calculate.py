from __future__ import annotations

from precompute.concentration.helpers import js_percent, round2
from tracing.step import record_warnings


def calculate_concentrations(data: dict) -> dict:
    """Node 15 — Calculate Concentrations. Ported line by line from the n8n code node."""
    ingredients = data["ingredients_with_bands"]
    marker_position = data["marker_position"]

    # JS reduce(..., ingredients[0]) on [] is undefined; find/filter never run.
    if not ingredients:
        result = {"concentrations": {}, "warnings": None}
        record_warnings(result["warnings"])
        return result

    # Position 1 ingredient is ALWAYS the base solvent — calculate as remainder
    base_ingredient = ingredients[0]
    for item in ingredients:
        if item["display_order"] < base_ingredient["display_order"]:
            base_ingredient = item

    def is_water(name) -> bool:
        return name.upper() == base_ingredient["ingredient_name"].upper()

    water_ingredient = next(
        (item for item in ingredients if is_water(item["ingredient_name"])),
        None,
    )
    non_water_ingredients = [
        item for item in ingredients if not is_water(item["ingredient_name"])
    ]

    zone_a = [
        item for item in non_water_ingredients if item["zone"] == "A"
    ]
    zone_a.sort(key=lambda item: item["display_order"])
    zone_b = [item for item in non_water_ingredients if item["zone"] == "B"]
    zone_c = [item for item in non_water_ingredients if item["zone"] == "C"]

    concentrations = {}

    # ZONE A — position-weighted estimation with descending order enforcement
    prev_conc = None
    for ing in zone_a:
        p = ing["display_order"]
        a = marker_position
        lower = ing["band_L"]
        upper = ing["band_U"]

        ratio = (a - p) / (a - 2 + 0.001)
        weight = 0.40 + 0.40 * min(max(ratio, 0), 1)
        conc = lower + weight * (upper - lower)
        conc = round2(conc)

        if prev_conc is not None and conc >= prev_conc:
            conc = round2(prev_conc * 0.85)

        if conc < lower:
            conc = lower

        concentrations[ing["ingredient_name"]] = conc
        prev_conc = conc

    # ZONE B — midpoint
    for ing in zone_b:
        conc = round2((ing["band_L"] + ing["band_U"]) / 2)
        concentrations[ing["ingredient_name"]] = conc

    # ZONE C — midpoint, no descending order enforced
    for ing in zone_c:
        conc = round2((ing["band_L"] + ing["band_U"]) / 2)
        concentrations[ing["ingredient_name"]] = conc

    # BASE INGREDIENT — calculate as remainder
    non_water_sum = sum(concentrations.values())
    water_conc = round2(100 - non_water_sum)
    if water_ingredient:
        concentrations[water_ingredient["ingredient_name"]] = (
            water_conc if water_conc > 0 else 0
        )

    # FAILURE CHECKS
    warnings = []

    zone_a_concs = [
        {
            "name": item["ingredient_name"],
            "conc": concentrations[item["ingredient_name"]],
        }
        for item in zone_a
    ]
    for index in range(1, len(zone_a_concs)):
        current = zone_a_concs[index]
        previous = zone_a_concs[index - 1]
        if current["conc"] >= previous["conc"]:
            warnings.append(
                f"DESCENDING ORDER VIOLATION: {current['name']} "
                f"({js_percent(current['conc'])}) >= {previous['name']} "
                f"({js_percent(previous['conc'])})"
            )

    below1 = [
        item
        for item in zone_a
        if concentrations[item["ingredient_name"]] < 1
    ]
    if len(below1) > len(zone_a) * 0.5:
        warnings.append(
            "MARKER MAY BE TOO LATE: more than half of pre-marker "
            "ingredients estimated below 1%"
        )

    if (
        water_ingredient
        and concentrations[water_ingredient["ingredient_name"]] < 0
    ):
        warnings.append(
            "WATER IS NEGATIVE: non-water ingredients sum exceeds 100%"
        )

    formatted_concentrations = {}
    for name, conc in concentrations.items():
        formatted_concentrations[name] = js_percent(conc)

    result = {
        "concentrations": formatted_concentrations,
        "warnings": warnings if warnings else None,
    }
    record_warnings(result["warnings"])
    return result
