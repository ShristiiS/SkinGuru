import random
from datetime import datetime, timezone

from tracing import traced

NO_INGREDIENTS_MESSAGE = "No ingredients found in webhook payload"


@traced("input_parser")
def input_parser(payload) -> list[dict]:
    """Node 1 — INPUT PARSER. One item per ingredient; id duplicates ingredient_id."""
    data = payload
    if isinstance(data, dict) and data.get("body"):
        data = data["body"]

    if isinstance(data, list):
        ingredients = data
    elif isinstance(data, dict):
        ingredients = data.get("ingredients") or []
    else:
        ingredients = []

    if not ingredients:
        raise ValueError(NO_INGREDIENTS_MESSAGE)

    return [
        {
            "ingredient_id": item.get("ingredient_id"),
            "canonical_name": item.get("canonical_name"),
            "id": item.get("ingredient_id"),
        }
        for item in ingredients
    ]


def _n8n_math_random_uuid() -> str:
    # Literal port of n8n: 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(...)
    # using Math.random(), not uuid.uuid4().
    template = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
    chars = []
    for char in template:
        if char == "x":
            r = int(random.random() * 16)
            chars.append(format(r, "x"))
        elif char == "y":
            r = int(random.random() * 16)
            chars.append(format((r & 0x3) | 0x8, "x"))
        else:
            chars.append(char)
    return "".join(chars)


def _iso_timestamp() -> str:
    # JS new Date().toISOString() — UTC with milliseconds and Z.
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    return now.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


@traced("generate_run_id")
def generate_run_id(items: list[dict]) -> list[dict]:
    """Node 2 — stamp the same run_id and started_at onto every item."""
    run_id = _n8n_math_random_uuid()
    started_at = _iso_timestamp()
    return [{**item, "run_id": run_id, "started_at": started_at} for item in items]


@traced("count_and_prepare_list")
def count_and_prepare_list(items: list[dict]) -> dict:
    """Node 3 — Count & Prepare List. One worklist object."""
    return {
        "total_count": len(items),
        "ingredient_list": [item["canonical_name"] for item in items],
        "ingredient_ids": [
            {"name": item["canonical_name"], "id": item["id"]}
            for item in items
        ],
    }


@traced("parse_and_prepare")
def parse_and_prepare(payload) -> tuple[list[dict], dict]:
    """Nodes 1–3. Returns (stamped items, worklist). No branches."""
    items = generate_run_id(input_parser(payload))
    return items, count_and_prepare_list(items)
