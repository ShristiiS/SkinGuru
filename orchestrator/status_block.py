from orchestrator.retry import (
    STEP_CONCERN,
    STEP_FUNCTIONAL,
    STEP_LLM,
    STEP_MARK,
)

_COLUMNS = ("concern", "llm", "functional")
_STEP_COLUMN = {
    STEP_CONCERN: "concern",
    STEP_MARK: "concern",
    STEP_LLM: "llm",
    STEP_FUNCTIONAL: "functional",
}


def _empty_columns() -> dict:
    return {name: ("OK", None) for name in _COLUMNS}


def format_status_block(
    parsed_items: list[dict],
    not_processed: list[dict],
    product_url: str | None,
) -> str:
    by_name = {}
    order = []
    for item in parsed_items:
        name = item.get("canonical_name")
        if name not in by_name:
            by_name[name] = _empty_columns()
            order.append(name)
    for row in not_processed or []:
        name = row.get("ingredient_name")
        column = _STEP_COLUMN.get(row.get("step"))
        if not name or not column:
            continue
        if name not in by_name:
            by_name[name] = _empty_columns()
            order.append(name)
        status, _reason = by_name[name][column]
        if status == "OK":
            by_name[name][column] = ("NOT PROCESSED", row.get("error") or "")

    overall = "OK"
    for columns in by_name.values():
        if any(status != "OK" for status, _reason in columns.values()):
            overall = "PARTIAL"
            break

    product = product_url or "-"
    lines = [f"ORCHESTRATOR RUN — {overall} (product: {product})"]
    for name in order:
        columns = by_name[name]
        parts = [f"{col} {columns[col][0]}" for col in _COLUMNS]
        lines.append(f"  {name} — {', '.join(parts)}")
        for col in _COLUMNS:
            status, reason = columns[col]
            if status != "OK":
                lines.append(f"    {col}: {reason}")
    return "\n".join(lines) + "\n"
