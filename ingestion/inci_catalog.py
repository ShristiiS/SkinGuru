from __future__ import annotations

from clients.supabase import CANONICAL_NAME_PAGE_SIZE, get_ingredients_catalog_page
from ingestion.product_record import record_warning
from precompute.call_retry import call_with_retry
from tracing.step import record_warnings

INCI_TABLE_CATALOG_WARNING = "ingredient table catalog unavailable"

_STATUS_UNLOADED = "unloaded"
_STATUS_LOADED = "loaded"
_STATUS_FAILED = "failed"

_rows: list[dict] | None = None
_exact: dict[str, dict] | None = None
_near: dict[str, list[dict]] | None = None
_status = _STATUS_UNLOADED
_autoload = False


def reset_inci_ingredient_catalog() -> None:
    global _rows, _exact, _near, _status, _autoload
    _rows = None
    _exact = None
    _near = None
    _status = _STATUS_UNLOADED
    _autoload = False


def enable_inci_catalog_autoload() -> None:
    global _autoload
    _autoload = True


def disable_inci_catalog_autoload() -> None:
    global _autoload
    _autoload = False


def inci_catalog_autoload_enabled() -> bool:
    return _autoload


def inci_catalog_unavailable() -> bool:
    return _status == _STATUS_FAILED


def mark_inci_catalog_failed() -> None:
    global _rows, _exact, _near, _status
    _rows = None
    _exact = None
    _near = None
    _status = _STATUS_FAILED


def near_match_key(name: str) -> str:
    return str(name).upper().replace(" ", "").replace("-", "")


def _has_function_categories(row: dict) -> bool:
    value = row.get("function_categories")
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return bool(value)


def _index_rows(rows: list[dict]) -> None:
    global _rows, _exact, _near, _status
    exact: dict[str, dict] = {}
    near: dict[str, list[dict]] = {}
    usable = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = row.get("canonical_name")
        if not isinstance(name, str) or not name.strip():
            continue
        usable.append(row)
        current = exact.get(name)
        row_id = row.get("id")
        if current is None:
            exact[name] = row
        elif row_id is not None and (
            current.get("id") is None or row_id < current.get("id")
        ):
            exact[name] = row
        near.setdefault(near_match_key(name), []).append(row)
    _rows = usable
    _exact = exact
    _near = near
    _status = _STATUS_LOADED


def set_inci_ingredient_catalog(rows) -> None:
    """Tests: install an in-memory ingredient table without HTTP."""
    _index_rows(list(rows or []))


def get_inci_catalog_index() -> tuple[dict[str, dict], dict[str, list[dict]]] | None:
    if _status != _STATUS_LOADED or _exact is None or _near is None:
        return None
    return _exact, _near


def pick_near_match(matches: list[dict]) -> dict:
    if len(matches) == 1:
        return matches[0]
    with_fn = [row for row in matches if _has_function_categories(row)]
    pool = with_fn if with_fn else matches

    def sort_key(row):
        row_id = row.get("id")
        if row_id is None:
            return (1, 0)
        return (0, row_id)

    return min(pool, key=sort_key)


def load_inci_ingredient_catalog() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Paged GET of ingredients id/canonical_name/function_categories. Cached for the run."""
    if _status == _STATUS_LOADED and _exact is not None and _near is not None:
        return _exact, _near
    collected = []
    offset = 0
    while True:
        page = call_with_retry(
            get_ingredients_catalog_page, offset, CANONICAL_NAME_PAGE_SIZE
        )
        collected.extend(page)
        if len(page) < CANONICAL_NAME_PAGE_SIZE:
            break
        offset += CANONICAL_NAME_PAGE_SIZE
    _index_rows(collected)
    return _exact, _near


def _warn_catalog_failed(exc: Exception) -> None:
    message = f"{INCI_TABLE_CATALOG_WARNING}: {exc}"
    record_warning(message)
    record_warnings([message])


def ensure_inci_ingredient_catalog():
    """Load once if autoload is on. Failure skips the table-match step."""
    if _status == _STATUS_FAILED:
        return None
    if _status == _STATUS_LOADED:
        return get_inci_catalog_index()
    if not _autoload:
        return None
    try:
        return load_inci_ingredient_catalog()
    except Exception as exc:
        mark_inci_catalog_failed()
        _warn_catalog_failed(exc)
        return None
