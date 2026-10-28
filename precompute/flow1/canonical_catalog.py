from __future__ import annotations

from clients.supabase import CANONICAL_NAME_PAGE_SIZE, get_canonical_names_page
from precompute.call_retry import call_with_retry

_STATUS_UNLOADED = "unloaded"
_STATUS_LOADED = "loaded"
_STATUS_FAILED = "failed"

_names: frozenset[str] | None = None
_status = _STATUS_UNLOADED
_autoload = False


def reset_canonical_catalog() -> None:
    global _names, _status, _autoload
    _names = None
    _status = _STATUS_UNLOADED
    _autoload = False


def enable_catalog_autoload() -> None:
    global _autoload
    _autoload = True


def disable_catalog_autoload() -> None:
    global _autoload
    _autoload = False


def catalog_autoload_enabled() -> bool:
    return _autoload


def set_canonical_catalog(names) -> None:
    """Tests: install an in-memory catalog without HTTP."""
    global _names, _status
    collected = []
    for name in names or []:
        if isinstance(name, str) and name.strip():
            collected.append(name)
    _names = frozenset(collected)
    _status = _STATUS_LOADED


def mark_catalog_failed() -> None:
    global _names, _status
    _names = None
    _status = _STATUS_FAILED


def catalog_unavailable() -> bool:
    return _status == _STATUS_FAILED


def get_canonical_names() -> frozenset[str] | None:
    if _status != _STATUS_LOADED:
        return None
    return _names


def load_canonical_catalog() -> frozenset[str]:
    """Paged GET of ingredients.canonical_name. Cached for the rest of the run."""
    global _names, _status
    if _status == _STATUS_LOADED and _names is not None:
        return _names
    collected = []
    offset = 0
    while True:
        page = call_with_retry(
            get_canonical_names_page, offset, CANONICAL_NAME_PAGE_SIZE
        )
        collected.extend(page)
        if len(page) < CANONICAL_NAME_PAGE_SIZE:
            break
        offset += CANONICAL_NAME_PAGE_SIZE
    _names = frozenset(collected)
    _status = _STATUS_LOADED
    return _names
