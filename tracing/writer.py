from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from tracing.context import current_product_url, current_run_id

logger = logging.getLogger("tracing")

_API_KEY_RE = re.compile(r"(?i)(api_key=)[^&\s|]*")
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-+=/]+")
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9\-_]{8,}")
_AUTH_ASSIGN_RE = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?)[^\s\"']+"
)
_APIKEY_ASSIGN_RE = re.compile(
    r"(?i)((?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}&]+"
)
_SECRET_ENV_NAMES = (
    "SERPAPI_KEY",
    "OPENAI_API_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_ANON_KEY",
    "SUPABASE_KEY",
)


def _default_traces_dir() -> Path:
    override = os.environ.get("TRACES_DIR")
    if override:
        return Path(override)
    if Path("/app").is_dir():
        return Path("/app/traces")
    return Path("traces")


TRACES_DIR = _default_traces_dir()
COLUMNS = (
    "started_at",
    "product_url",
    "step_name",
    "parent_step",
    "latency_ms",
    "status",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "error_message",
)

_lock = threading.Lock()
_paths: dict[str, Path] = {}
_debug_paths: dict[str, Path] = {}


def _known_secret_strings() -> list[str]:
    found: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 4:
            found.append(value)
    try:
        import config as cfg

        for attr in _SECRET_ENV_NAMES:
            value = getattr(cfg, attr, None)
            if isinstance(value, str) and len(value) >= 4:
                found.append(value)
    except Exception:
        pass
    return sorted(set(found), key=len, reverse=True)


def redact_secrets(value) -> str:
    """Keys and auth header values must never appear in traces or logs."""
    text = str(value)
    for secret in _known_secret_strings():
        text = text.replace(secret, "***")
    text = _API_KEY_RE.sub(r"\1***", text)
    text = _BEARER_RE.sub(r"\1***", text)
    text = _OPENAI_KEY_RE.sub("***", text)
    text = _AUTH_ASSIGN_RE.sub(lambda match: match.group(1) + "***", text)
    text = _APIKEY_ASSIGN_RE.sub(lambda match: match.group(1) + "***", text)
    return text


def _cell(value) -> str:
    if value is None:
        return ""
    text = redact_secrets(value).replace("|", "/").replace("\r", " ").replace("\n", " ")
    return text


def _ensure_run_files(run_id: str) -> tuple[Path, Path]:
    with _lock:
        existing = _paths.get(run_id)
        if existing is not None:
            debug = _debug_paths.get(run_id)
            if debug is None:
                debug = existing.with_name(f"{existing.stem}_debug.txt")
                _debug_paths[run_id] = debug
            return existing, debug
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = run_id[:8]
        path = TRACES_DIR / f"run_{stamp}_{short_id}.txt"
        debug = TRACES_DIR / f"run_{stamp}_{short_id}_debug.txt"
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("|".join(COLUMNS) + "\n")
                handle.flush()
        if not debug.exists():
            debug.touch()
        _paths[run_id] = path
        _debug_paths[run_id] = debug
        return path, debug


def _run_file(run_id: str) -> Path:
    return _ensure_run_files(run_id)[0]


def _debug_file(run_id: str) -> Path:
    return _ensure_run_files(run_id)[1]


def _append_line(path: Path, line: str) -> None:
    with _lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()


def write_trace(row: dict) -> None:
    """Append one step line to the run file. Never raises into the pipeline."""
    try:
        run_id = row.get("run_id")
        if not run_id:
            logger.warning("trace file write skipped: no run_id")
            return
        line = "|".join(_cell(row.get(column)) for column in COLUMNS)
        _append_line(_run_file(str(run_id)), line)
    except Exception as exc:
        logger.warning("trace file write failed: %s", redact_secrets(exc))


def write_debug_block(text: str) -> None:
    """Append one debug step block. Never raises into the pipeline."""
    try:
        run_id = current_run_id()
        if not run_id:
            logger.warning("debug trace write skipped: no run_id")
            return
        path = _debug_file(str(run_id))
        block = text if text.endswith("\n") else text + "\n"
        _append_line_raw(path, block)
    except Exception as exc:
        logger.warning("debug trace write failed: %s", redact_secrets(exc))


def _append_line_raw(path: Path, text: str) -> None:
    with _lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()


def write_product_totals(latency_ms, tokens, cost_usd) -> None:
    """Append one product totals line. Never raises into the pipeline."""
    try:
        run_id = current_run_id()
        if not run_id:
            logger.warning("trace totals write skipped: no run_id")
            return
        line = "|".join(
            (
                "TOTALS",
                _cell(current_product_url()),
                _cell(latency_ms),
                _cell(tokens),
                _cell(cost_usd),
            )
        )
        _append_line(_run_file(run_id), line)
    except Exception as exc:
        logger.warning("trace file write failed: %s", redact_secrets(exc))
