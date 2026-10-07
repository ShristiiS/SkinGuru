from __future__ import annotations

import queue
import threading
import uuid

from precompute.flow1.product import process_one_product
from tracing import start_run, traced

ACCEPTED_REPLY = {
    "success": True,
    "message": "Pre-computation started",
}
PRODUCT_IDS_ERROR = "product_ids must be a list"

_jobs: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


@traced("split_product_ids")
def split_product_ids(product_ids: list) -> list:
    """Node 2 — one item per id, in the given order."""
    return list(product_ids)


@traced("run_flow1")
def run_flow1(product_ids: list) -> None:
    """Node 3 — products one at a time. One product error does not stop the list."""
    for product_id in split_product_ids(product_ids):
        try:
            process_one_product(product_id)
        except Exception:
            continue


def _worker_loop() -> None:
    while True:
        job = _jobs.get()
        try:
            with start_run(job["run_id"]):
                run_flow1(job["product_ids"])
        except Exception:
            pass
        finally:
            _jobs.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(
            target=_worker_loop, name="flow1-worker", daemon=True
        )
        thread.start()
        _worker_started = True


def enqueue(product_ids: list, run_id: str) -> None:
    _ensure_worker()
    _jobs.put({"product_ids": product_ids, "run_id": run_id})


def parse_product_ids(payload) -> list:
    """Node 1 body: product_ids must be a list (empty list is a no-op run)."""
    if not isinstance(payload, dict) or not isinstance(
        payload.get("product_ids"), list
    ):
        raise ValueError(PRODUCT_IDS_ERROR)
    return payload["product_ids"]


def accept_precompute(product_ids: list, run_id_header: str | None) -> dict:
    """Node 1 CHANGED: reply immediately; one worker; whole list enqueued."""
    run_id = (run_id_header or "").strip() or str(uuid.uuid4())
    enqueue(product_ids, run_id)
    return ACCEPTED_REPLY


def wait_until_idle(timeout: float = 5.0) -> None:
    done = threading.Event()

    def _join() -> None:
        _jobs.join()
        done.set()

    thread = threading.Thread(target=_join, daemon=True)
    thread.start()
    if not done.wait(timeout):
        raise TimeoutError("Flow 1 worker did not become idle")
