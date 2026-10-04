from clients.supabase import get_products_by_nykaa_url
from tracing import traced


@traced("check_if_already_processed")
def check_if_already_processed(url: str) -> dict:
    """Node 4 — Check If Already Processed.

    already_processed is true only when a row exists AND catalog_processed === true.
    """
    result = get_products_by_nykaa_url(url)
    already_processed = len(result) > 0 and result[0].get("catalog_processed") is True
    return {"url": url, "already_processed": already_processed}
