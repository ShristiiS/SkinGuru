import httpx

from config import SCRAPER_TIMEOUT_SECONDS, SCRAPER_URL
from tracing import record_http, traced

_PARSE_FIELDS = (
    "product_name",
    "brand",
    "price",
    "rating",
    "total_reviews",
    "images",
    "description",
    "how_to_use",
    "key_ingredients",
)


@traced("call_python_scraper")
def call_python_scraper(url: str) -> dict:
    """Node 6 — POST {url} to the existing scraper. Timeout 600000ms."""
    response = httpx.post(
        SCRAPER_URL,
        json={"url": url},
        timeout=SCRAPER_TIMEOUT_SECONDS,
    )
    record_http(
        response.status_code,
        url=SCRAPER_URL,
        method="POST",
        request_body={"url": url},
        response_body=response.text,
    )
    response.raise_for_status()
    return response.json()


@traced("parse_scraper_data")
def parse_scraper_data(scraped: dict, url: str) -> dict:
    """Node 7 — fail if !success; nykaa_url from the loop item, not the scraper body."""
    if not scraped.get("success"):
        raise RuntimeError(f"Scraping failed: {scraped.get('error')}")

    parsed = {"nykaa_url": url}
    for field in _PARSE_FIELDS:
        parsed[field] = scraped.get(field)
    parsed["ingredients"] = scraped.get("ingredients") or []
    parsed["reviews"] = scraped.get("reviews") or []
    return parsed
