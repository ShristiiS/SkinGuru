import httpx

from config import SCRAPER_TIMEOUT_SECONDS, SCRAPER_URL
from ingestion.product_record import record_rerun
from precompute.call_retry import call_with_retry
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


def _post_scraper(url: str) -> dict:
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


SCRAPER_MAX_RUNS = 3


@traced("call_python_scraper")
def call_python_scraper(url: str) -> dict:
    """Node 6 — POST {url} to the existing scraper. Timeout 600000ms."""
    return call_with_retry(_post_scraper, url)


def scraper_payload_error(scraped: dict):
    if not scraped.get("success"):
        return f"Scraping failed: {scraped.get('error')}"
    name = scraped.get("product_name")
    if name is None or not str(name).strip():
        return "product_name empty"
    ingredients = scraped.get("ingredients")
    if not isinstance(ingredients, list) or len(ingredients) == 0:
        return "ingredients empty"
    return None


@traced("parse_scraper_data")
def parse_scraper_data(scraped: dict, url: str) -> dict:
    """Node 7 — fail if !success; nykaa_url from the loop item, not the scraper body."""
    error = scraper_payload_error(scraped)
    if error:
        raise RuntimeError(error)

    parsed = {"nykaa_url": url}
    for field in _PARSE_FIELDS:
        parsed[field] = scraped.get(field)
    parsed["ingredients"] = scraped.get("ingredients") or []
    parsed["reviews"] = scraped.get("reviews") or []
    return parsed


@traced("scrape_and_parse")
def scrape_and_parse(url: str) -> dict:
    """Call the scraper up to 3 times until the payload check passes."""
    last_error = "scraper payload invalid"
    for run in range(1, SCRAPER_MAX_RUNS + 1):
        scraped = call_python_scraper(url)
        error = scraper_payload_error(scraped)
        if error is None:
            return parse_scraper_data(scraped, url)
        last_error = error
        record_rerun("scraper", run, error)
    raise RuntimeError(last_error)
