# Node 2 — Product urls catalog.
# Hardcoded list, edited in this file when the product set changes.
# Not loaded from a database (Open Item #1 is out of scope for this port).
from tracing import traced

PRODUCT_URLS = [
    "https://www.nykaa.com/cetaphil-advanced-recovery-serum/p/25165279?productId=25165279&pps=2",
    "https://www.nykaa.com/kay-beauty-rich-milky-essence/p/24369007?productId=24369007&pps=4",
    "https://www.nykaa.com/minimalist-10percent-vitamin-c-serum-for-face-for-illuminating-skin-for-beginners/p/15022070?productId=15022070&pps=11&skuId=1068002",
]


@traced("product_urls_catalog")
def product_urls_catalog() -> list[dict]:
    """Return one {url} item per catalog URL, matching n8n's urls.map(url => ({ json: { url } }))."""
    return [{"url": url} for url in PRODUCT_URLS]
