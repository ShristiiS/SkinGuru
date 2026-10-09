# Node 2 — Product urls catalog.
# Hardcoded list, edited in this file when the product set changes.
# Not loaded from a database (Open Item #1 is out of scope for this port).
from tracing import traced

PRODUCT_URLS = [
    "https://www.nykaa.com/the-derma-co-2percent-kojic-acid-face-serum-with-1percent-alpha-arbutin-niacinamide/p/3622184",
    "https://www.nykaa.com/hyphen-dual-phase-advanced-de-pigmentation-serum/p/25233193",
    "https://www.nykaa.com/fixderma-pigment-correcting-face-serum-for-melasma-hyperpigmentation-acne-spots/p/15753792",
    "https://www.nykaa.com/the-derma-co-tran-zelaic-pigmentation-corrector-serum/p/15901858",
    "https://www.nykaa.com/clayco-hyperpigmentation-mushroom-complex-serum/p/18734869",
]


@traced("product_urls_catalog")
def product_urls_catalog() -> list[dict]:
    """Return one {url} item per catalog URL, matching n8n's urls.map(url => ({ json: { url } }))."""
    return [{"url": url} for url in PRODUCT_URLS]
