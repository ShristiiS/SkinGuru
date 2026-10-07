import os

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
APP_PORT = int(os.environ.get("APP_PORT", "8080"))
SCRAPER_URL = os.environ.get("SCRAPER_URL", "http://host.docker.internal:8000/scrape")
SCRAPER_TIMEOUT_SECONDS = 600
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_TIMEOUT_SECONDS = 180

# Sub-flow 1 POSTs to this service's orchestrator (Chunk F), not n8n.
# Override if the app is not reachable at 127.0.0.1:{APP_PORT}.
ORCHESTRATOR_WEBHOOK_URL = os.environ.get(
    "ORCHESTRATOR_WEBHOOK_URL",
    f"http://127.0.0.1:{APP_PORT}/recommendation-orchestrator",
)
PMC_WEBHOOK_URL = "http://host.docker.internal:5111/webhook/europe-pmc-on-demand"
PUBCHEM_WEBHOOK_URL = (
    "http://host.docker.internal:5111/webhook/3269c5d1-9023-425e-83f0-d6a1add2dd7b"
)
PRECOMPUTE_WEBHOOK_URL = (
    "http://host.docker.internal:5111/webhook/recommendation-precompute"
)
PMC_PUBCHEM_TIMEOUT_SECONDS = 900
ORCHESTRATOR_TIMEOUT_SECONDS = 1800
PRECOMPUTE_TIMEOUT_SECONDS = 1800

# Recommendation Orchestrator → n8n small workflows (PORT DECISIONS #3+#4).
N8N_WEBHOOK_BASE = "http://host.docker.internal:5111/webhook/webhook"
CONCERN_ANALYSIS_URL = f"{N8N_WEBHOOK_BASE}/concern-analysis-on-demand"
CONCERN_ANALYSIS_TIMEOUT_SECONDS = 300
CONCERN_ANALYSIS_WAIT_SECONDS = 60
LLM_ENRICHMENT_URL = f"{N8N_WEBHOOK_BASE}/llm-enrichment-on-demand"
LLM_ENRICHMENT_TIMEOUT_SECONDS = 120
LLM_BATCH_SIZE = 5
# Node 18 Wait amount is not in the n8n export. 1s is the common unconfigured
# default; unconfirmed against the live n8n Wait node.
LLM_INTER_BATCH_WAIT_SECONDS = 1
FUNCTIONAL_CATEGORY_URL = f"{N8N_WEBHOOK_BASE}/functional-category-on-demand"
FUNCTIONAL_CATEGORY_TIMEOUT_SECONDS = 60
FUNCTIONAL_INTER_BATCH_WAIT_SECONDS = 20
# Per-ingredient / concern-trigger retries in the Recommendation Orchestrator.
# 1 initial attempt + 5 retries; 5s between attempts of the same work.
ORCHESTRATOR_MAX_ATTEMPTS = 6
ORCHESTRATOR_RETRY_WAIT_SECONDS = 5
# Flow 1 call retry: original + 2; 5s between tries. Used by Flow 1
# OpenAI / Supabase / SerpAPI wrappers. Not used by ingestion or Orchestrator.
FLOW1_CALL_RETRY_ATTEMPTS = 3
FLOW1_CALL_RETRY_WAIT_SECONDS = 5


def require_supabase_config() -> tuple[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return SUPABASE_URL, SUPABASE_KEY


def require_openai_config() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set")
    return OPENAI_API_KEY


SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
SERPAPI_TIMEOUT_SECONDS = 180


def require_serpapi_config() -> str:
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY must be set")
    return SERPAPI_KEY


# USD per 1M tokens. Source: https://developers.openai.com/api/docs/models/gpt-4o-mini
# (official OpenAI pricing; openai.com/api/pricing timed out) — 2026-09-24
# gpt-4.1-mini: https://developers.openai.com/api/docs/models/gpt-4.1-mini — 2026-09-25
# gpt-4.1: https://developers.openai.com/api/docs/models/gpt-4.1 — 2026-09-25
# gpt-5.4-mini: https://developers.openai.com/api/docs/models/gpt-5.4-mini — 2026-09-25
# gpt-5.4-nano: https://developers.openai.com/api/docs/models/gpt-5.4-nano — 2026-09-25
PRICING = {
    "gpt-4o-mini": {
        "prompt_per_1m": 0.15,
        "completion_per_1m": 0.60,
    },
    "gpt-4.1-mini": {
        "prompt_per_1m": 0.40,
        "completion_per_1m": 1.60,
    },
    "gpt-4.1": {
        "prompt_per_1m": 2.00,
        "completion_per_1m": 8.00,
    },
    "gpt-5.4-mini": {
        "prompt_per_1m": 0.75,
        "completion_per_1m": 4.50,
    },
    "gpt-5.4-nano": {
        "prompt_per_1m": 0.20,
        "completion_per_1m": 1.25,
    },
    "gpt-5.6-luna": {
        # TODO real price
        "prompt_per_1m": 0.0,
        "completion_per_1m": 0.0,
    },
}
