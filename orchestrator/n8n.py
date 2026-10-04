import httpx

from config import N8N_WEBHOOK_BASE
from tracing import record_http, trace_step

N8N_SKIP_BROWSER_WARNING = {"ngrok-skip-browser-warning": "true"}


async def post_n8n_webhook(path: str, body, timeout_seconds: float) -> httpx.Response:
    """POST http://host.docker.internal:5111/webhook/webhook/<path> (doubled path kept)."""
    url = f"{N8N_WEBHOOK_BASE}/{path.lstrip('/')}"
    with trace_step(f"n8n.{path}"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=body,
                headers=N8N_SKIP_BROWSER_WARNING,
                timeout=timeout_seconds,
            )
            record_http(
                response.status_code,
                url=url,
                method="POST",
                request_body=body,
                response_body=response.text,
            )
            response.raise_for_status()
            return response
