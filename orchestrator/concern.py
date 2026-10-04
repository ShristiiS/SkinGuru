import asyncio

from clients.supabase import mark_concerns_analyzed
from config import CONCERN_ANALYSIS_TIMEOUT_SECONDS, CONCERN_ANALYSIS_WAIT_SECONDS
from orchestrator.n8n import post_n8n_webhook
from tracing import traced


class ConcernAnalysisError(Exception):
    """Raised when the Concern Analysis branch fails. Chunk E will cancel other tasks."""


def _trigger_body(ingredient_ids: list[dict]) -> list[dict]:
    return [
        {
            "canonical_name": item["name"],
            "ingredient_id": item["id"],
            "evidence_count": 0,
        }
        for item in ingredient_ids
    ]


@traced("run_concern_analysis")
async def run_concern_analysis(worklist: dict) -> dict:
    """Nodes 4–6: one n8n trigger for the whole list, then mark RPC, then 60s wait."""
    ingredient_ids = worklist["ingredient_ids"]
    body = _trigger_body(ingredient_ids)
    p_names = [item["name"] for item in ingredient_ids]

    try:
        await post_n8n_webhook(
            "concern-analysis-on-demand",
            body,
            CONCERN_ANALYSIS_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise ConcernAnalysisError(
            f"Trigger Concern Analysis failed: {exc}"
        ) from exc

    try:
        await asyncio.to_thread(mark_concerns_analyzed, p_names)
    except Exception as exc:
        raise ConcernAnalysisError(
            f"Mark Concerns Analyzed failed: {exc}"
        ) from exc

    await asyncio.sleep(CONCERN_ANALYSIS_WAIT_SECONDS)
    return {"concern_triggered": True, "concerns_marked": True}
