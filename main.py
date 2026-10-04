from fastapi import FastAPI, HTTPException, Request

from ingestion.orchestrator import run_subflow1
from orchestrator.concern import ConcernAnalysisError
from orchestrator.parser import NO_INGREDIENTS_MESSAGE
from orchestrator.run import run_orchestrator
from precompute.flow1.run import PRODUCT_IDS_ERROR, accept_precompute, parse_product_ids
from tracing import PRODUCT_URL_HEADER, RUN_ID_HEADER, bind_product, start_run

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "SkinGuru agent is running"}


@app.post("/ingestion/run")
def run_ingestion():
    """Node 1 equivalent — manual trigger for Sub-flow 1."""
    return run_subflow1()


@app.post("/recommendation-orchestrator")
async def recommendation_orchestrator(request: Request):
    """Recommendation Orchestrator — waits until all three branches finish."""
    payload = await request.json()
    with start_run(request.headers.get(RUN_ID_HEADER)):
        with bind_product(request.headers.get(PRODUCT_URL_HEADER)):
            try:
                return await run_orchestrator(payload)
            except ValueError as exc:
                if str(exc) == NO_INGREDIENTS_MESSAGE:
                    raise HTTPException(
                        status_code=400, detail=NO_INGREDIENTS_MESSAGE
                    ) from exc
                raise
            except ConcernAnalysisError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/recommendation-precompute")
async def recommendation_precompute(request: Request):
    """Flow 1 node 1 — reply immediately; work runs on a single worker thread."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=PRODUCT_IDS_ERROR) from exc
    try:
        product_ids = parse_product_ids(payload)
    except ValueError as exc:
        if str(exc) == PRODUCT_IDS_ERROR:
            raise HTTPException(status_code=400, detail=PRODUCT_IDS_ERROR) from exc
        raise
    # Enqueue only. Do not process products on this request (worker thread).
    return accept_precompute(
        product_ids,
        request.headers.get(RUN_ID_HEADER),
    )
