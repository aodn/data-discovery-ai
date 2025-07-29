import gzip
import json

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from http import HTTPStatus
from dotenv import load_dotenv
import os
import httpx
import copy
import asyncio
import time

from data_discovery_ai.config.constants import (
    API_PREFIX,
    FILTER_FOLDER,
    KEYWORD_FOLDER,
    KEYWORD_LABEL_FILE,
)
from data_discovery_ai.utils.api_utils import api_key_auth
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.es_connector import (
    store_ai_generated_data,
    delete_es_document,
)
from data_discovery_ai.agents.supervisorAgent import SupervisorAgent

load_dotenv()
router = APIRouter(prefix=API_PREFIX)


async def ensure_ready():
    """
    Service check if the following requirements are met:
      - Pretrained model required resources exist.
      - OpenAI API key is set in production and staging.
      - Ollama server is running in development.
    Raises HTTPException(503) if any check fails.
    """
    config = ConfigUtil.get_config()
    base_path = config.base_dir / "resources"

    # Paths for keyword classification
    keyword_model = config.get_keyword_classification_config().pretrained_model
    keyword_model_path = (base_path / KEYWORD_FOLDER / keyword_model).with_suffix(
        ".keras"
    )
    keyword_label_path = base_path / KEYWORD_FOLDER / KEYWORD_LABEL_FILE

    # Path for delivery classification
    delivery_model = config.get_delivery_classification_config().pretrained_model
    delivery_model_path = (base_path / FILTER_FOLDER / delivery_model).with_suffix(
        ".pkl"
    )

    # Check resource existence
    missing = []
    if not keyword_model_path.exists():
        missing.append(f"Keyword model resource not found at {keyword_model_path}")
    if not keyword_label_path.exists():
        missing.append(f"Keyword label resource not found at {keyword_label_path}")
    if not delivery_model_path.exists():
        missing.append(f"Delivery model resource not found at {delivery_model_path}")
    if missing:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="; ".join(missing)
        )

    # Environment-specific checks
    env = os.getenv("PROFILE", "development")
    if env != "development":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="OpenAI API key not set",
            )
    else:
        ollama_base_url = "http://localhost:11434"
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{ollama_base_url}/models", timeout=2)
        except httpx.RequestError:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="Ollama server not running",
            )


class HealthCheckResponse(BaseModel):
    status_code: int
    status: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health endpoint: always returns a HealthCheckResponse, using status_code and status fields.
    """
    try:
        await ensure_ready()
        return HealthCheckResponse(status_code=HTTPStatus.OK, status="healthy")
    except HTTPException as e:
        return HealthCheckResponse(status_code=e.status_code, status=str(e.detail))


@router.delete(
    "/delete_doc", dependencies=[Depends(api_key_auth), Depends(ensure_ready)]
)
async def delete_doc(request: Request, doc_id: str):
    """
    To delete a document stored in the AI-related Elasticsearch index.
    Input:
        doc_id: the id of the document to delete.
    Output:
        JSONResponse(status_code=HTTPStatus.OK) if deleted successfully.
        JSONResponse(status_code=NOT_FOUND) if not deleted successfully.
    """
    client = request.app.state.client
    index = request.app.state.index

    is_deleted = delete_es_document(doc_id, client, index)
    if is_deleted:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={"message": f"Document {doc_id} deleted."},
        )
    else:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"message": f"Document {doc_id} not found."},
        )


async def event_stream_handler(
    supervisor: SupervisorAgent,
    body: dict,
    client: Elasticsearch,
    index: str,
    max_timeout: int,
    sse_interval: float,
    uuid: str,
    original_request: dict,
    background_tasks: BackgroundTasks,
):
    stored_body, matched_models = supervisor.search_stored_data(
        body, client=client, index=index
    )

    body_selected_models = set(body.get("selected_model", []))
    remaining_models = list(body_selected_models - set(matched_models))

    if not remaining_models:
        yield f"event: done\ndata: {json.dumps(stored_body)}\n\n"
        return

    body["selected_model"] = remaining_models
    start_time = time.time()
    task = asyncio.create_task(asyncio.to_thread(supervisor.execute, body))

    last_sent_sse = 0
    yield f"event: processing\ndata: Start processing record UUID {uuid}...\n\n"

    while not task.done():
        elapsed = time.time() - start_time
        # send timeout error and exit if reach maximum timeout duration
        if elapsed > max_timeout:
            yield f"event: error\ndata: Processing timeout after {max_timeout} seconds.\n\n"
            return
        # send sse message at each interval to keep connection alive
        if elapsed - last_sent_sse >= sse_interval:
            yield f"event: processing\ndata: Processing record UUID {uuid}... elapsed {int(elapsed)}s\n\n"
            last_sent_sse = elapsed
        # add delay to prevent busy waiting
        await asyncio.sleep(0.1)

    await task

    # formatting response to align with data schema
    full_response = copy.deepcopy(stored_body)
    new_response = supervisor.response

    if "summaries" in new_response:
        full_response.setdefault("summaries", {}).update(new_response["summaries"])
    for key in ["links", "themes"]:
        if key in new_response:
            full_response[key] = new_response[key]

    supervisor.response = full_response

    doc = supervisor.process_request_response(original_request)
    # store the raw request and response to AI-related Elasticsearch index
    background_tasks.add_task(
        store_ai_generated_data, data=doc, client=client, index=index
    )

    yield f"event: done\ndata: {json.dumps(full_response)}\n\n"


@router.post(
    "/process_record", dependencies=[Depends(api_key_auth), Depends(ensure_ready)]
)
async def process_record(
    request: Request, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Process a record through the SupervisorAgent with the following logic:
        1. unzip the request if it was compressed
        2. initialise a supervisor agent to execute processing tasks
        3. return BAD_REQUEST if the request is not valid such as missing required fields
        4. search the request in Elasticsearch to see if this record was processed previously
        5. return the processed record if the request are the same
        6. execute processing tasks if the request content changed
        7. return the agent's response as the API call response
    Requires a valid API key and that the service is ready.
    """
    app_config = ConfigUtil.get_config().get_application_config()
    max_timeout = app_config.max_timeout
    sse_interval = app_config.sse_interval

    # receive gzip or json format request, unzip request if it is gzip format
    content_encoding = request.headers.get("Content-Encoding", "").lower()
    try:
        if "gzip" in content_encoding:
            raw_body = await request.body()
            body = json.loads(gzip.decompress(raw_body))
        else:
            body = await request.json()
    except Exception as e:
        error_msg = str(e)

        def error_stream():
            yield f"event: error\ndata: Invalid JSON format. Error {error_msg}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # initialise a supervisor agent to execute request
    uuid = body.get("uuid", "")
    original_request = copy.deepcopy(body)
    client = request.app.state.client
    index = request.app.state.index

    supervisor = SupervisorAgent()
    supervisor.set_tokenizer(request.app.state.tokenizer)
    supervisor.set_embedding_model(request.app.state.embedding_model)

    if not supervisor.is_valid_request(body):

        def error_stream():
            yield "event: error\ndata: Invalid request format\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    stream = event_stream_handler(
        supervisor=supervisor,
        body=body,
        client=client,
        index=index,
        max_timeout=max_timeout,
        sse_interval=sse_interval,
        uuid=uuid,
        original_request=original_request,
        background_tasks=background_tasks,
    )

    return StreamingResponse(stream, media_type="text/event-stream")
