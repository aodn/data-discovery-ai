import gzip
import json

from fastapi import APIRouter, Depends, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from http import HTTPStatus
from dotenv import load_dotenv
import os
import httpx

from data_discovery_ai.config.constants import (
    API_PREFIX,
    FILTER_FOLDER,
    KEYWORD_FOLDER,
    KEYWORD_LABEL_FILE,
)
from data_discovery_ai.utils.api_utils import api_key_auth
from data_discovery_ai import logger
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.es_connector import store_ai_generated_data
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
        # Return HealthCheckResponse with appropriate code and message
        return HealthCheckResponse(status_code=e.status_code, status=str(e.detail))


@router.post(
    "/process_record", dependencies=[Depends(api_key_auth), Depends(ensure_ready)]
)
async def process_record(
    request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    Process a record through the SupervisorAgent.
    Requires a valid API key and that the service is ready.
    """
    content_encoding = request.headers.get("Content-Encoding", "").lower()
    if "gzip" in content_encoding:
        raw_body = await request.body()
        try:
            decompressed_data = gzip.decompress(raw_body)
            body = json.loads(decompressed_data)
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Invalid gzip format"
            )
    else:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Invalid JSON format"
            )

    logger.info("Request details: %s", body)

    supervisor = SupervisorAgent()
    supervisor.set_tokenizer(request.app.state.tokenizer)
    supervisor.set_embedding_model(request.app.state.embedding_model)

    if not supervisor.is_valid_request(body):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid request format."
        )

    supervisor.execute(body)

    # store to Elasticsearch
    doc = supervisor.process_request_response(body)
    client = request.app.state.client
    index = request.app.state.index
    background_tasks.add_task(
        func=store_ai_generated_data, data=doc, client=client, index=index
    )

    return JSONResponse(content=supervisor.response, status_code=HTTPStatus.OK)
