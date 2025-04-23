from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from http import HTTPStatus
import json

from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.utils.api_utils import api_key_auth
from data_discovery_ai import logger
from data_discovery_ai.utils.config_utils import ConfigUtil
from data_discovery_ai.common.constants import FILTER_FOLDER, KEYWORD_FOLDER
from data_discovery_ai.model.supervisorAgent import SupervisorAgent

router = APIRouter(prefix=API_PREFIX)


class HealthCheckResponse(BaseModel):
    status_code: int
    status: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    # if the pretrained models are not available, the service should be unavailable
    config = ConfigUtil()
    base_path = config.base_dir / "resources"
    keyword_default_model = config.get_keyword_classification_config().get(
        "pretrained_model"
    )
    delivery_default_model = config.get_delivery_classification_config().get(
        "pretrained_model"
    )

    keyword_model_path = (
        base_path / KEYWORD_FOLDER / keyword_default_model
    ).with_suffix(".keras")
    delivery_model_path = (
        base_path / FILTER_FOLDER / delivery_default_model
    ).with_suffix(".pkl")
    if not keyword_model_path.exists() or not delivery_model_path.exists():
        return HealthCheckResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            status="Service Unavailable: Pretrained models not found",
        )
    else:
        return HealthCheckResponse(status_code=HTTPStatus.OK, status="healthy")


@router.post("/process_record", dependencies=[Depends(api_key_auth)])
async def process_record(request: Request) -> None:
    body = await request.json()
    logger.info("Request details: %s", body)
    supervisor = SupervisorAgent()
    if not supervisor.is_valid_request(body):
        return JSONResponse(
            content={"Error": "Invalid request format."},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    else:
        supervisor.execute(body)
        return JSONResponse(
            content=supervisor.response,
            status_code=HTTPStatus.OK,
        )
