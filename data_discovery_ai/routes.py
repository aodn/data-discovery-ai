from fastapi import APIRouter, Depends
from pydantic import BaseModel
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.utils.api_utils import api_key_auth
from data_discovery_ai import logger


router = APIRouter(prefix=API_PREFIX)


class HealthCheckResponse(BaseModel):
    status_code: int
    status: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    # if the pretrained models are not available, and they are on call, the service should be unavailable
    # TODO
    response = HealthCheckResponse(status="healthy")
    return response
