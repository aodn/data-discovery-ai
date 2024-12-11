from fastapi import APIRouter, Depends
from pydantic import BaseModel
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.utils.api_utils import api_key_auth, validate_model_name
from data_discovery_ai.pipeline.pipeline import (
    KeywordClassifierPipeline,
)
from data_discovery_ai import logger


router = APIRouter(prefix=API_PREFIX)


class PredictKeywordRequest(BaseModel):
    selected_model: str
    title: str
    abstract: str


class HealthCheckResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    response = HealthCheckResponse(status="healthy")
    return response


@router.post("/predict", dependencies=[Depends(api_key_auth)])
async def predict_keyword(payload: PredictKeywordRequest):
    # selected_model = validate_model_name(payload.selected_model)
    keyword_classifier_pipeline = KeywordClassifierPipeline(
        is_data_changed=False,
        use_pretrained_model=True,
        model_name=payload.selected_model,
    )
    logger.info(
        f"selected_model: {payload.selected_model}, title: {payload.title}, abstract: {payload.abstract}"
    )
    keyword_classifier_pipeline.pipeline(title=payload.title, abstract=payload.abstract)
    response = {"predicted_labels": keyword_classifier_pipeline.predicted_labels}
    return response
