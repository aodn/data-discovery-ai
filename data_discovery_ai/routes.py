from typing import Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.utils.api_utils import api_key_auth, validate_model_name
from data_discovery_ai.pipeline import KeywordClassifierPipeline

router = APIRouter(prefix=API_PREFIX)
logger = logging.getLogger(__name__)


class PredictKeywordRequest(BaseModel):
    selected_model: str = "default"  # Default value
    raw_input: str  # Required field


@router.get("/hello", dependencies=[Depends(api_key_auth)])
async def hello():
    logger.info("hello endpoint is called")
    return {"content": "Hello World!"}


@router.post("/predict", dependencies=[Depends(api_key_auth)])
async def predict_keyword(payload: PredictKeywordRequest) -> dict[str, str]:
    # selected_model = validate_model_name(payload.selected_model)
    keyword_classifier_pipeline = KeywordClassifierPipeline(
        isDataChanged=False, usePretrainedModel=True, model_name=payload.selected_model
    )
    logger.info(
        f"selected_model: {payload.selected_model}, raw_input: {payload.raw_input}"
    )
    predicted_labels = keyword_classifier_pipeline.make_prediction(payload.raw_input)
    response = {"predicted_labels": predicted_labels}
    return response
