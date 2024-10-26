from typing import Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.service.keywordClassifier import keywordClassifier
from data_discovery_ai.utils.api_utils import api_key_auth, validate_model_name

router = APIRouter(prefix=API_PREFIX)
logger = logging.getLogger(__name__)


class PredictKeywordRequest(BaseModel):
    selected_model: str = "default"  # Default value
    raw_input: str  # Required field


@router.get("/hello", dependencies=[Depends(api_key_auth)])
async def hello():
    logger.info("hello endpoint is called")
    return {"content": "Hello World!"}


@router.post("/predict-keyword", dependencies=[Depends(api_key_auth)])
async def predict_keyword(payload: PredictKeywordRequest) -> dict[str, str]:
    # TODO: just placeholder for now, the client where calling this endpoint should only know 2 things: selected
    #  model name, and the raw input
    selected_model = validate_model_name(payload.selected_model)
    raw_input = payload.raw_input
    logger.info(f"selected_model: {selected_model}, raw_input: {raw_input}")
    # predicted_keyword = keywordClassifier(None, None, None, None, None)
    response = {"predicted_keyword": "sample_predicted_keyword"}
    return response
