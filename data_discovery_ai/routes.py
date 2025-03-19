from fastapi import APIRouter, Depends
from pydantic import BaseModel
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.utils.api_utils import api_key_auth, validate_model_name
from data_discovery_ai.pipeline.pipeline import (
    KeywordClassifierPipeline,
    DataDeliveryModeFilterPipeline,
)
from data_discovery_ai.model.linkGroupingModel import LinkGroupingAgent
from data_discovery_ai.model.descriptionFormatingModel import DescriptionFormatingAgent
from data_discovery_ai.model.supervisorAgent import SupervisorAgent
from typing import List, Dict
from data_discovery_ai import logger


router = APIRouter(prefix=API_PREFIX)


class PredictKeywordRequest(BaseModel):
    selected_model: str
    title: str
    abstract: str


class PredictDataDeliveryModeRequest(BaseModel):
    selected_model: str
    title: str
    abstract: str
    lineage: str


class LinkGroupingRequest(BaseModel):
    links: List[Dict[str, str]]


class DescriptionFormatterRequest(BaseModel):
    selected_model: str
    title: str
    abstract: str

class ProcessRecordRequest(BaseModel):
    selected_model: str
    document_id: str

class HealthCheckResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    response = HealthCheckResponse(status="healthy")
    return response


@router.post("/predict", dependencies=[Depends(api_key_auth)])
async def predict_keyword(payload: PredictKeywordRequest):
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


@router.post("/datadeliverymode", dependencies=[Depends(api_key_auth)])
async def predict_data_delivery_mode(payload: PredictDataDeliveryModeRequest):
    ddm_classifier_pipeline = DataDeliveryModeFilterPipeline(
        is_data_changed=False,
        use_pretrained_model=True,
        model_name=payload.selected_model,
    )
    logger.info(
        f"selected_model: {payload.selected_model}, title: {payload.title}, abstract: {payload.abstract}, lineage: {payload.lineage}"
    )
    ddm_classifier_pipeline.pipeline(
        title=payload.title, abstract=payload.abstract, lineage=payload.lineage
    )
    response = {"predicted_mode": ddm_classifier_pipeline.predicted_class}
    return response


@router.post("/grouplinks", dependencies=[Depends(api_key_auth)])
async def group_links(payload: LinkGroupingRequest):
    links = payload.links
    link_grouping_agent = LinkGroupingAgent()
    grouped_links = link_grouping_agent.group_links(links)
    response = {"grouped_links": grouped_links}
    return response


@router.post("/descriptionformatter", dependencies=[Depends(api_key_auth)])
async def format_description(payload: DescriptionFormatterRequest):
    description_formatter_agent = DescriptionFormatingAgent(
        llm_tool=payload.selected_model
    )
    formatted_description = description_formatter_agent.take_action(
        title=payload.title, abstract=payload.abstract
    )
    response = {"formatted_description": formatted_description}
    return response

@router.post("/processrecord", dependencies=[Depends(api_key_auth)])
async def process_record(payload: ProcessRecordRequest):
    selected_model = payload.selected_model
    document_id = payload.document_id
    supervisor = SupervisorAgent(selected_model)
    supervisor.take_action(document_id=document_id)
    response = supervisor.response
    return response