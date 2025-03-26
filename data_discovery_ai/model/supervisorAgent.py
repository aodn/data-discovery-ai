# the basic agent and supervisor agent class to manage the agent models to cooperate with each other
import os

from data_discovery_ai.common.constants import *
from data_discovery_ai.utils.config_utils import ConfigUtil
import data_discovery_ai.utils.es_connector as connector

from data_discovery_ai.model.baseAgent import BaseAgent
from data_discovery_ai.model.descriptionFormatingModel import (
    DescriptionFormatingAgent as DescriptionFormatingAgent,
)
from data_discovery_ai.model.linkGroupingModel import (
    LinkGroupingAgent as LinkGroupingAgent,
)
from data_discovery_ai.pipeline.pipeline import (
    KeywordClassifierPipeline,
    DataDeliveryModeFilterPipeline,
)


class SupervisorAgent(BaseAgent):
    def __init__(self, environment):
        super().__init__(environment_name=environment_name)
        self.type = "Supervisor"

    def take_action(self, document_id):
        response = {}
        # fetch the record by document ID
        es_config = self.config.load_es_config()
        client = connector.connect_es(es_config)
        index = os.getenv("ES_INDEX_NAME", default=ES_INDEX_NAME)
        resp = connector.fetch_record_by_documentId(
            client=client, index=index, document_id=document_id
        )

        # predict keyword
        # TODO: change keyword model to agent based model
        title = resp["_source"]["title"]
        abstract = resp["_source"]["description"]
        keyword_classifier_pipeline = KeywordClassifierPipeline(
            is_data_changed=False,
            use_pretrained_model=True,
            model_name=self.model_name,
        )
        keyword_classifier_pipeline.pipeline(title=title, abstract=abstract)
        response["predicted_labels"] = keyword_classifier_pipeline.predicted_labels

        # predict data delivery mode
        # TODO: change data delivery mode model to agent based model
        lineage = resp["_source"]["summaries"]["statement"]
        ddm_classifier_pipeline = DataDeliveryModeFilterPipeline(
            is_data_changed=False,
            use_pretrained_model=True,
            model_name=self.model_name,
        )
        ddm_classifier_pipeline.pipeline(
            title=title, abstract=abstract, lineage=lineage
        )
        response["predicted_mode"] = {ddm_classifier_pipeline.predicted_class}

        # update links
        links = resp["_source"]["links"]
        linkGroupingAgent = LinkGroupingAgent()
        grouped_links = linkGroupingAgent.group_links(links)

        if linkGroupingAgent.status == "active":
            response["grouped_links"] = grouped_links
        else:
            response["grouped_links"] = None

        # update description
        description_formatter_agent = DescriptionFormatingAgent(llm_tool="openai")
        formatted_description = description_formatter_agent.take_action(
            title=title, abstract=abstract
        )

        if description_formatter_agent.status == "active":
            response["formatted_description"] = formatted_description
        else:
            response["formatted_description"] = None

        self.response = response
