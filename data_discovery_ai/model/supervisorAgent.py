# the basic agent and supervisor agent class to manage the agent models to cooperate with each other
import os

from data_discovery_ai.common.constants import *
from data_discovery_ai.utils.config_utils import ConfigUtil
import data_discovery_ai.utils.es_connector as connector
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


class BaseAgent:
    def __init__(self, model_name: str):
        self.type = "Base"
        self.status = "active"
        self.model_name = model_name
        if not self.is_valid_model():
            raise ValueError(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )
        self.config = ConfigUtil()
        self.response = None

    """
        Validate model name within fixed selections
        Input:
            model_name: str. The file name of the saved model.
    """

    def is_valid_model(self) -> bool:
        valid_model_name = AVAILABLE_MODELS
        self.model_name = self.model_name.lower()
        if self.model_name in valid_model_name:
            return True
        else:
            return False


class SupervisorAgent(BaseAgent):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        self.type = "Supervisor"
        if not self.is_valid_model():
            raise ValueError(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )

    def perceive_environment(self):
        # TODO: This is a placeholder for the integration of the es-indexer. This function should be able to compare whether the document is changed or not. A newly indexed record is treated as a changed document. It should be able to fetch all recent records and compare with the past stored in its memory, and update its memory accordingly.
        # Or we don't need this function here, but do the search at the es-indexer side, so that we only call the take_action function when the document is changed.
        pass

    def make_decision(self):
        # TODO: This is a placeholder for the integration of the es-indexer. This function should be able to make a decision based on the perceived environment. So that to decide whether the perceived records are significantly changed or not, if so, it should take action to update the record with distributed agents. It should return a list of ducument IDs pending for actions.
        # Or we don't need this function here, but do the search at the es-indexer side, so that we only call the take_action function when the document is changed.
        pass

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
