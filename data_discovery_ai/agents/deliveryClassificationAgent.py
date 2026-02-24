# The agent-based model for data delivery mode classification task, the classes include: {completed, real-time, delayed, other}
import structlog

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import AgentType
from data_discovery_ai.enum.delivery_mode_enum import UpdateFrequency
from data_discovery_ai.ml.filteringModel import (
    mapping_update_frequency,
    DeliveryModeInferencer,
    extract_evidence,
)

from typing import Dict

logger = structlog.get_logger(__name__)


class DeliveryClassificationAgent(BaseAgent):
    """
    The agent-based model for data delivery mode classification task.
    This agent is responsible for classifying the delivery mode of a given dataset based on its title and abstract.
    """

    def __init__(self):
        super().__init__()
        self.type = AgentType.DELIVERY_CLASSIFICATION.value
        self.config = ConfigUtil.get_config()
        self.model_config = self.config.get_delivery_classification_config()
        self.supervisor = None

    def set_supervisor(self, supervisor):
        self.supervisor = supervisor

    def set_required_fields(self, required_fields) -> None:
        return super().set_required_fields(required_fields)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return super().is_valid_request(request)

    def make_decision(self, request) -> bool:
        # check if the request is valid
        return self.is_valid_request(request)

    def take_action(self, title: str, abstract: str, lineage: str) -> str:
        """
        The action module of the Delivery Classification Agent. Its task is to classify the delivery mode based on the provided title, abstract, and lineage.
        """
        nli_model = self.supervisor.nli_model
        nli_tokenizer = self.supervisor.nli_tokenizer
        # initialize the inference model
        infer_model = DeliveryModeInferencer(nli_tokenizer, nli_model)
        # extract evidence sentences
        evidence = extract_evidence(title, abstract, lineage)
        # calculate probabilities to each case. the example structure is a dict, such as:
        # {'mode': 'REAL_TIME', 'reason': 'nli_entails_real_time', 'evidence': Evidence(rt=['This dataset is delivered in real-time.'], delayed=['The silk is removed from the CPR cassette and processed as described in Richardson et al 2006.'], rt_unprocessed=[]), 'nli': {'ent_rt': 0.9275655746459961, 'ent_dl': 0.14127250015735626, 'rt_scores': [0.9275655746459961], 'dl_scores': [0.14127250015735626]}}
        probs = infer_model.decide_with_nli(evidence)
        # process 'UNKNOWN' prediction to default 'other' mode
        if probs["mode"] == UpdateFrequency.UNKNOWN.value:
            return UpdateFrequency.OTHER.value
        # process 'CONFLICT' prediction (indication to both high probability of being delayed and real-time to 'DELAYED` mode
        if probs["mode"] == UpdateFrequency.BOTH.value:
            return UpdateFrequency.DELAYED.value
        else:
            # return inferred real-time or delayed
            return probs["mode"]

    def execute(self, request: dict) -> None:
        """
        Execute the action module of the Delivery Classification Agent. The action is to classify the delivery mode based on the provided request.
        The agent perceives the request, and make decision based on the received request. If it decides to take action, it will call the LLM module to classify the delivery mode and set self response as the classified delivery mode.
        Otherwise, it will set self.response as an empty string.
        Input:
            request (dict): The request format.
        """
        # from es-indexer end, if these fields have empty values, the request will have no such fields. But from the data-discovery-ai side, these fields are required for the model to make decision.
        # so we set default empty values for these fields to catch null error
        request.setdefault("status", "")
        request.setdefault("lineage", "")
        request.setdefault("temporal", [])
        flag = self.make_decision(request)
        if not flag:
            self.response = {self.model_config.response_key: ""}
        else:
            status = request["status"]
            temporal = request["temporal"]
            title = request["title"]
            mapped_update_frequency = mapping_update_frequency(status, temporal, title)
            # only conduct language inference if update frequency is 'other'
            if mapped_update_frequency == UpdateFrequency.OTHER.value:
                abstract = request["abstract"]
                lineage = request["lineage"]
                prediction = self.take_action(title, abstract, lineage)
                self.response = {self.model_config.response_key: prediction}
            else:
                self.response = {
                    self.model_config.response_key: mapped_update_frequency
                }

        logger.debug(f"{self.type} agent finished, it responses: \n {self.response}")
