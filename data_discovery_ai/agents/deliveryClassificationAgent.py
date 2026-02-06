# The agent-based model for data delivery mode classification task, the classes include: {completed, real-time, delayed, other}
import structlog
from enum import Enum

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding, load_from_file
from data_discovery_ai.config.constants import FILTER_FOLDER
from data_discovery_ai.enum.agent_enums import AgentType

from typing import Any, Dict

logger = structlog.get_logger(__name__)


class UpdateFrequency(Enum):
    completed = "completed"
    real_time = "real-time"
    delayed = "delayed"
    other = "other"


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
        # the complex decision-making is done by the es-indexer end, here we simply check if the request is valid
        return self.is_valid_request(request)

    def take_action(self, title: str, abstract: str, lineage: str) -> str:
        """
        The action module of the Delivery Classification Agent. Its task is to classify the delivery mode based on the provided title, abstract, and lineage.
        """
        # load the model
        pretrained_model = self.load_saved_model()
        if pretrained_model:
            # calculate the embedding of the title, abstract, and lineage
            request_text = (
                title
                + self.model_config.separator
                + abstract
                + self.model_config.separator
                + lineage
            )
            tokenizer = self.supervisor.tokenizer
            embedding_model = self.supervisor.embedding_model
            text_embedding = get_text_embedding(
                request_text, tokenizer, embedding_model
            )
            dimension = text_embedding.shape[0]
            target_X = text_embedding.reshape(1, dimension)

            y_pred = pretrained_model.predict(target_X)
            class_map = {
                0: UpdateFrequency.real_time.value,
                1: UpdateFrequency.delayed.value,
                2: UpdateFrequency.other.value,
            }
            pred_class = class_map.get(y_pred[0])
            return pred_class
        else:
            return UpdateFrequency.other.value

    def execute(self, request: dict) -> None:
        """
        Execute the action module of the Delivery Classification Agent. The action is to classify the delivery mode based on the provided request.
        The agent perceives the request, and make decision based on the received request. If it decides to take action, it will call the LLM module to classify the delivery mode and set self response as the classified delivery mode.
        Otherwise, it will set self.response as an empty string.
        Input:
            request (dict): The request format.
        """
        flag = self.make_decision(request)
        if not flag:
            self.response = {self.model_config.response_key: ""}
        else:
            status = request["status"]
            temporal = request["temporal"]
            mapped_update_frequency = map_status_update_frequency(status, temporal)
            if mapped_update_frequency == UpdateFrequency.other.value:
                title = request["title"]
                abstract = request["abstract"]
                lineage = request["lineage"]
                prediction = self.take_action(title, abstract, lineage)
                self.response = {self.model_config.response_key: prediction}
            else:
                self.response = {
                    self.model_config.response_key: mapped_update_frequency
                }

        logger.debug(f"{self.type} agent finished, it responses: \n {self.response}")

    def load_saved_model(self) -> Any:
        pretrained_model_name = self.model_config.pretrained_model
        # load model pickle file
        model_file_path = (
            self.config.base_dir / "resources" / FILTER_FOLDER / pretrained_model_name
        )
        trained_model = load_from_file(model_file_path.with_suffix(".pkl"))
        return trained_model


def map_status_update_frequency(status: str, temporal: list) -> str:
    """
    Input:
        - status: str - the status of the record, such as "completed", "onGoing", can have free text in one or few words.
        - temporal: List<Map>, for example:
        "temporal": [
            {
                "start": "2023-01-22T13:00:00Z",
                "end": "2023-01-23T12:59:59Z"
            },
            {}
        ]
    Output:
        str: the mapped update frequency in terms of "completed" or "other"
    Given status and temporal range of the record, decide update_frequency based on a set of predefined rules
    (see: https://github.com/aodn/backlog/issues/7978#issuecomment-3821164737). Specifically:
    1. these statuses could also be regarded as 'completed'
        historicalArchive
        obsolete
        deprecated
        complete
        Complete
    2. onGoing | historicalArchive - records have 2 x status identifed, 'ongoing' is priority
    3. Under development = check has date range. Yes = completed
        Planned = check has date range. Yes = completed
        Tentative = check has date range. Yes = completed
        No status = check has date range. Yes = completed (likely they are completed)
    Set update_frequency to 'completed' if meet rules or 'other' if not meet
    """
    if status is None:
        status = ""
    normalised_status = status.replace(" ", "").lower()
    # rule 2: check ongoing priority first
    if "ongoing" in normalised_status:
        return UpdateFrequency.other.value

    # rule 1: check completed status with its variants
    completed_status = ["historicalarchive", "obsolete", "deprecated", "complete"]
    if normalised_status in completed_status:
        return UpdateFrequency.completed.value

    # rule 3: check free text status with temporal range
    free_text_status = ["underdevelopment", "planned", "tentative", ""]
    if normalised_status in free_text_status:
        for temporal_entry in temporal:
            if temporal_entry.get("start") and temporal_entry.get("end"):
                return UpdateFrequency.completed.value

    # default to 'other'
    return UpdateFrequency.other.value
