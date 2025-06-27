# The agent-based model for data delivery mode classification task
from data_discovery_ai import logger
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding, load_from_file
from data_discovery_ai.config.constants import FILTER_FOLDER

from typing import Any, Tuple, Dict


class DeliveryClassificationAgent(BaseAgent):
    """
    The agent-based model for data delivery mode classification task.
    This agent is responsible for classifying the delivery mode of a given dataset based on its title and abstract.
    """

    def __init__(self):
        super().__init__()
        self.type = "delivery_classification"
        self.config = ConfigUtil()
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
            class_map = {0: "Real-Time", 1: "Delayed", 2: "Other"}
            pred_class = class_map.get(y_pred[0])
            return pred_class
        else:
            return ""

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
            title = request["title"]
            abstract = request["abstract"]
            lineage = request["lineage"]
            prediction = self.take_action(title, abstract, lineage)
            self.response = {self.model_config.response_key: prediction}

        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")

    def load_saved_model(self) -> Any:
        pretrained_model_name = self.model_config.pretrained_model
        # load model pickle file
        model_file_path = (
            self.config.base_dir / "resources" / FILTER_FOLDER / pretrained_model_name
        )
        trained_model = load_from_file(model_file_path.with_suffix(".pkl"))
        return trained_model
