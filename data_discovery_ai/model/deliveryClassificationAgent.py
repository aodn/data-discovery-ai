# The agent-based model for data delivery mode classification task
from data_discovery_ai import logger
from data_discovery_ai.model.baseAgent import BaseAgent
from data_discovery_ai.utils.config_utils import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding, load_from_file
from data_discovery_ai.common.constants import FILTER_FOLDER

from typing import Any, Tuple
from pathlib import Path


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

    def set_required_fields(self, required_fields) -> None:
        return super().set_required_fields(required_fields)

    def is_valid_request(self, request, required_fields):
        return super().is_valid_request(request, required_fields)

    def make_decision(self, request):
        # the complex decision making is done by the es-indexer end, here we simply check if the request is valid
        if self.is_valid_request(request, self.required_fields):
            return True
        else:
            return False

    def take_action(self, title: str, abstract: str, lineage: str) -> str:
        """
        The action module of the Delivery Classification Agent. Its task is to classify the delivery mode based on the provided title, abstract, and lineage.
        """
        # load the model and pca
        pretrained_model, pca = self.load_saved_model()
        if pretrained_model and pca:
            # calculate the embedding of the title, abstract, and lineage
            request_text = (
                title
                + self.model_config["separator"]
                + abstract
                + self.model_config["separator"]
                + lineage
            )
            text_embedding = get_text_embedding(request_text)
            dimension = text_embedding.shape[0]
            target_X = text_embedding.reshape(1, dimension)
            target_X_pca = pca.transform(target_X)

            y_pred = pretrained_model.predict(target_X_pca)
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
            self.response = {self.model_config["response_key"]: ""}
        else:
            title = request["title"]
            abstract = request["abstract"]
            lineage = request["lineage"]
            prediction = self.take_action(title, abstract, lineage)
            self.response = {self.model_config["response_key"]: prediction}

        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")
        self.set_status(2)

    def load_saved_model(self) -> Tuple[Any, Any]:
        pretrained_model_name = self.model_config["pretrained_model"]
        # load model pickle file
        model_file_path = (
            Path(__file__).resolve().parent.parent
            / "resources"
            / FILTER_FOLDER
            / pretrained_model_name
        )
        trained_model = load_from_file(model_file_path.with_suffix(".pkl"))

        # load pca pickle file
        pca = load_from_file(model_file_path.with_suffix(".pca.pkl"))
        return trained_model, pca
