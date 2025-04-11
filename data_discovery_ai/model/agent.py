import os
from typing import Dict, List, Union, Any
from data_discovery_ai import logger
import dotenv

from data_discovery_ai.common.constants import API_VALID_FIELD, AVAILABLE_AI_MODELS
from data_discovery_ai import logger


# TODO: consolidate the agent classes into one single environment
class BaseAgent:
    def __init__(self):
        self.type = "base"
        self.id = None
        # 0 as inactivate, 1 as active, 2 as finished
        self.status = 0

    def set_status(self, status: int):
        self.status = status


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "supervisor"
        self.set_status(1)

    def make_decision(self, request: Dict):
        # distribute tasks to other agents based on the perceived metadata
        # TODO
        # execute only if the metadata has at least one of the valid fields
        if len(is_valid_request(request)) > 0:
            self.set_status(1)


def is_valid_request(request: Dict[str, Union[str, List]]) -> List[str]:
    """
    Validate request format with supported fields.
    Input:
        request (Dict[str, Union[str, List]]): The request format.
    Output:
        List[str]: matched valid fields in the requet
    """
    if type(request) is not dict:
        logger.error("Invalid request")
        return []
    else:
        if "selected_model" not in request.keys():
            logger.error("No selected model found in the request")
            return []
        else:
            selected_model = request["selected_model"]
            if selected_model not in AVAILABLE_AI_MODELS:
                logger.error(
                    f"Invalid model name: {selected_model}. Please choose from {AVAILABLE_AI_MODELS}"
                )
                return []

        matched_fields = []
        for field in request.keys():
            if field in API_VALID_FIELD:
                matched_fields.append(field)
        return matched_fields
