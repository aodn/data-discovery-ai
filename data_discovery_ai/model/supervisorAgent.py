from multiprocessing import Pool
from typing import Dict, Union

from data_discovery_ai.common.constants import AVAILABLE_AI_AGENT
from data_discovery_ai import logger
from data_discovery_ai.model.baseAgent import BaseAgent
from data_discovery_ai.model.descriptionFormattingAgent import (
    DescriptionFormattingAgent,
)
from data_discovery_ai.model.keywordClassificationAgent import (
    KeywordClassificationAgent,
)
from data_discovery_ai.model.linkGroupingAgent import LinkGroupingAgent

# TODO: import the rest of models

from data_discovery_ai.common.constants import MAX_PROCESS


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "supervisor"
        # set status to 1 as active
        self.set_status(1)
        self.task_agents = []

        self.model_name_class_map = {
            "description_formatting": DescriptionFormattingAgent,
            "keyword_classification": KeywordClassificationAgent,
            "link_grouping": LinkGroupingAgent,
        }

    def make_decision(self, request: Dict):
        """
        Assign the task agents based on the selected models in the request.
        Input: request: Dict. The request from API call
        """
        is_valid = self.is_valid_request(request)
        if is_valid:
            selected_models = request.get("selected_model", [])
            for selected_model in selected_models:
                AgentClass = self.model_name_class_map.get(selected_model)
                if AgentClass:
                    task_agent = AgentClass()
                    task_agent_config = next(
                        (m for m in AVAILABLE_AI_AGENT if m["model"] == selected_model),
                        None,
                    )
                    if task_agent_config:
                        task_agent.set_required_fields(
                            task_agent_config["required fields"]
                        )

                        logger.debug(task_agent.required_fields)
                    else:
                        logger.error(f"Agent class not found for {selected_model}.")

                    self.task_agents.append(task_agent)

    @staticmethod
    def run_agent(agent_request_tuple):
        agent, request = agent_request_tuple
        agent.execute(request)
        return agent.response

    def take_action(self, request: Dict):
        """
        Run the task agents in parallel using multiprocessing.
        Input: request: Dict. The request from API call
        """
        with Pool(processes=MAX_PROCESS) as pool:
            results = pool.map(
                self.run_agent, [(agent, request) for agent in self.task_agents]
            )
        self.repsonse = results
        # set as finished status
        self.set_status(2)

    def is_valid_request(self, request: Dict[str, Union[str, list]]) -> bool:
        """
        Validate request format with supported AI models.
        Check all selected models' required fields are present in the request.
        """
        if not isinstance(request, dict):
            logger.error("Invalid request format: expected a dictionary.")
            return False

        selected_models = request.get("selected_model")
        if not selected_models or not isinstance(selected_models, list):
            logger.error(
                "No selected model found or format is invalid (expected a list)."
            )
            return False

        for model_name in selected_models:
            model_config = next(
                (m for m in AVAILABLE_AI_AGENT if m["model"] == model_name), None
            )
            if not model_config:
                logger.error(
                    f"Invalid model name: {model_name}. Choose from {[m['model'] for m in AVAILABLE_AI_AGENT]}"
                )
                return False
        return True
