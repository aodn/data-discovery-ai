from multiprocessing import Pool
from typing import Dict, Union

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai import logger
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.agents.descriptionFormattingAgent import (
    DescriptionFormattingAgent,
)
from data_discovery_ai.agents.keywordClassificationAgent import (
    KeywordClassificationAgent,
)
from data_discovery_ai.agents.linkGroupingAgent import LinkGroupingAgent
from data_discovery_ai.agents.deliveryClassificationAgent import (
    DeliveryClassificationAgent,
)

from data_discovery_ai.config.constants import MAX_PROCESS


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "supervisor"
        self.task_agents = []
        self.model_config = ConfigUtil().get_supervisor_config().settings

        # add name-task agent map here if more models added in
        self.model_name_class_map = {
            "description_formatting": DescriptionFormattingAgent,
            "keyword_classification": KeywordClassificationAgent,
            "link_grouping": LinkGroupingAgent,
            "delivery_classification": DeliveryClassificationAgent,
        }

    def make_decision(self, request: Dict) -> bool:
        """
        Assign the task agents based on the selected models in the request.
        Input: request: Dict. The request from API call
        Output: bool. True if the request is valid and task agents are assigned, False otherwise.
        """
        is_valid = self.is_valid_request(request)
        if is_valid:
            selected_models = request.get("selected_model", [])
            for selected_model in selected_models:
                AgentClass = self.model_name_class_map.get(selected_model)
                if AgentClass:
                    task_agent = AgentClass()
                    task_agent_config = self.model_config["task_agents"].get(
                        selected_model, None
                    )
                    if task_agent_config:
                        task_agent.set_required_fields(
                            task_agent_config["required_fields"]
                        )
                    else:
                        logger.error(f"Agent class not found for {selected_model}.")

                    self.task_agents.append(task_agent)
            return True
        else:
            return False

    @staticmethod
    def run_agent(agent_request_tuple):
        agent, request = agent_request_tuple
        agent.execute(request)
        return agent.response

    def take_action(self, request: Dict) -> Dict:
        """
        Run the task agents in parallel using multiprocessing.
        Input: request: Dict. The request from API call
        Output: Dict. The combined response from all task agents.
        """
        with Pool(processes=MAX_PROCESS) as pool:
            results = pool.map(
                self.run_agent, [(agent, request) for agent in self.task_agents]
            )

        # combine the response from all task agents
        combined_response = {}
        summaries = {}

        for response in results:
            summary_fields = {
                k.split(".")[1]: v
                for k, v in response.items()
                if "summaries" in k and len(k.split(".")) > 1
            }
            other_fields = {k: v for k, v in response.items() if "summaries" not in k}
            if summary_fields:
                summaries.update(summary_fields)

            combined_response.update(other_fields)
            if summaries:
                combined_response["summaries"] = summaries
        return combined_response

    def execute(self, request: Dict) -> None:
        """
        Execute the action module of the Supervisor Agent.
        The agent perceives the request, and make decision based on the received request.
        Input:
            request (dict): The request format.
        """
        flag = self.make_decision(request)
        if flag:
            self.response = self.take_action(request)
        else:
            self.response = {}
        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")

    def is_valid_request(self, request: Dict[str, Union[str, list]]) -> bool:
        """
        Validate request format with supported AI models.
        Check all selected models' required fields are present in the request.
        """
        if not isinstance(request, dict):
            logger.error("Invalid request format: expected a dictionary.")
            return False

        uuid = request.get("uuid", None)
        if uuid is None:
            logger.error("Invalid request format: expected a UUID.")
            return False

        selected_models = request.get("selected_model")
        if not selected_models or not isinstance(selected_models, list):
            logger.error(
                "No selected model found or format is invalid (expected a list)."
            )
            return False

        for model_name in selected_models:
            model_config = self.model_config["task_agents"].get(model_name, None)
            if not model_config:
                available_models = self.model_config["task_agents"]
                logger.error(
                    f"Invalid model name: {model_name}. Choose from {[m for m in available_models]}"
                )
                return False
        return True
