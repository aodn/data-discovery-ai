from elasticsearch import Elasticsearch
from typing import Dict, Union, Any

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


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "supervisor"
        self.task_agents = []
        self.model_config = ConfigUtil.get_config().get_supervisor_config().settings

        # add name-task agent map here if more models added in
        self.model_name_class_map = {
            "description_formatting": DescriptionFormattingAgent,
            "keyword_classification": KeywordClassificationAgent,
            "link_grouping": LinkGroupingAgent,
            "delivery_classification": DeliveryClassificationAgent,
        }
        self.tokenizer = None
        self.embedding_model = None

    def set_tokenizer(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def set_embedding_model(self, embedding_model: Any):
        self.embedding_model = embedding_model

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
                    task_agent.set_supervisor(self)
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
        results = [self.run_agent((agent, request)) for agent in self.task_agents]

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

    def process_request_response(self, request: Dict) -> Dict | None:
        """
        Convert JSON request and response into dictionary, so that their combination can be stored in allowed data schema.
        :param request: Dict. The request to call API
        :return: Dict. The combined request and the responses from all task agents. Follows the mapping with required data schema.
        """
        data = {}
        if request is None:
            return None

        data["id"] = request.get("uuid")
        data["title"] = request.get("title", None)
        data["description"] = request.get("abstract", None)
        data["summaries"] = {"statement": request.get("lineage", None)}

        response = self.response
        if response:
            if "themes" in response:
                data["themes"] = response["themes"]
            if "links" in response:
                data["links"] = response["links"]
            if "summaries" in response:
                data["summaries"].update(response["summaries"])

        request_raw = request
        data["ai:request_raw"] = request_raw

        return data

    def search_stored_data(
        self, request: Dict, client: Elasticsearch, index: str
    ) -> Dict[str, Any]:

        uuid = request.get("uuid", None)

        query = {"query": {"term": {"id.keyword": uuid}}}
        resp = client.search(index=index, body=query)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return {}

        # return the first hit
        existing_doc = hits[0]["_source"]
        old_request = existing_doc.get("ai:request_raw", {})
        old_models = set(old_request.get("selected_model", []))
        current_models = set(request.get("selected_model", []))
        if not current_models.issubset(old_models):
            return {}

        model_fields = {
            "link_grouping": self.model_config.get("task_agents")
            .get("link_grouping")
            .get("required_fields"),
            "description_formatting": self.model_config.get("task_agents")
            .get("description_formatting")
            .get("required_fields"),
            "delivery_classification": self.model_config.get("task_agents")
            .get("delivery_classification")
            .get("required_fields"),
            "keyword_classification": self.model_config.get("task_agents")
            .get("keyword_classification")
            .get("required_fields"),
        }

        matched_models = []

        for model in current_models:
            if model not in old_models:
                continue
            required_fields = model_fields.get(model, [])
            if all(request.get(f) == old_request.get(f) for f in required_fields):
                matched_models.append(model)

        if not matched_models:
            return {}

        partial_response = {}

        old_links = existing_doc["links"]
        if "link_grouping" in matched_models and "links" in existing_doc:
            partial_response["links"] = existing_doc["links"]

        if "description_formatting" in matched_models:
            desc = existing_doc.get("summaries", {}).get("ai:description")
            if desc:
                partial_response.setdefault("summaries", {})["ai:description"] = desc

        if "delivery_classification" in matched_models:
            freq = existing_doc.get("summaries", {}).get("ai:update_frequency")
            if freq:
                partial_response.setdefault("summaries", {})[
                    "ai:update_frequency"
                ] = freq

        if "keyword_classification" in matched_models and "themes" in existing_doc:
            partial_response["themes"] = existing_doc["themes"]

        return partial_response
