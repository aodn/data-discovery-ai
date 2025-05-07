from typing import Dict, List, Any


class BaseAgent:
    def __init__(self):
        self.type = "base"
        self.response = {}
        self.required_fields = []

    def set_required_fields(self, required_fields: list) -> None:
        """
        Set the required fields for task agents. This is defined in the 'common/constants.py' file and should only be executed by the SupervisorAgent.
        Input:
            required_fields (list): The list of required fields.
        """
        self.required_fields = required_fields

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        """
        Check if the request is valid, i.e., if it contains all required fields. The required field is defined in 'common/constants.py' file.
        Input:
            request (Dict[str, str]): The request format.
        Output:
            bool: True if the request is valid, False otherwise.
        """
        return all(field in request.keys() for field in self.required_fields)

    def make_decision(self, request: Dict[str, str]) -> List[Any] | bool:
        """
        Make decision based on the request. This should be overridden by subclasses.
        """
        pass

    def execute(self, request: Dict[str, str]):
        """
        Execute the agent's task. This should be overridden by subclasses.
        """
        pass
