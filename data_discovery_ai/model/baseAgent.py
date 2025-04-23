from typing import Dict


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

    def is_valid_request(self, request: Dict[str, str], required_fields) -> bool:
        """
        Check if the request is valid, i.e., if it contains all required fields. The required fileds are defiend in 'common/constants.py' file.
        Input:
            request (Dict[str, str]): The request format.
            required_fields (list): The list of required fields.
        Output:
            bool: True if the request is valid, False otherwise.
        """
        return all(field in request.keys() for field in required_fields)

    def make_decision(self, request: Dict[str, str]) -> bool:
        """
        Make decision based on the request. This should be overridden by subclasses.
        """
        return True

    def response(self) -> Dict[str, str]:
        """
        Agent makes response. This should be overridden by subclasses.
        """
        return self.response
