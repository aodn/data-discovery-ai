# unit test for agent.py
import unittest
from unittest.mock import patch

from data_discovery_ai.model.agent import SupervisorAgent, is_valid_request


class TestAgentLogic(unittest.TestCase):

    @patch("data_discovery_ai.model.agent.API_VALID_FIELD", ["title"])
    @patch("data_discovery_ai.model.agent.AVAILABLE_AI_MODELS", ["available_model"])
    @patch("data_discovery_ai.model.agent.logger")
    def test_is_valid_reques(self, mock_logger):
        # Case 1: missing selected_model
        request1 = {"title": "test title"}
        result1 = is_valid_request(request1)
        self.assertEqual(result1, [])
        mock_logger.error.assert_called_once_with(
            "No selected model found in the request"
        )

        mock_logger.reset_mock()  # Reset logger call history

        # Case 2: valid model, valid fields
        request2 = {
            "selected_model": "available_model",
            "title": "test title",
            "abstract": "test abstract",
        }
        result2 = is_valid_request(request2)
        self.assertEqual(result2, ["title"])
        mock_logger.error.assert_not_called()

        # Case 3: input is not a dict
        mock_logger.reset_mock()
        result3 = is_valid_request("invalid_request")
        self.assertEqual(result3, [])
        mock_logger.error.assert_called_once_with("Invalid request")


if __name__ == "__main__":
    unittest.main()
