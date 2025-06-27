# unit test for agent.py
import unittest
from unittest.mock import patch, MagicMock

from data_discovery_ai.agents.supervisorAgent import SupervisorAgent
from data_discovery_ai.agents.descriptionFormattingAgent import (
    DescriptionFormattingAgent,
)


class TestSupervisorAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SupervisorAgent()
        self.valid_request = {
            "uuid": "Example uuid",
            "selected_model": ["description_formatting"],
            "title": "Example title",
            "abstract": "Example abstract",
        }

        self.invalid_model_request = {
            "uuid": "Example uuid",
            "selected_model": ["description_formatting", "unknown model"],
            "title": "Example title",
            "abstract": "Example abstract",
            "links": ["https://example.com"],
        }

        self.invalid_no_uuid_request = {
            "selected_model": ["description_formatting"],
            "title": "Example title",
            "abstract": "Example abstract",
            "links": ["https://example.com"],
        }

        self.non_dict_request = "this is a non dict request"

    @patch("data_discovery_ai.agents.supervisorAgent.logger")
    def test_is_valid_request(self, mock_logger):
        # Test valid request
        self.assertTrue(self.agent.is_valid_request(self.valid_request))
        mock_logger.error.assert_not_called()

        # Test invalid model name
        self.assertFalse(self.agent.is_valid_request(self.invalid_model_request))
        mock_logger.error.assert_called_with(
            "Invalid model name: unknown model. Choose from ['keyword_classification', 'delivery_classification', 'description_formatting', 'link_grouping']"
        )

        # Test no uuid request
        self.assertFalse(self.agent.is_valid_request(self.invalid_no_uuid_request))
        mock_logger.error.assert_called_with("Invalid request format: expected a UUID.")

        # Test non-dict request
        self.assertFalse(
            self.agent.is_valid_request(self.non_dict_request)
        )  # type:ignore

    def test_create_task_agents(self):
        self.agent.make_decision(self.valid_request)
        self.assertEqual(len(self.agent.task_agents), 1)
        self.assertIsInstance(self.agent.task_agents[0], DescriptionFormattingAgent)

    def test_execute(self):
        self.agent.execute(self.valid_request)

        self.assertEqual(
            self.agent.response,
            {"summaries": {"ai:description": "Example abstract"}},
        )


if __name__ == "__main__":
    unittest.main()
