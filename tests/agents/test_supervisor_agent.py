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
            "selected_model": ["description_formatting"],
            "title": "Example title",
            "abstract": "Example abstract",
        }

        self.invalid_model_request = {
            "selected_model": ["description_formatting", "unknown model"],
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

        # Test non-dict request
        self.assertFalse(
            self.agent.is_valid_request(self.non_dict_request)
        )  # type:ignore

    def test_create_task_agents(self):
        self.agent.make_decision(self.valid_request)
        self.assertEqual(len(self.agent.task_agents), 1)
        self.assertIsInstance(self.agent.task_agents[0], DescriptionFormattingAgent)

    @patch("data_discovery_ai.agents.supervisorAgent.Pool")
    def test_execute(self, mock_pool_class):
        mock_pool = MagicMock()
        mock_pool.map.return_value = [
            {"formatted_abstract": "#descrption agent response: **mock_response**"}
        ]
        mock_pool_class.return_value.__enter__.return_value = mock_pool

        mock_agent = MagicMock()
        mock_agent.execute = MagicMock()
        mock_agent.response = {
            "formatted_abstract": "#descrption agent response: **mock_response**"
        }

        self.agent.task_agents = [mock_agent]

        self.agent.execute(self.valid_request)

        mock_pool.map.assert_called_once()
        self.assertEqual(
            self.agent.response,
            {
                "result": {
                    "formatted_abstract": "#descrption agent response: **mock_response**"
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
