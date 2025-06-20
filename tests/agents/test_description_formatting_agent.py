# unit test for model/descriptionFormattingAgent.py
import unittest
from unittest.mock import patch, MagicMock, call
from data_discovery_ai.agents.descriptionFormattingAgent import (
    DescriptionFormattingAgent,
    retrieve_json,
)


class TestDescriptionFormattingAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DescriptionFormattingAgent()
        self.agent.set_required_fields(["title", "abstract"])
        self.valid_request = {
            "title": "Test",
            "abstract": "This is a sentence.\n" * 200,
        }
        self.invalid_request = {"title": "Test", "abstract": "Short abstract"}
        self.invalid_request_missing_field = {"title": "Test"}
        self.test_formatted_abstract = """This is the formatted abstract.
        {
        "formatted_abstract": "#title **Formatted abstract**"
        }"""

    def test_is_valid_request(self):
        # Test valid request
        self.assertTrue(self.agent.is_valid_request(self.valid_request))

        # Test invalid request with missing field
        self.assertFalse(
            self.agent.is_valid_request(self.invalid_request_missing_field)
        )

    def test_make_decision_valid_request(self):
        # valid request for taking action
        self.assertTrue(self.agent.make_decision(self.valid_request))

        # short abstract, no action needed
        self.assertFalse(self.agent.make_decision(self.invalid_request))

    def test_retrieve_json_valid(self):
        output = self.test_formatted_abstract
        result = retrieve_json(output)
        self.assertEqual(result, "#title **Formatted abstract**")

    @patch("data_discovery_ai.agents.descriptionFormattingAgent.logger")
    def test_retrieve_json_invalid(self, mock_logger):
        test_no_json_output = "A test string with no JSON output"
        result = retrieve_json(test_no_json_output)
        self.assertEqual(result, "A test string with no JSON output")
        mock_logger.error.assert_called_once_with("No JSON found in LLM response.")

    @patch("data_discovery_ai.agents.descriptionFormattingAgent.logger")
    @patch("data_discovery_ai.agents.descriptionFormattingAgent.chat")
    def test_execute_agent(self, mock_chat, mock_logger):
        fake_resp = MagicMock()
        fake_resp.message.content = (
            '{"formatted_abstract": "#title **Formatted abstract**"}'
        )
        mock_chat.return_value = fake_resp

        self.agent.execute(self.valid_request)

        self.assertEqual(
            self.agent.response["summaries.ai:description"],
            "#title **Formatted abstract**",
        )
        expected_response = {
            "summaries.ai:description": "#title **Formatted abstract**"
        }
        expected_calls = [
            call("Description is being reformatted by description_formatting agent"),
            call(
                f"description_formatting agent finished, it responses: \n {expected_response}",
            ),
        ]
        mock_logger.info.assert_has_calls(expected_calls)
        self.assertEqual(mock_logger.info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
