# unit test for model/descriptionFormattingAgent.py
import unittest
from unittest.mock import patch, MagicMock, call
from data_discovery_ai.model.descriptionFormattingAgent import DescriptionFormattingAgent


class TestDescriptionFormattingAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DescriptionFormattingAgent()
        self.valid_request = {"title": "Test", "abstract": "This is a sentence.\n" * 200}
        self.invalid_request = {"title": "Test", "abstract": "Short abstract"}
        self.invalid_request_missing_field = {"title": "Test"}
        self.test_formatted_abstract = """This is the formatted abstract.
        {
        "formatted_abstract": "#title **Formatted abstract**"
        }"""


    def test_make_decision_valid_request(self):
        # valid request for taking action
        self.assertTrue(self.agent.make_decision(self.valid_request))

        # invalid request as missing field
        self.assertFalse(self.agent.make_decision(self.invalid_request_missing_field))

        # short abstract, no action needed
        self.assertFalse(self.agent.make_decision(self.invalid_request))

    def test_retrieve_json_valid(self):
        output = self.test_formatted_abstract
        result = self.agent.retrieve_json(output)
        self.assertEqual(result, "#title **Formatted abstract**")

    @patch("data_discovery_ai.model.descriptionFormattingAgent.logger")
    def test_retrieve_json_invalid(self, mock_logger):
        test_no_json_output = "A test string with no JSON output"
        result = self.agent.retrieve_json(test_no_json_output)
        self.assertEqual(result, "A test string with no JSON output")
        mock_logger.error.assert_called_once_with("No JSON found in LLM response.")

    @patch("data_discovery_ai.model.descriptionFormattingAgent.logger")
    @patch("data_discovery_ai.model.descriptionFormattingAgent.OpenAI")
    def test_execute_agent(self, mock_openai, mock_logger):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = '''
        {
            "formatted_abstract": "#title **Formatted abstract**"
        }
        '''
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_completion
        mock_openai.return_value.chat = mock_chat

        self.agent.execute(self.valid_request)

        self.assertEqual(self.agent.response["formatted_abstract"], "#title **Formatted abstract**")
        self.assertEqual(self.agent.status, 2)
        expected_calls = [
            call("Description is being reformatted by DescriptionFormattingAgent"),
            call(
                "DescriptionFormattingAgent finished, it responses: \n %s",
                {"formatted_abstract": "#title **Formatted abstract**"},
            ),
        ]
        mock_logger.info.assert_has_calls(expected_calls)
        self.assertEqual(mock_logger.info.call_count, 2)
        mock_openai.assert_called_once_with(api_key=self.agent.openai_api_key)

if __name__ == "__main__":
    unittest.main()

