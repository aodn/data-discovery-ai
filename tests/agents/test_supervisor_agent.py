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

        self.response = {
            "links": [
                {
                    "href": "https://www.example.com/",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Example Data Download Link",
                    "ai:group": "Data Access",
                }
            ],
            "summaries": {
                "ai:update_frequency": "delayed",
                "ai:description": "ai generated description",
            },
            "themes": [
                {
                    "scheme": "Scheme A",
                    "title": "Theme A",
                    "description": "Desc A",
                    "ai:description": "AI theme summary",
                    "concepts": [
                        {"id": "Concept1", "url": "http://example.com/concept1"}
                    ],
                }
            ],
        }

        self.model_config = {
            "task_agents": {
                "link_grouping": {"required_fields": ["title"]},
                "description_formatting": {"required_fields": ["description"]},
                "delivery_classification": {"required_fields": ["description"]},
                "keyword_classification": {"required_fields": ["title"]},
            }
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

    def test_process_request_response_valid(self):
        # mock response
        self.agent.response = self.response

        request = {
            "uuid": "Example uuid",
            "title": "Example title",
            "abstract": "Example description",
            "lineage": "Example lineage",
            "links": [],
        }
        result = self.agent.process_request_response(request)

        self.assertEqual(result["id"], "Example uuid")
        self.assertEqual(result["title"], "Example title")
        self.assertEqual(result["description"], "Example description")
        self.assertEqual(result["summaries"]["statement"], "Example lineage")
        self.assertEqual(
            result["summaries"]["ai:description"], "ai generated description"
        )
        self.assertEqual(result["themes"][0]["title"], "Theme A")
        self.assertEqual(result["themes"][0]["concepts"][0]["id"], "Concept1")
        self.assertEqual(result["links"][0]["ai:group"], "Data Access")

    def test_search_stored_data_hit(self):
        mock_es_client = MagicMock()
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "id": "Example uuid",
                            "ai:request_raw": {
                                "uuid": "Example uuid",
                                "selected_model": [
                                    "description_formatting",
                                    "keyword_classification",
                                    "delivery_classification",
                                    "link_grouping",
                                ],
                                "title": "Example title",
                                "abstract": "Example description",
                                "lineage": "Example lineage",
                                "links": [],
                            },
                            "links": self.response["links"],
                            "summaries": self.response["summaries"],
                            "themes": self.response["themes"],
                        }
                    }
                ]
            }
        }

        request = {
            "uuid": "Example uuid",
            "selected_model": ["link_grouping", "description_formatting"],
            "title": "Example title",
            "abstract": "Example description",
            "links": [
                {
                    "href": "https://www.example.com/",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Example Data Download Link",
                }
            ],
        }

        self.agent.model_config = self.model_config

        result, matched_model = self.agent.search_stored_data(
            request, mock_es_client, index="test_index"
        )

        self.assertEqual(set(matched_model), set(request["selected_model"]))

        self.assertIn("links", result)
        self.assertEqual(
            result["summaries"]["ai:description"], "ai generated description"
        )


if __name__ == "__main__":
    unittest.main()
