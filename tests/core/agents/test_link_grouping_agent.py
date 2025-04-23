# unit test for the link grouping agent in model/linkGroupingAgent.py
import unittest
from unittest.mock import patch, MagicMock
from data_discovery_ai.core.agents.linkGroupingAgent import LinkGroupingAgent
from data_discovery_ai.utils.config_utils import ConfigUtil


class TestLinkGroupingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = LinkGroupingAgent()
        self.agent.set_required_fields(["links"])
        self.valid_request = {
            "links": [
                {
                    "href": "https://example.com",
                    "rel": "excluded_irrelated_link",
                    "type": "text/html",
                },
                {
                    "href": "https://example.ipynb",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Example Notebook Link",
                },
                {
                    "href": "https://example.com",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Example Document Link",
                },
                {
                    "href": "https://example.wms",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Example Data Link",
                },
            ]
        }

        self.invalid_request = {
            "links": [
                {
                    "href": "https://example.com",
                    "rel": "parant",
                    "type": "text/html",
                    "title": "Example Link",
                }
            ]
        }

    def test_make_decision(self):
        result = self.agent.make_decision(self.valid_request)
        # expect to skip the first irrelated link
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["title"], "Example Notebook Link")

    def test_make_decision_invalid_request(self):
        result = self.agent.make_decision(self.invalid_request)
        self.assertEqual(result, [])

    def test_load_config(self):
        # Test loading configuration
        self.assertIsInstance(self.agent.config, ConfigUtil)
        self.assertTrue(self.agent.model_config)
        self.assertIn("grouping_rules", self.agent.model_config)

    def test_execute(self):
        self.agent.execute(self.valid_request)
        self.assertTrue(self.agent.response["grouped_links"], 3)
        # expect output:
        # [{'href': 'https://example.com', 'rel': 'excluded_irrelated_link', 'type': 'text/html'}, {'href': 'https://example.ipynb', 'rel': 'related', 'type': 'text/html', 'title': 'Example Notebook Link', 'group': 'Python Notebook'}, {'href': 'https://example.com', 'rel': 'related', 'type': 'text/html', 'title': 'Example Document Link', 'group': 'Document'}, {'href': 'https://example.wms', 'rel': 'related', 'type': 'text/html', 'title': 'Example Data Link', 'group': 'Data Access'}]
        self.assertEqual(
            self.agent.response["grouped_links"][1]["group"], "Python Notebook"
        )
