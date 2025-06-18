# unit test for the link grouping agent in model/linkGroupingAgent.py
import unittest
from data_discovery_ai.agents.linkGroupingAgent import LinkGroupingAgent
from data_discovery_ai.config.config import ConfigUtil


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

        self.valid_protocol_request = {
            "links": [
                {
                    "href": "http://nesptropical.edu.au/wp-content/uploads/2016/03/NESP-TWQ-3.1-FINAL-REPORT.pdf",
                    "rel": "WWW:LINK-1.0-http--publication",
                    "type": "",
                    "title": "REPORT - Project Final Report [PDF]",
                },
                {
                    "href": "https://catalogue.aodn.org.au:443/geonetwork/srv/api/records/05818c50-14c2-11dd-bdaa-00188b4c0af8/attachments/1989_01_12.zip",
                    "rel": "data",
                    "type": "",
                    "title": "1989_01_12.zip",
                },
                {
                    "href": "https://processes.aodn.org.au/wps",
                    "rel": "OGC:WPS--gogoduck",
                    "type": "",
                    "title": "csiro_oa_reconstruction_url",
                },
            ]
        }

        self.invalid_request = {
            "links": [
                {
                    "href": "https://example.com",
                    "rel": "parent",
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
        self.assertTrue(self.agent.response["links"], 3)
        # expect output:
        # [{'href': 'https://example.com', 'rel': 'excluded_irrelated_link', 'type': 'text/html'}, {'href': 'https://example.ipynb', 'rel': 'related', 'application/x-ipynb+json', 'title': 'Example Notebook Link', 'group': 'Python Notebook'}, {'href': 'https://example.com', 'rel': 'related', 'type': 'text/html', 'title': 'Example Document Link', 'group': 'Document'}, {'href': 'https://example.wms', 'rel': 'related', 'type': 'text/html', 'title': 'Example Data Link', 'group': 'Data Access'}]
        self.assertEqual(self.agent.response["links"][1]["ai:group"], "Python Notebook")
        self.assertEqual(
            self.agent.response["links"][1]["type"], "application/x-ipynb+json"
        )

    def test_execute_with_protocol(self):
        self.agent.execute(self.valid_protocol_request)
        self.assertEqual(
            self.agent.response["links"],
            [
                {
                    "href": "http://nesptropical.edu.au/wp-content/uploads/2016/03/NESP-TWQ-3.1-FINAL-REPORT.pdf",
                    "rel": "WWW:LINK-1.0-http--publication",
                    "type": "",
                    "title": "REPORT - Project Final Report [PDF]",
                    "ai:group": "Document",
                },
                {
                    "href": "https://catalogue.aodn.org.au:443/geonetwork/srv/api/records/05818c50-14c2-11dd-bdaa-00188b4c0af8/attachments/1989_01_12.zip",
                    "rel": "data",
                    "type": "",
                    "title": "1989_01_12.zip",
                    "ai:group": "Data Access",
                },
                {
                    "href": "https://processes.aodn.org.au/wps",
                    "rel": "OGC:WPS--gogoduck",
                    "type": "",
                    "title": "csiro_oa_reconstruction_url",
                    "ai:group": "Data Access",
                },
            ],
        )
