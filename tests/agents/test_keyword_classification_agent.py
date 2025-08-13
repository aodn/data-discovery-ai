#  unit test for model/keywordClassificationAgent.py
import unittest
from data_discovery_ai.agents.keywordClassificationAgent import (
    KeywordClassificationAgent,
    reformat_response,
)


class TestKeywordClassificationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = KeywordClassificationAgent()
        self.agent.set_required_fields(["title", "abstract"])
        self.valid_request = {"title": "Test Title", "abstract": "Test abstract."}

        self.invalid_request = {"title": "Test Title"}

        self.prediction = {
            "themes": [
                {
                    "id": "Concept1",
                    "url": "http://example.com/concept1",
                    "title": "Theme A",
                    "description": "Desc A",
                    "theme": {"scheme": "Scheme A"},
                },
                {
                    "id": "Concept2",
                    "url": "http://example.com/concept2",
                    "title": "Theme A",
                    "description": "Desc A",
                    "theme": {"scheme": "Scheme A"},
                },
                {
                    "id": "Concept3",
                    "url": "http://example.com/concept3",
                    "title": "Theme B",
                    "description": "Desc B",
                    "theme": {"scheme": "Scheme B"},
                },
            ]
        }

    def test_make_decision_valid_request(self):
        # it should always take action if the request is valid, otherwise it will not take action.
        self.assertTrue(self.agent.make_decision(self.valid_request))

        self.assertFalse(self.agent.make_decision(self.invalid_request))

    def test_reformat_response(self):
        formatted_response = reformat_response(self.prediction)
        self.assertIn("themes", formatted_response)

        # expect two grouped themes: A and B
        self.assertEqual(len(formatted_response["themes"]), 2)
        self.assertIsInstance(formatted_response["themes"], list)

        for theme in formatted_response["themes"]:
            with self.subTest(theme=theme):
                self.assertIn("scheme", theme)
                self.assertIsInstance(theme["concepts"], list)

                for concept in theme["concepts"]:
                    with self.subTest(concept=concept):
                        # expected to have STAC fields
                        self.assertIn("id", concept)
                        self.assertIn("url", concept)
                        self.assertIn("title", concept)
                        self.assertIn("description", concept)

                        # expected to have ai:description field
                        self.assertIn("ai:description", concept)
                        self.assertEqual(
                            concept["ai:description"],
                            "This is the prediction provided by AI model.",
                        )
