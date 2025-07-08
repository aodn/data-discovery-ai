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
                    "theme": {
                        "title": "Theme A",
                        "scheme": "Scheme A",
                        "description": "Desc A",
                    },
                },
                {
                    "id": "Concept2",
                    "url": "http://example.com/concept2",
                    "theme": {
                        "title": "Theme A",
                        "scheme": "Scheme A",
                        "description": "Desc A",
                    },
                },
                {
                    "id": "Concept3",
                    "url": "http://example.com/concept3",
                    "theme": {
                        "title": "Theme B",
                        "scheme": "Scheme B",
                        "description": "Desc B",
                    },
                },
            ]
        }

    def test_make_decision_valid_request(self):
        # it should always take action if the request is valid, otherwise it will not take action.
        self.assertTrue(self.agent.make_decision(self.valid_request))

        self.assertFalse(self.agent.make_decision(self.invalid_request))

    def test_reformat_response(self):
        formated_response = reformat_response(self.prediction)
        self.assertIn("themes", formated_response)
        # expected to have two themes A and B
        self.assertEqual(len(formated_response["themes"]), 2)

        for theme in formated_response["themes"]:
            self.assertIn("concepts", theme)
            self.assertIn("scheme", theme)
            self.assertIn("title", theme)
            self.assertIn("description", theme)
            # test if AI prediction field tagged
            self.assertEqual(
                theme["ai:description"], "This is the prediction provided by AI model"
            )

        concepts_by_theme = {
            t["title"]: t["concepts"] for t in formated_response["themes"]
        }
        self.assertEqual(len(concepts_by_theme["Theme A"]), 2)
        self.assertEqual(len(concepts_by_theme["Theme B"]), 1)
