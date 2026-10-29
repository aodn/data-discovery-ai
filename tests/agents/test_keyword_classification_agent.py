#  unit test for model/keywordClassificationAgent.py
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from data_discovery_ai.agents.keywordClassificationAgent import (
    KeywordClassificationAgent,
    reformat_response,
)
from data_discovery_ai.agents.supervisorAgent import SupervisorAgent


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
                    "theme": {"scheme": "Scheme B.1"},
                },
                {
                    "id": "Concept4",
                    "url": "http://example.com/concept3",
                    "title": "Theme B",
                    "description": "Desc B",
                    "theme": {"scheme": "Scheme B.2"},
                },
            ]
        }

    def test_make_decision_valid_request(self):
        # it should always take action if the request is valid, otherwise it will not take action.
        self.assertTrue(self.agent.make_decision(self.valid_request))

        self.assertFalse(self.agent.make_decision(self.invalid_request))

    @patch(
        "data_discovery_ai.agents.keywordClassificationAgent.get_text_embedding",
        return_value=np.zeros(768),
    )
    def test_take_action_uses_supervisor_keyword_model(self, mock_embedding):
        # the model and labels come from the supervisor, which gets them from the server at startup
        keyword_model = MagicMock()
        keyword_model.predict.return_value = np.array([[0.9, 0.1]])
        labels = {
            0: {"id": "Concept1", "title": "Theme A"},
            1: {"id": "Concept2", "title": "Theme B"},
        }
        supervisor = SupervisorAgent()
        supervisor.set_keyword_model(keyword_model)
        supervisor.set_keyword_labels(labels)
        self.agent.set_supervisor(supervisor)

        result = self.agent.take_action("Test Title", "Test abstract.")

        self.assertEqual(result, [labels[0]])
        keyword_model.predict.assert_called_once()

    @patch(
        "data_discovery_ai.agents.keywordClassificationAgent.get_text_embedding",
        return_value=np.zeros(768),
    )
    def test_take_action_without_keyword_model_returns_empty(self, mock_embedding):
        self.agent.set_supervisor(SupervisorAgent())

        self.assertEqual(self.agent.take_action("Test Title", "Test abstract."), [])

    def test_reformat_response(self):
        formatted_response = reformat_response(self.prediction)
        self.assertIn("themes", formatted_response)

        # expect three groups of concepts grouped by title and scheme: A, B.1, and B.2
        self.assertEqual(len(formatted_response["themes"]), 3)
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
