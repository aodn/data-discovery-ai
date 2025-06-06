#  unit test for model/keywordClassificationAgent.py
import unittest
from data_discovery_ai.agents.keywordClassificationAgent import (
    KeywordClassificationAgent,
)


class TestKeywordClassificationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = KeywordClassificationAgent()
        self.agent.set_required_fields(["title", "abstract"])
        self.valid_request = {"title": "Test Title", "abstract": "Test abstract."}

        self.invalid_request = {"title": "Test Title"}

    def test_make_decision_valid_request(self):
        # it should always take action if the request is valid, otherwise it will not take action.
        self.assertTrue(self.agent.make_decision(self.valid_request))

        self.assertFalse(self.agent.make_decision(self.invalid_request))
