#  unit test for model/keywordClassificationAgent.py
import unittest
from unittest.mock import patch, MagicMock
from data_discovery_ai.model.keywordClassificationAgent import (
    KeywordClassificationAgent,
)
from data_discovery_ai.utils.config_utils import ConfigUtil


class TestKeywordClassificationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = KeywordClassificationAgent()
        self.agent.set_required_fields(["title", "abstract"])
        self.valid_request = {"title": "Test Title", "abstract": "Test abstract."}

        self.invalid_request = {"title": "Test Title"}

    def test_is_valid_request(self):
        # Test valid request
        self.assertTrue(self.agent.is_valid_request(self.valid_request))

        # Test invalid request with missing field
        self.assertFalse(self.agent.is_valid_request(self.invalid_request))

    def test_make_decision_valid_request(self):
        # it should always take action if the request is valid, otherwise it will not take action.
        self.assertTrue(self.agent.make_decision(self.valid_request))

        self.assertFalse(self.agent.make_decision(self.invalid_request))
