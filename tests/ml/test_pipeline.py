# unit test for pipeline.py
import unittest
from unittest.mock import patch
from data_discovery_ai.ml.pipeline import KeywordClassifierPipeline


class TestKeywordClassifierPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = KeywordClassifierPipeline()

    def test_valid_model_name(self):
        for name in ["development", "Production", "EXPERIMENTAL"]:
            self.assertTrue(self.pipeline.is_valid_model(name))

    @patch("data_discovery_ai.ml.pipeline.logger")
    def test_invalid_model_name(self, mock_logger):
        in_valid_name = "invalid_model_name"
        self.assertFalse(self.pipeline.is_valid_model(in_valid_name))
        mock_logger.error.assert_called_with(
            'Model name invalid! \nAvailable model name: ["development", "staging", "production", "experimental", "benchmark"]'
        )
