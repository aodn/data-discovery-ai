# unit test for config_utils.py
import unittest
from unittest.mock import patch, mock_open
from data_discovery_ai.config.config import ConfigUtil


MOCK_YAML_CONTENT = """
elasticsearch:
  batch_size: 100
  sleep_time: 5
  es_index_name: test_index

model:
  keyword_classification:
    confidence: 0.5
    top_N: 2
    separator: " [SEP] "
    model: development
"""


class TestConfigUtil(unittest.TestCase):
    @patch("data_discovery_ai.config.config.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML_CONTENT)
    def setUp(self, mock_file, mock_exists):
        self.config_util = ConfigUtil()

    def test_get_es_config(self):
        es_config = self.config_util.get_es_config()
        self.assertEqual(es_config["batch_size"], 100)
        self.assertEqual(es_config["es_index_name"], "test_index")

    def test_get_keyword_classification_config(self):
        model_config = self.config_util.get_keyword_classification_config()
        self.assertEqual(model_config["top_N"], 2)


if __name__ == "__main__":
    unittest.main()
