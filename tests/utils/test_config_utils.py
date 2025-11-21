# unit test for config_utils.py
import unittest
from unittest.mock import patch, mock_open
from data_discovery_ai.config.config import ConfigUtil


MOCK_YAML_CONTENT = """
elasticsearch:
  es_index_name: test_index

model:
  keyword_classification:
    confidence: 0.5
    top_N: 2
    separator: " [SEP] "
    model: development

ogcapi:
  host: "http://localhost:8080/"
  endpoint: "api/collections"
  query: "?properties=test-properties"
  max_retries: 3
"""


class TestConfigUtil(unittest.TestCase):
    @patch("data_discovery_ai.config.config.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML_CONTENT)
    def setUp(self, mock_file, mock_exists):
        self.config_util = ConfigUtil.get_config()

    def test_get_es_config(self):
        es_config = self.config_util.get_es_config()
        self.assertEqual(es_config.es_index_name, "test_index")

    def test_get_keyword_classification_config(self):
        model_config = self.config_util.get_keyword_classification_config()
        self.assertEqual(model_config.top_N, 2)

    def test_get_ogcapi_config(self):
        ogcapi_config = self.config_util.get_ogcapi_config()
        self.assertEqual(ogcapi_config.host, "http://localhost:8080/")
        self.assertEqual(ogcapi_config.endpoint, "api/collections")
        self.assertEqual(ogcapi_config.query, "?properties=test-properties")
        self.assertEqual(ogcapi_config.max_retries, 3)


if __name__ == "__main__":
    unittest.main()
