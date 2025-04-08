# unit test for es_connector.py
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch

from data_discovery_ai.utils import es_connector


class TestESConnector(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "elasticsearch": {
                "end_point": "http://example.com",
                "api_key": "TEST_API_KEY",
            }
        }

    @patch("data_discovery_ai.utils.es_connector.Elasticsearch")
    def test_connect_es_success(self, mock_es):
        client = es_connector.connect_es(self.mock_config)
        mock_es.assert_called_with("http://example.com", api_key="TEST_API_KEY")
        self.assertIsNotNone(client)

    @patch("data_discovery_ai.utils.es_connector.logger")
    @patch(
        "data_discovery_ai.utils.es_connector.Elasticsearch",
        side_effect=Exception("Connection Error"),
    )
    def test_connect_es_failure(self, mock_es, mock_logger):
        client = es_connector.connect_es(self.mock_config)
        self.assertIsNone(client)
        mock_logger.error.assert_called_once()

    @patch("data_discovery_ai.utils.es_connector.time.sleep", return_value=None)
    def test_search_es_success(self, _):
        mock_client = MagicMock()
        mock_client.count.return_value = {"count": 2}
        mock_client.open_point_in_time.return_value = {"id": "123"}
        mock_client.search.side_effect = [
            {"hits": {"hits": [{"_source": {"val": 1}, "sort": [1]}]}, "pit_id": "123"},
            {"hits": {"hits": [{"_source": {"val": 2}, "sort": [2]}]}, "pit_id": "123"},
        ]

        result = es_connector.search_es(
            client=mock_client, index="test-index", batch_size=1, sleep_time=0
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("data_discovery_ai.utils.es_connector.logger")
    def test_search_es_invalid_response(self, mock_logger):
        mock_client = MagicMock()
        mock_client.count.return_value = {"count": 1}
        mock_client.open_point_in_time.return_value = {"id": "pit123"}
        mock_client.search.return_value = {"no_hits": {}}

        result = es_connector.search_es(mock_client, "test-index", 1, 0)
        self.assertIsNone(result)
        self.assertTrue(mock_logger.error.called)


if __name__ == "__main__":
    unittest.main()
