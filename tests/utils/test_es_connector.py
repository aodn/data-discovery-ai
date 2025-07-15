# unit test for es_connector.py
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from pytest_elasticsearch import factories

from data_discovery_ai.utils.es_connector import connect_es, search_es, create_es_index


class TestESConnector(unittest.TestCase):
    @patch("data_discovery_ai.utils.es_connector.logger")
    @patch("data_discovery_ai.utils.es_connector.logging.info")
    @patch("data_discovery_ai.utils.es_connector.Elasticsearch")
    @patch("data_discovery_ai.utils.es_connector.os.getenv")
    def test_connect_es_success(
        self, mock_getenv, mock_es_class, mock_logging_info, mock_logger
    ):
        mock_getenv.side_effect = lambda key: {
            "ES_ENDPOINT": "http://example.com:9200",
            "ES_API_KEY": "mock_api_key",
        }[key]

        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        client = connect_es()
        self.assertEqual(client, mock_client)
        mock_es_class.assert_called_once_with(
            "http://example.com:9200", api_key="mock_api_key"
        )
        mock_logging_info.assert_called_once_with("Connected to ElasticSearch")
        mock_logger.error.assert_not_called()

    @patch("data_discovery_ai.utils.es_connector.logger")
    @patch(
        "data_discovery_ai.utils.es_connector.Elasticsearch",
        side_effect=Exception("Connection Failed"),
    )
    @patch("data_discovery_ai.utils.es_connector.os.getenv")
    def test_connect_es_failure(self, mock_getenv, mock_es_class, mock_logger):
        mock_getenv.side_effect = lambda key: {
            "ES_ENDPOINT": "http://example.com:9200",
            "ES_API_KEY": "mock_api_key",
        }[key]

        client = connect_es()
        self.assertIsNone(client)
        mock_logger.error.assert_called_once()
        self.assertIn("Connection Failed", mock_logger.error.call_args[0][0])

    @patch("data_discovery_ai.utils.es_connector.logger")
    @patch("data_discovery_ai.utils.es_connector.pd")
    def test_search_es_success(self, mock_pd, mock_logger):
        # Setup
        mock_client = MagicMock()
        mock_client.count.return_value = {"count": 3}
        mock_client.open_point_in_time.return_value = {"id": "mock_pit_id"}

        mock_hit = {"_source": {"field": "value"}, "sort": ["mock_sort_value"]}
        search_result = {
            "hits": {"hits": [mock_hit, mock_hit]},
            "pit_id": "mock_pit_id",
        }
        mock_client.search.side_effect = [search_result, search_result]

        mock_client.close_point_in_time.return_value = True

        df_mock = pd.DataFrame([{"field": "value"}])
        mock_pd.json_normalize.return_value = df_mock
        mock_pd.concat.return_value = pd.DataFrame([{"field": "value"}])

        result = search_es(mock_client, "test-index", batch_size=2, sleep_time=0)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(mock_client.search.call_count, 2)
        self.assertEqual(mock_pd.concat.call_count, 1)

    @patch("data_discovery_ai.utils.es_connector.connect_es")
    @patch("data_discovery_ai.utils.es_connector.ConfigUtil.get_config")
    @patch("builtins.open", new_callable=mock_open, read_data='{"mappings": {}}')
    @patch("os.path.exists", return_value=True)
    def test_create_es_index_success(
        self, mock_exists, mock_open_file, mock_configutil, mock_connect_es
    ):
        mock_config = MagicMock()
        mock_config.get_es_config.return_value.es_ai_index_name = "test-index"
        mock_config.base_dir = Path("/tmp")
        mock_configutil.return_value = mock_config

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_connect_es.return_value = mock_client

        create_es_index()

        mock_client.indices.exists.assert_called_once_with(index="test-index")
        mock_client.indices.create.assert_called_once()
        mock_open_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
