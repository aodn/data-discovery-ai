# unit test for es_connector.py
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch, mock_open

from elasticsearch import ApiError, ConnectionError as ESConnectionError
from tenacity import wait_none

from data_discovery_ai.utils.es_connector import (
    ES_REQUEST_RETRY,
    ES_REQUEST_TIMEOUT,
    ES_STARTUP_REQUEST_TIMEOUT,
    RETRYABLE_HTTP_STATUS,
    _check_connection,
    _index_document,
    connect_es,
    create_es_index,
    search_es_documents,
    store_ai_generated_data,
)


class TestESConnector(unittest.TestCase):
    def setUp(self):
        retry_logger_patcher = patch("data_discovery_ai.utils.retry_template.logger")
        retry_logger_patcher.start()
        self.addCleanup(retry_logger_patcher.stop)

    @staticmethod
    def _without_wait(retry_wrapped_function):
        return retry_wrapped_function.retry_with(wait=wait_none())

    @staticmethod
    def _api_error(status: int) -> ApiError:
        return ApiError(
            f"HTTP {status}",
            meta=MagicMock(status=status),
            body={"status": status},
        )

    @patch("data_discovery_ai.utils.es_connector._check_connection")
    @patch("data_discovery_ai.utils.es_connector.logger")
    @patch("data_discovery_ai.utils.es_connector.logging.info")
    @patch("data_discovery_ai.utils.es_connector.Elasticsearch")
    @patch("data_discovery_ai.utils.es_connector.os.getenv")
    def test_connect_es_success(
        self,
        mock_getenv,
        mock_es_class,
        mock_logging_info,
        mock_logger,
        mock_check_connection,
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
            "http://example.com:9200",
            api_key="mock_api_key",
            max_retries=0,
        )
        mock_check_connection.assert_called_once_with(mock_client)
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

    @patch("data_discovery_ai.utils.es_connector.logging.info")
    @patch("data_discovery_ai.utils.es_connector.Elasticsearch")
    @patch("data_discovery_ai.utils.es_connector.os.getenv")
    def test_connect_es_retries_connection_error_without_client_level_retries(
        self, mock_getenv, mock_es_class, mock_logging_info
    ):
        mock_getenv.side_effect = lambda key: {
            "ES_ENDPOINT": "http://example.com:9200",
            "ES_API_KEY": "mock_api_key",
        }[key]
        client = MagicMock()
        request_client = MagicMock()
        client.options.return_value = request_client
        request_client.info.side_effect = [
            ESConnectionError("temporary failure"),
            {},
        ]
        mock_es_class.return_value = client

        with patch(
            "data_discovery_ai.utils.es_connector._check_connection",
            self._without_wait(_check_connection),
        ):
            result = connect_es()

        self.assertIs(result, client)
        self.assertEqual(request_client.info.call_count, 2)
        self.assertEqual(
            client.options.call_args_list,
            [call(request_timeout=ES_STARTUP_REQUEST_TIMEOUT)] * 2,
        )
        mock_es_class.assert_called_once_with(
            "http://example.com:9200",
            api_key="mock_api_key",
            max_retries=0,
        )
        mock_logging_info.assert_called_once_with("Connected to ElasticSearch")

    def test_search_retries_connection_error_and_applies_request_timeout(self):
        client = MagicMock()
        request_client = MagicMock()
        client.options.return_value = request_client
        expected_response = {"hits": {"hits": []}}
        request_client.search.side_effect = [
            ESConnectionError("temporary failure"),
            expected_response,
        ]

        result = self._without_wait(search_es_documents)(
            client, "test-index", {"query": {"match_all": {}}}
        )

        self.assertEqual(result, expected_response)
        self.assertEqual(request_client.search.call_count, 2)
        self.assertEqual(
            client.options.call_args_list,
            [call(request_timeout=ES_REQUEST_TIMEOUT)] * 2,
        )

    def test_store_retries_connection_error_and_applies_request_timeout(self):
        client = MagicMock()
        request_client = MagicMock()
        client.options.return_value = request_client
        request_client.index.side_effect = [
            ESConnectionError("temporary failure"),
            None,
        ]
        data = {"id": "document-id", "value": "test"}

        with patch(
            "data_discovery_ai.utils.es_connector._index_document",
            self._without_wait(_index_document),
        ):
            store_ai_generated_data(data, client, "test-index")

        self.assertEqual(request_client.index.call_count, 2)
        self.assertEqual(
            client.options.call_args_list,
            [call(request_timeout=ES_REQUEST_TIMEOUT)] * 2,
        )
        request_client.index.assert_called_with(
            index="test-index", document=data, id="document-id"
        )

    def test_search_retries_retryable_http_statuses(self):
        retryable_statuses = RETRYABLE_HTTP_STATUS - {408}

        for status in retryable_statuses:
            with self.subTest(status=status):
                client = MagicMock()
                request_client = MagicMock()
                client.options.return_value = request_client
                expected_response = {"hits": {"hits": []}}
                request_client.search.side_effect = [
                    self._api_error(status),
                    expected_response,
                ]

                result = self._without_wait(search_es_documents)(
                    client, "test-index", {"query": {"match_all": {}}}
                )

                self.assertEqual(result, expected_response)
                self.assertEqual(request_client.search.call_count, 2)

    def test_search_does_not_retry_non_retryable_4xx_response(self):
        client = MagicMock()
        request_client = MagicMock()
        client.options.return_value = request_client
        original_error = self._api_error(400)
        request_client.search.side_effect = original_error

        with self.assertRaises(ApiError) as raised:
            self._without_wait(search_es_documents)(
                client, "test-index", {"query": {"match_all": {}}}
            )

        self.assertIs(raised.exception, original_error)
        request_client.search.assert_called_once()

    def test_storage_stops_at_attempt_limit_and_reraises_original_error(self):
        client = MagicMock()
        request_client = MagicMock()
        client.options.return_value = request_client
        original_error = ESConnectionError("persistent failure")
        request_client.index.side_effect = original_error

        with self.assertRaises(ESConnectionError) as raised:
            self._without_wait(_index_document)(
                client,
                "test-index",
                "document-id",
                {"id": "document-id"},
            )

        self.assertIs(raised.exception, original_error)
        self.assertEqual(request_client.index.call_count, ES_REQUEST_RETRY.max_attempts)
        self.assertEqual(
            client.options.call_count,
            ES_REQUEST_RETRY.max_attempts,
        )

    @patch("data_discovery_ai.utils.es_connector._create_index")
    @patch("data_discovery_ai.utils.es_connector._index_exists", return_value=False)
    @patch("data_discovery_ai.utils.es_connector.connect_es")
    @patch("data_discovery_ai.utils.es_connector.ConfigUtil.get_config")
    @patch("builtins.open", new_callable=mock_open, read_data='{"mappings": {}}')
    @patch("os.path.exists", return_value=True)
    def test_create_es_index_success(
        self,
        mock_exists,
        mock_open_file,
        mock_configutil,
        mock_connect_es,
        mock_index_exists,
        mock_create_index,
    ):
        mock_config = MagicMock()
        mock_config.get_es_config.return_value.es_ai_index_name = "test-index"
        mock_config.base_dir = Path("/tmp")
        mock_configutil.return_value = mock_config

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_connect_es.return_value = mock_client

        result = create_es_index()

        self.assertEqual(result, (mock_client, "test-index"))
        mock_index_exists.assert_called_once_with(mock_client, "test-index")
        mock_create_index.assert_called_once()
        mock_open_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
