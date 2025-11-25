import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from data_discovery_ai.utils.ogcapi_connector import OGCAPIConnector
from data_discovery_ai.config.config import ConfigUtil


class TestOGCAPIConnector(unittest.TestCase):
    """
    Test suite for OGCAPIConnector class
    """

    def setUp(self):
        """
        Set up test fixtures before each test method
        This method is called before every test
        """
        # Create mock config
        self.mock_config = Mock(spec=ConfigUtil)
        self.mock_ogcapi_config = Mock()
        self.mock_ogcapi_config.host = "http://localhost:8080"
        self.mock_ogcapi_config.endpoint = "/api/v1/ogc/collections"
        self.mock_ogcapi_config.page_size = 3
        self.mock_ogcapi_config.sleep_time = 0.1
        self.mock_ogcapi_config.query = (
            "?properties=id,title,description,statement,themes,status&filter=page_size="
        )
        self.mock_ogcapi_config.max_retries = 3
        self.mock_ogcapi_config.timeout = 30

        self.mock_config.get_ogcapi_config.return_value = self.mock_ogcapi_config

        # Create connector instance
        self.connector = OGCAPIConnector(config=self.mock_config)

    def tearDown(self):
        """
        Clean up after each test method
        """
        self.connector = None

    def test_init_with_config(self):
        """
        Test initialization with provided config
        """
        connector = OGCAPIConnector(config=self.mock_config)

        self.assertEqual(connector.host, "http://localhost:8080")
        self.assertEqual(connector.endpoint, "/api/v1/ogc/collections")
        self.assertEqual(connector.page_size, 3)
        self.assertEqual(connector.sleep_time, 0.1)
        self.assertEqual(connector.max_retries, 3)
        self.assertEqual(connector.timeout, 30)
        self.assertIsInstance(connector.session, requests.Session)

    @patch("data_discovery_ai.utils.ogcapi_connector.ConfigUtil.get_config")
    def test_init_without_config(self, mock_get_config):
        """
        Test initialization without config (uses default ConfigUtil.get_config())
        """
        mock_config = Mock()
        mock_ogcapi_config = Mock()
        mock_ogcapi_config.host = "http://example.com"
        mock_ogcapi_config.endpoint = "/collections"
        mock_ogcapi_config.page_size = 10
        mock_ogcapi_config.sleep_time = 1
        mock_ogcapi_config.query = "?filter=page_size="
        mock_ogcapi_config.max_retries = 5
        mock_ogcapi_config.timeout = 60

        mock_config.get_ogcapi_config.return_value = mock_ogcapi_config
        mock_get_config.return_value = mock_config

        connector = OGCAPIConnector()

        self.assertEqual(connector.host, "http://example.com")
        mock_get_config.assert_called_once()

    def test_init_without_host_raises_error(self):
        """
        Test that ValueError is raised when host is not available
        """
        self.mock_ogcapi_config.host = None

        with self.assertRaises(ValueError) as context:
            OGCAPIConnector(config=self.mock_config)

        self.assertIn("OGCAPI host is not available", str(context.exception))

    def test_create_session(self):
        """
        Test that session is created with retry strategy
        """
        session = self.connector.session

        self.assertIsInstance(session, requests.Session)
        # Check that adapters are mounted
        self.assertIn("http://", session.adapters)
        self.assertIn("https://", session.adapters)
        self.assertIsInstance(session.adapters["http://"], HTTPAdapter)
        self.assertIsInstance(session.adapters["https://"], HTTPAdapter)

    def test_build_search_after_filter(self):
        """
        Test building search_after filter with various inputs
        """
        # Test with list of strings
        search_after = ["value1", "value2", "value3"]
        result = self.connector.build_search_after_filter(search_after)

        self.assertIn("search_after", result)
        self.assertIn("value1", result)
        self.assertIn("value2", result)
        self.assertIn("value3", result)
        self.assertTrue(result.startswith("+AND+"))

    def test_build_search_after_filter_with_numbers(self):
        """
        Test building search_after filter with numbers
        """
        search_after = [123, 456]
        result = self.connector.build_search_after_filter(search_after)

        self.assertIn("123", result)
        self.assertIn("456", result)
        self.assertTrue(result.startswith("+AND+"))

    def test_build_search_after_filter_with_mixed_types(self):
        """
        Test building search_after filter with mixed types
        """
        search_after = ["text", 123, "another"]
        result = self.connector.build_search_after_filter(search_after)

        self.assertIn("text", result)
        self.assertIn("123", result)
        self.assertIn("another", result)

    def test_build_ogcapi_query_url(self):
        """
        Test building OGCAPI query URL
        """
        url = self.connector.build_ogcapi_query_url()

        expected_url = "http://localhost:8080/api/v1/ogc/collections?properties=id,title,description,statement,themes,status&filter=page_size=3"
        self.assertEqual(url, expected_url)

    def test_build_ogcapi_query_url_with_trailing_slash(self):
        """
        Test URL building handles trailing/leading slashes correctly
        """
        self.mock_ogcapi_config.host = "http://localhost:8080/"
        self.mock_ogcapi_config.endpoint = "/api/v1/ogc/collections"

        connector = OGCAPIConnector(config=self.mock_config)
        url = connector.build_ogcapi_query_url()

        # Should not have double slashes after protocol
        url_without_protocol = url.replace("http://", "")
        self.assertNotIn("//", url_without_protocol)

    def test_parse_fetched_collection_data(self):
        """
        Test parsing collection data from API response
        """
        resp_dict = {
            "collections": [
                {
                    "id": "collection-1",
                    "title": "Test Collection 1",
                    "description": "Description 1",
                    "properties": {
                        "statement": "Statement 1",
                        "themes": ["theme1", "theme2"],
                        "status": "onGoing",
                    },
                },
                {
                    "id": "collection-2",
                    "title": "Test Collection 2",
                    "description": "Description 2",
                    "properties": {
                        "statement": "Statement 2",
                        "themes": ["theme3"],
                        "status": "Completed",
                    },
                },
            ]
        }

        result = self.connector._parse_fetched_collection_data(resp_dict)

        self.assertEqual(len(result), 2)

        # Check first collection
        self.assertEqual(result[0]["id"], "collection-1")
        self.assertEqual(result[0]["title"], "Test Collection 1")
        self.assertEqual(result[0]["description"], "Description 1")
        self.assertEqual(result[0]["statement"], "Statement 1")
        self.assertEqual(result[0]["themes"], ["theme1", "theme2"])
        self.assertEqual(result[0]["status"], "onGoing")

        # Check second collection
        self.assertEqual(result[1]["id"], "collection-2")
        self.assertEqual(result[1]["title"], "Test Collection 2")

    def test_parse_fetched_collection_data_with_missing_properties(self):
        """
        Test parsing collection data when some properties are missing
        """
        resp_dict = {
            "collections": [
                {
                    "id": "collection-1",
                    "title": "Test Collection",
                    "description": "Description",
                    "properties": {},  # Missing statement, themes, status
                }
            ]
        }

        result = self.connector._parse_fetched_collection_data(resp_dict)

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["statement"])
        self.assertIsNone(result[0]["themes"])
        self.assertIsNone(result[0]["status"])

    def test_parse_fetched_collection_data_empty_response(self):
        """
        Test parsing empty collection data
        """
        resp_dict = {"collections": []}

        result = self.connector._parse_fetched_collection_data(resp_dict)

        self.assertEqual(result, [])

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_get_all_collections_single_page(self, mock_sleep):
        """
        Test getting all collections when only one page exists (no search_after)
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 2,
            "collections": [
                {
                    "id": "col-1",
                    "title": "Collection 1",
                    "description": "Desc 1",
                    "properties": {
                        "statement": "Statement 1",
                        "themes": ["theme1"],
                        "status": "onGoing",
                    },
                },
                {
                    "id": "col-2",
                    "title": "Collection 2",
                    "description": "Desc 2",
                    "properties": {
                        "statement": "Statement 2",
                        "themes": ["theme2"],
                        "status": "onGoing",
                    },
                },
            ],
            "search_after": None,  # No more pages
        }

        self.connector.session.get = Mock(return_value=mock_response)

        result = self.connector.get_all_collections()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["id"], "col-1")
        self.assertEqual(result.iloc[1]["id"], "col-2")
        self.connector.session.get.assert_called_once()
        mock_sleep.assert_not_called()  # No sleep for single page

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_get_all_collections_multiple_pages(self, mock_sleep):
        """
        Test getting all collections across multiple pages with search_after
        """
        # First page response
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "total": 5,
            "collections": [
                {
                    "id": "col-1",
                    "title": "Collection 1",
                    "description": "Desc 1",
                    "properties": {
                        "statement": "S1",
                        "themes": ["t1"],
                        "status": "onGoing",
                    },
                },
                {
                    "id": "col-2",
                    "title": "Collection 2",
                    "description": "Desc 2",
                    "properties": {
                        "statement": "S2",
                        "themes": ["t2"],
                        "status": "onGoing",
                    },
                },
            ],
            "search_after": ["cursor1", "cursor2"],
        }

        # Second page response
        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "total": 5,
            "collections": [
                {
                    "id": "col-3",
                    "title": "Collection 3",
                    "description": "Desc 3",
                    "properties": {
                        "statement": "S3",
                        "themes": ["t3"],
                        "status": "onGoing",
                    },
                },
                {
                    "id": "col-4",
                    "title": "Collection 4",
                    "description": "Desc 4",
                    "properties": {
                        "statement": "S4",
                        "themes": ["t4"],
                        "status": "onGoing",
                    },
                },
            ],
            "search_after": ["cursor3", "cursor4"],
        }

        # Third page response (last page)
        third_response = Mock()
        third_response.status_code = 200
        third_response.json.return_value = {
            "total": 5,
            "collections": [
                {
                    "id": "col-5",
                    "title": "Collection 5",
                    "description": "Desc 5",
                    "properties": {
                        "statement": "S5",
                        "themes": ["t5"],
                        "status": "onGoing",
                    },
                }
            ],
            "search_after": [],  # Empty list means no more pages
        }

        self.connector.session.get = Mock(
            side_effect=[first_response, second_response, third_response]
        )

        result = self.connector.get_all_collections()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(result.iloc[0]["id"], "col-1")
        self.assertEqual(result.iloc[4]["id"], "col-5")
        self.assertEqual(self.connector.session.get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Sleep between pages

    def test_get_all_collections_http_error_first_request(self):
        """
        Test error handling when first request fails
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")

        self.connector.session.get = Mock(return_value=mock_response)

        with self.assertRaises(requests.HTTPError):
            self.connector.get_all_collections()

    @patch("time.sleep")
    def test_get_all_collections_http_error_subsequent_request(self, mock_sleep):
        """
        Test error handling when a subsequent paginated request fails
        """
        # First request succeeds
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "total": 5,
            "collections": [
                {
                    "id": "col-1",
                    "title": "Collection 1",
                    "description": "Desc 1",
                    "properties": {
                        "statement": "S1",
                        "themes": ["t1"],
                        "status": "onGoing",
                    },
                }
            ],
            "search_after": ["cursor1"],
        }

        # Second request fails
        second_response = Mock()
        second_response.status_code = 500
        second_response.raise_for_status.side_effect = requests.HTTPError(
            "Server Error"
        )

        self.connector.session.get = Mock(side_effect=[first_response, second_response])

        with self.assertRaises(requests.HTTPError):
            self.connector.get_all_collections()

    @patch("time.sleep")
    def test_get_all_collections_empty_result(self, mock_sleep):
        """
        Test getting collections when result is empty
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 0,
            "collections": [],
            "search_after": None,
        }

        self.connector.session.get = Mock(return_value=mock_response)

        result = self.connector.get_all_collections()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
        self.assertTrue(result.empty)

    def test_get_all_collections_with_none_search_after(self):
        """
        Test that None search_after is handled correctly (should stop time sleep)
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "collections": [
                {
                    "id": "col-1",
                    "title": "Collection 1",
                    "description": "Desc 1",
                    "properties": {
                        "statement": "S1",
                        "themes": ["t1"],
                        "status": "onGoing",
                    },
                }
            ],
            "search_after": None,
        }

        self.connector.session.get = Mock(return_value=mock_response)

        result = self.connector.get_all_collections()

        self.assertEqual(len(result), 1)
        self.connector.session.get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
