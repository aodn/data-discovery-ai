import unittest
import gzip
import json
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from data_discovery_ai.core.routes import ensure_ready
from data_discovery_ai.server import app
from data_discovery_ai.utils.api_utils import api_key_auth

client = TestClient(app)


async def override_dependency():
    return "test-api-key"


async def override_ensure_ready():
    pass


class TestRoutes(unittest.TestCase):
    def setUp(self):
        app.state.tokenizer = MagicMock()
        app.state.embedding_model = MagicMock()

        app.state.client = MagicMock()
        app.state.index = MagicMock()

        app.dependency_overrides[api_key_auth] = override_dependency
        app.dependency_overrides[ensure_ready] = override_ensure_ready

    def tearDown(self):
        app.dependency_overrides = {}

    def test_process_record_with_compressed_request(self):
        payload = {
            "selected_model": ["link_grouping"],
            "uuid": "test-uuid",
            "links": [
                {
                    "href": "https://example.com",
                    "title": "Example Link",
                    "rel": "related",
                    "type": "text/html",
                }
            ],
        }

        # Compress the payload
        compressed_body = BytesIO()
        with gzip.GzipFile(fileobj=compressed_body, mode="wb") as f:
            f.write(json.dumps(payload).encode("utf-8"))

        response = client.post(
            "/api/v1/ml/process_record",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "X-API-Key": "test-api-key",
            },
            data=compressed_body.getvalue(),
        )

        self.assertEqual(response.status_code, 200)

    def test_delete_doc_success(self):
        with patch("data_discovery_ai.core.routes.delete_es_document") as mock_delete:
            mock_delete.return_value = True
            response = client.delete(
                "/api/v1/ml/delete_doc",
                params={"doc_id": "test_doc_id"},
                headers={"X-API-Key": "test-api-key"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("deleted", response.json()["message"])

    def test_delete_doc_not_found(self):
        with patch("data_discovery_ai.core.routes.delete_es_document") as mock_delete:
            mock_delete.return_value = False
            response = client.delete(
                "/api/v1/ml/delete_doc",
                params={"doc_id": "test_doc_id"},
                headers={"X-API-Key": "test-api-key"},
            )
            self.assertEqual(response.status_code, 404)

    @patch("data_discovery_ai.core.routes.store_ai_generated_data")
    @patch("data_discovery_ai.core.routes.SupervisorAgent")
    def test_process_record_streaming_response(self, mock_agent, mock_store):
        mock_agent.is_valid_request.return_value = True
        mock_agent.response = {
            "links": [
                {
                    "href": "https://example.com",
                    "title": "Example Link",
                    "rel": "related",
                    "type": "text/html",
                    "ai:group": "Others",
                }
            ]
        }
        mock_agent.return_value.search_stored_data.return_value = ({}, [])
        mock_agent.process_request_response.return_value = {
            "id": "test-uuid",
            "stored": "data",
        }

        payload = {
            "selected_model": ["link_grouping"],
            "uuid": "test-uuid",
            "links": [
                {
                    "href": "https://example.com",
                    "title": "Example Link",
                    "rel": "related",
                    "type": "text/html",
                }
            ],
        }

        compressed_body = BytesIO()
        with gzip.GzipFile(fileobj=compressed_body, mode="wb") as f:
            f.write(json.dumps(payload).encode("utf-8"))

        response = client.post(
            "/api/v1/ml/process_record",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "X-API-Key": "test-api-key",
            },
            data=compressed_body.getvalue(),
        )

        lines = list(response.iter_lines())

        # Check if heartbeat and final message exist
        self.assertTrue(any("event: processing" in line for line in lines))
        self.assertTrue(any("event: done" in line for line in lines))

        mock_store.assert_called_once()
