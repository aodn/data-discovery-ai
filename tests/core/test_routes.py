import unittest
import gzip
import json
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

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
