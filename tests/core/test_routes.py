import unittest
import gzip
import json
import threading
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from data_discovery_ai.core.routes import ensure_ready, event_stream_handler
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

        app.state.nli_tokenizer = MagicMock()
        app.state.nli_model = MagicMock()

        app.state.client = MagicMock()
        app.state.index = MagicMock()

        app.state.llm_client = MagicMock()

        app.state.model_status = "UP"
        app.state.model_error = None

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
        self.assertEqual("event: processing", lines[0])
        self.assertEqual("data: Start processing record UUID test-uuid...", lines[1])
        self.assertTrue(any("event: processing" in line for line in lines))
        self.assertTrue(any("event: done" in line for line in lines))

        mock_store.assert_called_once()


class TestEventStreamHandler(unittest.IsolatedAsyncioTestCase):
    async def test_search_stored_data_runs_off_event_loop(self):
        event_loop_thread = threading.get_ident()
        search_threads = []
        supervisor = MagicMock()

        def search_stored_data(*args, **kwargs):
            search_threads.append(threading.get_ident())
            return {}, []

        supervisor.search_stored_data.side_effect = search_stored_data
        stream = event_stream_handler(
            supervisor=supervisor,
            body={"selected_model": []},
            client=MagicMock(),
            index="test-index",
            max_timeout=1,
            sse_interval=0.1,
            uuid="test-uuid",
            original_request={},
            background_tasks=MagicMock(),
        )

        event = await anext(stream)
        await stream.aclose()

        self.assertEqual(event, "event: done\ndata: {}\n\n")
        self.assertEqual(len(search_threads), 1)
        self.assertNotEqual(search_threads[0], event_loop_thread)


UP_COMPONENT = {"status": "UP", "detail": None}


@patch(
    "data_discovery_ai.utils.health_utils.check_llm",
    new=AsyncMock(return_value=UP_COMPONENT),
)
@patch(
    "data_discovery_ai.utils.health_utils.check_keyword_resources",
    new=MagicMock(return_value=UP_COMPONENT),
)
class TestHealthAndReadiness(unittest.TestCase):
    """
    Health check and readiness gate without overriding ensure_ready.
    """

    def setUp(self):
        app.state.client = MagicMock()
        app.state.index = MagicMock()
        app.dependency_overrides[api_key_auth] = override_dependency

    def tearDown(self):
        app.dependency_overrides = {}
        app.state.model_status = "UP"
        app.state.model_error = None

    def test_health_starting_returns_200(self):
        app.state.model_status = "STARTING"
        response = client.get("/api/v1/ml/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "STARTING")
        self.assertEqual(body["status_code"], 200)
        self.assertEqual(body["components"]["models"]["status"], "STARTING")

    def test_health_up_returns_200(self):
        app.state.model_status = "UP"
        response = client.get("/api/v1/ml/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "UP")

    def test_health_down_still_returns_200(self):
        app.state.model_status = "DOWN"
        app.state.model_error = "download failed"
        response = client.get("/api/v1/ml/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "DOWN")
        self.assertEqual(body["components"]["models"]["detail"], "download failed")

    def test_health_down_when_resource_missing(self):
        app.state.model_status = "STARTING"
        with patch(
            "data_discovery_ai.utils.health_utils.check_keyword_resources",
            return_value={"status": "DOWN", "detail": "missing"},
        ):
            response = client.get("/api/v1/ml/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "DOWN")

    def test_process_record_rejected_while_models_starting(self):
        app.state.model_status = "STARTING"
        response = client.post(
            "/api/v1/ml/process_record",
            headers={"X-API-Key": "test-api-key"},
            json={"selected_model": ["link_grouping"], "uuid": "test-uuid"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("models: STARTING", response.json()["detail"])

    def test_delete_doc_rejected_while_models_starting(self):
        app.state.model_status = "STARTING"
        response = client.delete(
            "/api/v1/ml/delete_doc",
            params={"doc_id": "test_doc_id"},
            headers={"X-API-Key": "test-api-key"},
        )
        self.assertEqual(response.status_code, 503)
