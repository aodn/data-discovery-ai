# request_id correlation through RequestContextMiddleware (issue 9305)
"""
Real requests through the real app: the request_id the middleware binds must
reach logs from the route, the SSE body, the supervisor running on a worker
thread (asyncio.to_thread) and the BackgroundTask that runs after the
response. BaseHTTPMiddleware has had contextvars bugs in older Starlette
releases, so this is verified with actual requests rather than assumed.
"""

import json
import logging
import re
import unittest
from unittest.mock import MagicMock, patch

import starlette
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from data_discovery_ai.config.log_formatter import JsonLogFormatter
from data_discovery_ai.core.middleware import (
    RequestContextMiddleware,
    configure_request_context_middleware,
)
from data_discovery_ai.core.routes import ensure_ready
from data_discovery_ai.server import app
from data_discovery_ai.utils.api_utils import api_key_auth
from data_discovery_ai.utils.log_context import current_context, install_context_filter

UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(JsonLogFormatter())
        install_context_filter(self)
        self.payloads = []

    def emit(self, record):
        self.payloads.append(json.loads(self.format(record)))

    def by_message(self, message):
        matches = [p for p in self.payloads if p["message"] == message]
        assert matches, f"no log line {message!r} in {self.payloads}"
        return matches[0]


class _CapturingTestCase(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.old_level = self.root.level
        self.handler = _ListHandler()
        self.root.addHandler(self.handler)
        self.root.setLevel(logging.DEBUG)

    def tearDown(self):
        self.root.removeHandler(self.handler)
        self.root.setLevel(self.old_level)


class TestResolvedStarlette(unittest.TestCase):
    def test_starlette_version_is_the_pinned_one(self):
        # pyproject pins starlette 1.0.1; the propagation tests below are
        # evidence for this version specifically.
        self.assertEqual(starlette.__version__, "1.0.1")

    def test_server_app_registers_request_context_middleware(self):
        self.assertTrue(
            any(m.cls is RequestContextMiddleware for m in app.user_middleware)
        )


class TestMiddlewareBoundary(_CapturingTestCase):
    def _app(self):
        mini = FastAPI()
        configure_request_context_middleware(mini)
        log = logging.getLogger("dda.test.route")

        @mini.get("/async")
        async def async_route():
            log.info("async handler")
            return current_context().get("request_id")

        @mini.get("/sync")
        def sync_route():
            log.info("sync handler")
            return current_context().get("request_id")

        @mini.get("/boom")
        async def boom_route():
            raise RuntimeError("handler failed")

        @mini.get("/missing")
        async def missing_route():
            raise HTTPException(status_code=404, detail="nope")

        return mini

    def test_one_fresh_request_id_per_request(self):
        client = TestClient(self._app())
        first = client.get("/async").json()
        second = client.get("/sync").json()

        self.assertRegex(first, UUID4)
        self.assertRegex(second, UUID4)
        self.assertNotEqual(first, second)
        self.assertEqual(self.handler.by_message("async handler")["request_id"], first)
        self.assertEqual(self.handler.by_message("sync handler")["request_id"], second)
        self.assertEqual(dict(current_context()), {})

    def test_unhandled_exception_logged_once_with_request_id(self):
        # raise_server_exceptions defaults to True: nothing may escape
        response = TestClient(self._app()).get("/boom")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")
        errors = [p for p in self.handler.payloads if p["level"] == "ERROR"]
        self.assertEqual(len(errors), 1, errors)
        self.assertEqual(errors[0]["message"], "Unhandled error processing GET /boom")
        self.assertRegex(errors[0]["request_id"], UUID4)
        self.assertEqual(errors[0]["thrown"]["name"], "RuntimeError")
        self.assertEqual(dict(current_context()), {})

    def test_handled_errors_are_left_to_fastapi(self):
        response = TestClient(self._app()).get("/missing")

        self.assertEqual(response.status_code, 404)
        self.assertFalse([p for p in self.handler.payloads if p["level"] == "ERROR"])


class TestProcessRecordCorrelation(_CapturingTestCase):
    """POST /process_record end to end, with the supervisor and Elasticsearch
    stubbed by functions that log from where the real ones run."""

    def setUp(self):
        super().setUp()
        for attr in (
            "tokenizer",
            "embedding_model",
            "nli_tokenizer",
            "nli_model",
            "client",
            "index",
            "llm_client",
        ):
            setattr(app.state, attr, MagicMock())
        app.dependency_overrides[api_key_auth] = lambda: "key"
        app.dependency_overrides[ensure_ready] = lambda: None
        self.addCleanup(app.dependency_overrides.clear)

    def test_request_id_reaches_worker_thread_and_background_task(self):
        log = logging.getLogger("dda.test.deep")

        def fake_search(self, body, client=None, index=None):
            log.info("search stored data")
            return {}, []

        def fake_execute(self, body):
            log.info("supervisor execute")
            self.response = {"summaries": {"ai:description": "x"}}

        def fake_store(data, client, index):
            log.info("store ai data")

        with patch(
            "data_discovery_ai.core.routes.SupervisorAgent.search_stored_data",
            fake_search,
        ), patch(
            "data_discovery_ai.core.routes.SupervisorAgent.execute", fake_execute
        ), patch(
            "data_discovery_ai.core.routes.SupervisorAgent.is_valid_request",
            return_value=True,
        ), patch(
            "data_discovery_ai.core.routes.SupervisorAgent.process_request_response",
            return_value={},
        ), patch(
            "data_discovery_ai.core.routes.store_ai_generated_data", fake_store
        ):
            response = TestClient(app).post(
                "/api/v1/ml/process_record",
                json={"uuid": "u-1", "selected_model": ["description_formatting"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)

        ids = {
            self.handler.by_message(m).get("request_id")
            for m in ("search stored data", "supervisor execute", "store ai data")
        }
        self.assertEqual(len(ids), 1, ids)
        self.assertRegex(ids.pop(), UUID4)
        self.assertEqual(dict(current_context()), {})

    def test_supervisor_failure_is_logged_with_thrown_and_request_id(self):
        def fake_search(self, body, client=None, index=None):
            return {}, []

        def failing_execute(self, body):
            raise RuntimeError("llm timeout")

        with patch(
            "data_discovery_ai.core.routes.SupervisorAgent.search_stored_data",
            fake_search,
        ), patch(
            "data_discovery_ai.core.routes.SupervisorAgent.execute", failing_execute
        ), patch(
            "data_discovery_ai.core.routes.SupervisorAgent.is_valid_request",
            return_value=True,
        ):
            response = TestClient(app).post(
                "/api/v1/ml/process_record",
                json={"uuid": "u-2", "selected_model": ["description_formatting"]},
            )

        # the client still gets the SSE error event, as before
        self.assertIn(
            "event: error\ndata: Processing failed: llm timeout", response.text
        )
        failed = self.handler.by_message("Processing failed for record u-2")
        self.assertEqual(failed["level"], "ERROR")
        self.assertRegex(failed["request_id"], UUID4)
        self.assertEqual(failed["thrown"]["name"], "RuntimeError")
        self.assertEqual(failed["thrown"]["message"], "llm timeout")
