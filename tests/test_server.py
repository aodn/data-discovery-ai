import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from data_discovery_ai import server

UP_COMPONENT = {"status": "UP", "detail": None}


@patch(
    "data_discovery_ai.utils.health_utils.check_llm",
    new=AsyncMock(return_value=UP_COMPONENT),
)
@patch(
    "data_discovery_ai.utils.health_utils.check_keyword_resources",
    new=MagicMock(return_value=UP_COMPONENT),
)
class TestServerLifespan(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.health_json = os.path.join(self.tmp_dir.name, "status", "health.json")
        self.health_patch = patch(
            "data_discovery_ai.utils.health_utils.HEALTH_JSON", self.health_json
        )
        self.health_patch.start()

        self.app = FastAPI()
        self.app.state.model_status = "STARTING"
        self.app.state.model_error = None

    def tearDown(self):
        self.health_patch.stop()
        self.tmp_dir.cleanup()

    def read_health(self):
        with open(self.health_json) as f:
            return json.load(f)

    @patch("data_discovery_ai.server.load_nli_tokenizer_model")
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_load_models_background_success(self, mock_embedding, mock_nli):
        mock_embedding.return_value = ("tokenizer", "embedding_model")
        mock_nli.return_value = ("nli_tokenizer", "nli_model")

        await server.load_models_background(self.app)

        self.assertEqual(self.app.state.model_status, "UP")
        self.assertEqual(self.app.state.tokenizer, "tokenizer")
        self.assertEqual(self.app.state.embedding_model, "embedding_model")
        self.assertEqual(self.app.state.nli_tokenizer, "nli_tokenizer")
        self.assertEqual(self.app.state.nli_model, "nli_model")
        health = self.read_health()
        self.assertEqual(health["status"], "UP")
        self.assertEqual(health["status_code"], 200)

    @patch("data_discovery_ai.server.load_nli_tokenizer_model")
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_load_models_background_failure_does_not_raise(
        self, mock_embedding, mock_nli
    ):
        mock_embedding.side_effect = OSError("connection reset")

        await server.load_models_background(self.app)

        self.assertEqual(self.app.state.model_status, "DOWN")
        self.assertIn("connection reset", self.app.state.model_error)
        mock_nli.assert_not_called()
        health = self.read_health()
        self.assertEqual(health["status"], "DOWN")
        self.assertEqual(health["status_code"], 200)

    @patch("data_discovery_ai.server.load_llm_client", return_value=MagicMock())
    @patch(
        "data_discovery_ai.server.create_es_index", return_value=(MagicMock(), "idx")
    )
    @patch(
        "data_discovery_ai.server.load_nli_tokenizer_model",
        return_value=("nli_tokenizer", "nli_model"),
    )
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_lifespan_does_not_wait_for_models(
        self, mock_embedding, mock_nli, mock_es, mock_llm
    ):
        release = asyncio.Event()
        loop = asyncio.get_running_loop()

        def slow_load():
            # block the loader thread until the test releases it
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            return "tokenizer", "embedding_model"

        mock_embedding.side_effect = slow_load

        async with server.lifespan(self.app):
            # startup completed while the models are still loading
            self.assertEqual(self.app.state.model_status, "STARTING")
            self.assertIsNone(self.app.state.tokenizer)
            self.assertEqual(self.read_health()["status"], "STARTING")
            release.set()

        # health file is removed on shutdown
        self.assertFalse(os.path.exists(self.health_json))

    @patch(
        "data_discovery_ai.server.create_es_index",
        side_effect=ConnectionError("es unreachable"),
    )
    async def test_lifespan_startup_failure_removes_health_file(self, mock_es):
        with self.assertRaises(ConnectionError):
            async with server.lifespan(self.app):
                pass
        self.assertFalse(os.path.exists(self.health_json))


if __name__ == "__main__":
    unittest.main()
