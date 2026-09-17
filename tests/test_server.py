import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from data_discovery_ai import server


class TestServerLifespan(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.state.model_status = "STARTING"
        self.app.state.model_error = None
        self.app.state.es_status = "STARTING"
        self.app.state.es_error = None

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

    @patch("data_discovery_ai.server.create_es_index")
    async def test_setup_elasticsearch_background_success(self, mock_es):
        es_client = MagicMock()
        mock_es.return_value = (es_client, "idx")

        await server.setup_elasticsearch_background(self.app)

        self.assertEqual(self.app.state.es_status, "UP")
        self.assertIsNone(self.app.state.es_error)
        self.assertIs(self.app.state.client, es_client)
        self.assertEqual(self.app.state.index, "idx")

    @patch("data_discovery_ai.server.asyncio.sleep", new_callable=AsyncMock)
    @patch("data_discovery_ai.server.create_es_index")
    async def test_setup_elasticsearch_background_retries_until_success(
        self, mock_es, mock_sleep
    ):
        es_client = MagicMock()
        mock_es.side_effect = [
            (None, None),
            ConnectionError("x"),
            (es_client, "idx"),
        ]

        await server.setup_elasticsearch_background(self.app)

        self.assertEqual(mock_es.call_count, 3)
        self.assertEqual(mock_sleep.await_count, 2)
        self.assertEqual(self.app.state.es_status, "UP")
        self.assertIsNone(self.app.state.es_error)
        self.assertIs(self.app.state.client, es_client)
        self.assertEqual(self.app.state.index, "idx")

    @patch("data_discovery_ai.server.asyncio.sleep", new_callable=AsyncMock)
    @patch(
        "data_discovery_ai.server.create_es_index",
        side_effect=FileNotFoundError("schema missing"),
    )
    async def test_setup_elasticsearch_background_missing_schema_stops(
        self, mock_es, mock_sleep
    ):
        await server.setup_elasticsearch_background(self.app)

        self.assertEqual(self.app.state.es_status, "DOWN")
        self.assertIn("schema missing", self.app.state.es_error)
        mock_es.assert_called_once_with()
        mock_sleep.assert_not_awaited()

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
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def slow_load():
            # block the loader thread until the test releases it
            loop.call_soon_threadsafe(started.set)
            release.wait()
            return "tokenizer", "embedding_model"

        mock_embedding.side_effect = slow_load

        async with server.lifespan(self.app):
            await started.wait()
            # startup completed while the models are still loading
            self.assertEqual(self.app.state.model_status, "STARTING")
            self.assertIsNone(self.app.state.tokenizer)
            release.set()

    @patch("data_discovery_ai.server.load_llm_client", return_value=MagicMock())
    @patch(
        "data_discovery_ai.server.load_nli_tokenizer_model",
        return_value=("nli_tokenizer", "nli_model"),
    )
    @patch(
        "data_discovery_ai.server.load_embedding_tokenizer_model",
        return_value=("tokenizer", "embedding_model"),
    )
    @patch(
        "data_discovery_ai.server.create_es_index",
    )
    async def test_lifespan_does_not_wait_for_elasticsearch(
        self, mock_es, mock_embedding, mock_nli, mock_llm
    ):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def slow_setup():
            loop.call_soon_threadsafe(started.set)
            release.wait()
            return MagicMock(), "idx"

        mock_es.side_effect = slow_setup

        async with server.lifespan(self.app):
            await started.wait()
            self.assertEqual(self.app.state.es_status, "STARTING")
            self.assertIsNone(self.app.state.client)
            release.set()


if __name__ == "__main__":
    unittest.main()
