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
        self.health_writer = AsyncMock()
        self.health_writer_patcher = patch.object(
            server, "write_health_file", self.health_writer
        )
        self.health_remover = MagicMock()
        self.health_remover_patcher = patch.object(
            server, "remove_health_file", self.health_remover
        )
        self.health_writer_patcher.start()
        self.health_remover_patcher.start()
        self.addCleanup(self.health_writer_patcher.stop)
        self.addCleanup(self.health_remover_patcher.stop)

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
        self.health_writer.assert_awaited_once_with(self.app)

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
        self.health_writer.assert_awaited_once_with(self.app)

    @patch("data_discovery_ai.server.create_es_index")
    async def test_setup_elasticsearch_background_success(self, mock_es):
        es_client = MagicMock()
        mock_es.return_value = (es_client, "idx")

        await server.setup_elasticsearch_background(self.app)

        self.assertEqual(self.app.state.es_status, "UP")
        self.assertIsNone(self.app.state.es_error)
        self.assertIs(self.app.state.client, es_client)
        self.assertEqual(self.app.state.index, "idx")
        self.health_writer.assert_awaited_once_with(self.app)

    @patch("data_discovery_ai.server.create_es_index")
    async def test_setup_elasticsearch_background_reports_down_after_retries(
        self, mock_es
    ):
        mock_es.return_value = (None, None)

        await server.setup_elasticsearch_background(self.app)

        mock_es.assert_called_once_with()
        self.assertEqual(self.app.state.es_status, "DOWN")
        self.assertIn("failed after startup retries", self.app.state.es_error)
        self.health_writer.assert_awaited_once_with(self.app)

    @patch(
        "data_discovery_ai.server.create_es_index",
        side_effect=FileNotFoundError("schema missing"),
    )
    async def test_setup_elasticsearch_background_missing_schema_stops(self, mock_es):
        await server.setup_elasticsearch_background(self.app)

        self.assertEqual(self.app.state.es_status, "DOWN")
        self.assertIn("schema missing", self.app.state.es_error)
        mock_es.assert_called_once_with()
        self.health_writer.assert_awaited_once_with(self.app)

    @patch("data_discovery_ai.server.load_llm_client", return_value=MagicMock())
    @patch("data_discovery_ai.server.setup_elasticsearch_background", new_callable=AsyncMock)
    @patch("data_discovery_ai.server.load_models_background", new_callable=AsyncMock)
    async def test_lifespan_publishes_starting_health_and_removes_it_on_exit(
        self, mock_models, mock_es, mock_llm
    ):
        async with server.lifespan(self.app):
            self.assertEqual(self.app.state.model_status, "STARTING")
            self.assertEqual(self.app.state.es_status, "STARTING")
            self.health_writer.assert_awaited_once_with(self.app)
            self.health_remover.assert_not_called()

        self.health_remover.assert_called_once_with()

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
